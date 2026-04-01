"""SuperNNova local expert wrapper.

Scores objects at arbitrary epochs by truncating the lightcurve
to detections up to epoch_jd before inference.

Requires:
  - ``pip install supernnova torch``
  - Trained model **state_dict** + ``cli_args.json`` placed in
    ``artifacts/local_experts/supernnova/``

Expected artifact layout::

    artifacts/local_experts/supernnova/
        model.pt               # torch.save(state_dict) — as shipped by Fink
        cli_args.json          # training settings (REQUIRED by classify_lcs)
        data_norm.json         # normalization constants (REQUIRED)

The canonical way to obtain these artifacts is to sparse-clone
``astrolabsoftware/fink-science`` and copy the ``snn_snia_vs_nonia``
directory contents.

Inference delegates to SuperNNova's own ``classify_lcs`` API which:
  1. Reads ``cli_args.json`` to reconstruct the RNN architecture
  2. Loads the state_dict from ``model.pt``
  3. Runs inference on CPU or CUDA

Runs on **CPU** — no GPU queue required.

If ``supernnova`` is not installed or weights are missing, the wrapper
returns uniform priors (stub mode) — it never raises on missing deps.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .base import ExpertOutput, LocalExpert

_DEFAULT_MODEL_DIR = Path("artifacts/local_experts/supernnova")

# ZTF filter ID → SuperNNova band string
_FID_TO_BAND = {
    1: "g", 2: "r", 3: "i",
    "1": "g", "2": "r", "3": "i",
    "g": "g", "r": "r", "i": "i",
}


class SuperNNovaExpert(LocalExpert):
    name = "supernnova"
    semantic_type = "probability"
    requires_gpu = False  # Fink runs SNN on CPU; works fine
    # Binary classification: Ia vs non-Ia
    CLASSES = ["SN Ia", "non-Ia"]

    def __init__(self, model_dir: Path = _DEFAULT_MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        self._snn_available = False
        self._classify_lcs: Any = None  # reference to SNN's classify_lcs
        self._model_file: Path | None = None
        self._settings: dict[str, Any] = {}
        self._device = "cpu"
        self._load_model()

    # ------------------------------------------------------------------ #
    # Model loading                                                       #
    # ------------------------------------------------------------------ #

    def _load_model(self) -> None:
        """Verify SuperNNova is importable and artifacts are present.

        We do NOT load the model into memory here. SuperNNova's
        ``classify_lcs`` handles loading internally from the file path.
        """
        try:
            from supernnova.validation.validate_onthefly import (
                classify_lcs,
            )
            self._classify_lcs = classify_lcs
            self._snn_available = True
        except ImportError:
            self._snn_available = False
            return

        try:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return

        # Load training settings if present
        settings_path = self.model_dir / "cli_args.json"
        if settings_path.exists():
            try:
                with open(settings_path) as fh:
                    self._settings = json.load(fh)
            except Exception:
                pass

        # Check that required artifacts exist (don't load — classify_lcs does that)
        self._model_file = self._find_model_file()
        if self._model_file is None:
            return

        # Verify cli_args.json exists (required by classify_lcs)
        if not (self.model_dir / "cli_args.json").exists():
            print(
                f"[supernnova] WARNING: cli_args.json not found in {self.model_dir}. "
                f"classify_lcs requires it to reconstruct the model architecture."
            )
            self._model_file = None

    def _find_model_file(self) -> Path | None:
        """Locate model weights in the artifact directory."""
        direct = self.model_dir / "model.pt"
        if direct.exists():
            return direct
        for pattern in ("*.pt", "*.pth"):
            matches = sorted(self.model_dir.glob(f"**/{pattern}"))
            if matches:
                return matches[0]
        return None

    # ------------------------------------------------------------------ #
    # Interface                                                            #
    # ------------------------------------------------------------------ #

    def fit(self, lightcurves: Any, labels: Any) -> None:
        """Kick off SuperNNova training. In practice, call the SNN CLI or API."""
        if not self._snn_available:
            raise RuntimeError("supernnova is not installed")
        # Training is expensive; delegate to scripts/train.py → SCC job
        raise NotImplementedError(
            "SuperNNova training should be run via scripts/train.py on SCC. "
            "Place trained model weights in artifacts/local_experts/supernnova/."
        )

    def predict_epoch(self, object_id: str, lightcurve: Any, epoch_jd: float) -> ExpertOutput:
        """Score object using lightcurve truncated to epoch_jd."""
        truncated = self._truncate(lightcurve, epoch_jd)

        if self._model_file is not None and self._snn_available:
            probs = self._run_snn(object_id, truncated)
        else:
            # Stub: return uniform priors when model not loaded
            probs = {cls: 1.0 / len(self.CLASSES) for cls in self.CLASSES}

        return ExpertOutput(
            expert=self.name,
            object_id=object_id,
            epoch_jd=epoch_jd,
            class_probabilities=probs,
            raw_output={"truncated_n_det": len(truncated) if isinstance(truncated, list) else 0},
            model_version="stub" if self._model_file is None else "loaded",
            available=self._snn_available,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "semantic_type": self.semantic_type,
            "classes": self.CLASSES,
            "model_dir": str(self.model_dir),
            "available": self._snn_available,
            "model_loaded": self._model_file is not None,
            "inference_implemented": True,
            "requires": "PyTorch, supernnova>=3.0, model.pt + cli_args.json in model_dir",
            "epoch_aware": True,
            "device": self._device,
        }

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _truncate(self, lightcurve: Any, epoch_jd: float) -> Any:
        """Return only detections at or before epoch_jd.

        Cached lightcurves use MJD; epoch_jd is passed as JD from local_infer.py.
        Handles both mjd (ALeRCE cache) and jd (ZTF alert stream) keys.
        """
        if isinstance(lightcurve, list):
            def _det_jd(d: dict) -> float:
                if "jd" in d:
                    return float(d["jd"])
                if "mjd" in d:
                    return float(d["mjd"]) + 2400000.5
                return 0.0
            return [det for det in lightcurve if _det_jd(det) <= epoch_jd]
        try:
            import pandas as pd
            if isinstance(lightcurve, pd.DataFrame):
                if "jd" in lightcurve.columns:
                    return lightcurve[lightcurve["jd"] <= epoch_jd]
                if "mjd" in lightcurve.columns:
                    return lightcurve[lightcurve["mjd"] + 2400000.5 <= epoch_jd]
        except ImportError:
            pass
        return lightcurve

    def _lc_to_snn_dataframe(self, object_id: str, lightcurve: list[dict]) -> Any:
        """Convert DEBASS lightcurve to SuperNNova DataFrame format.

        SuperNNova expects columns: SNID, MJD, FLUXCAL, FLUXCALERR, FLT.
        Our cache stores magnitudes; this converts mag → flux at zeropoint=27.5
        (the SNANA convention that SuperNNova uses).
        """
        import pandas as pd

        rows: list[dict[str, Any]] = []
        for det in lightcurve:
            # --- time ---
            mjd = det.get("mjd")
            if mjd is None:
                jd = det.get("jd")
                if jd is not None:
                    mjd = float(jd) - 2400000.5
            if mjd is None:
                continue

            # --- flux (prefer native, else convert from mag) ---
            flux = det.get("flux")
            fluxerr = det.get("fluxerr")
            if flux is not None and fluxerr is not None:
                try:
                    flux, fluxerr = float(flux), float(fluxerr)
                except (TypeError, ValueError):
                    flux, fluxerr = None, None

            if flux is None or fluxerr is None:
                mag = det.get("magpsf")
                magerr = det.get("sigmapsf")
                if mag is None or magerr is None:
                    continue
                try:
                    mag, magerr = float(mag), float(magerr)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(mag) or not math.isfinite(magerr) or magerr <= 0:
                    continue
                zp = 27.5  # SNANA convention
                flux = 10 ** (-0.4 * (mag - zp))
                fluxerr = flux * (math.log(10) / 2.5) * magerr

            if not math.isfinite(flux) or not math.isfinite(fluxerr) or fluxerr <= 0:
                continue

            # --- filter ---
            fid = det.get("fid", 1)
            band = _FID_TO_BAND.get(fid, _FID_TO_BAND.get(str(fid), "r"))

            rows.append({
                "SNID": object_id,
                "MJD": float(mjd),
                "FLUXCAL": flux,
                "FLUXCALERR": fluxerr,
                "FLT": band,
            })

        if not rows:
            raise ValueError(f"No usable detections for {object_id} after SNN conversion")

        return pd.DataFrame(rows).sort_values("MJD").reset_index(drop=True)

    def _run_snn(self, object_id: str, lightcurve: Any) -> dict[str, float]:
        """Run SuperNNova inference via classify_lcs (the Fink/SNN v3 API).

        ``classify_lcs(df, model_path, device)`` handles everything:
          - Reads ``cli_args.json`` to reconstruct the RNN architecture
          - Loads the state_dict from ``model.pt``
          - Formats data, normalizes, runs forward pass
          - Returns (ids, pred_probs) where pred_probs shape = (N, 1, n_classes)
        """
        import numpy as np

        dets = lightcurve if isinstance(lightcurve, list) else []
        if not dets:
            return {cls: 1.0 / len(self.CLASSES) for cls in self.CLASSES}

        df = self._lc_to_snn_dataframe(object_id, dets)

        # classify_lcs expects the model FILE PATH, not a loaded object
        model_path = str(self._model_file)

        try:
            ids, pred_probs = self._classify_lcs(df, model_path, self._device)
        except Exception as exc:
            raise RuntimeError(
                f"SuperNNova classify_lcs failed for {object_id}: {exc}. "
                f"Artifacts in {self.model_dir}: model.pt, cli_args.json, data_norm.json "
                f"must all be present and consistent."
            ) from exc

        # pred_probs shape: (n_objects, num_inference_samples, n_classes)
        # For vanilla models num_inference_samples=1
        probs = pred_probs[0]  # first (only) object
        if probs.ndim > 1:
            probs = probs.mean(axis=0)  # average over inference samples

        if len(probs) >= 2:
            return {"SN Ia": float(probs[0]), "non-Ia": float(probs[1])}

        # Single-output fallback
        p = float(probs[0]) if hasattr(probs, '__len__') else float(probs)
        return {"SN Ia": p, "non-Ia": 1.0 - p}
