"""SuperNNova local expert wrapper.

Scores objects at arbitrary epochs by truncating the lightcurve
to detections up to epoch_jd before inference.

Requires:
  - ``pip install supernnova torch``
  - Trained model weights placed in ``artifacts/local_experts/supernnova/``

Expected artifact layout::

    artifacts/local_experts/supernnova/
        model.pt               # torch.save(model) — full RNN object
        cli_args.json          # (optional) training settings

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
    # Binary classification: Ia vs non-Ia
    CLASSES = ["SN Ia", "non-Ia"]

    def __init__(self, model_dir: Path = _DEFAULT_MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        self._snn_available = False
        self._model: Any = None
        self._settings: dict[str, Any] = {}
        self._device = "cpu"
        self._load_model()

    # ------------------------------------------------------------------ #
    # Model loading                                                       #
    # ------------------------------------------------------------------ #

    def _load_model(self) -> None:
        """Attempt to import SuperNNova and load trained weights."""
        try:
            import supernnova  # noqa: F401
            self._snn_available = True
        except ImportError:
            self._snn_available = False
            return

        try:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            # torch missing — can't load weights
            return

        # Load training settings if present
        settings_path = self.model_dir / "cli_args.json"
        if settings_path.exists():
            try:
                with open(settings_path) as fh:
                    self._settings = json.load(fh)
            except Exception:
                pass

        # Locate and load model weights
        model_file = self._find_model_file()
        if model_file is None:
            return  # Weights not placed yet — stay in stub mode

        try:
            import torch
            checkpoint = torch.load(
                model_file,
                map_location=self._device,
                weights_only=False,
            )
            # SuperNNova typically saves the full model object via torch.save(model)
            if hasattr(checkpoint, "eval"):
                self._model = checkpoint
                self._model.eval()
            else:
                # state_dict or other format — store as-is
                self._model = checkpoint
        except Exception as exc:
            print(f"[supernnova] WARNING: failed to load {model_file}: {exc}")
            self._model = None

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

        if self._model is not None and self._snn_available:
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
            model_version="stub" if self._model is None else "loaded",
            available=self._snn_available,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "semantic_type": self.semantic_type,
            "classes": self.CLASSES,
            "model_dir": str(self.model_dir),
            "available": self._snn_available,
            "model_loaded": self._model is not None,
            "inference_implemented": True,
            "requires": "PyTorch, supernnova>=2.0, trained model.pt in model_dir",
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
        """Run SuperNNova inference on a truncated lightcurve.

        Tries three strategies in order:
          1. SuperNNova's high-level ``classify_lcs`` API (v3+)
          2. Direct model forward pass on a formatted tensor
          3. Raises RuntimeError with diagnostic information
        """
        import torch

        dets = lightcurve if isinstance(lightcurve, list) else []
        if not dets:
            return {cls: 1.0 / len(self.CLASSES) for cls in self.CLASSES}

        df = self._lc_to_snn_dataframe(object_id, dets)

        # --- Strategy 1: SuperNNova high-level API (v3+) ---
        try:
            from supernnova.visualization.prediction_distribution import (
                classify_lcs,
            )
            result = classify_lcs(self._model, df, self._device)
            if result is not None:
                prob_ia = float(result) if not hasattr(result, "__len__") else float(result[0])
                return {"SN Ia": prob_ia, "non-Ia": 1.0 - prob_ia}
        except (ImportError, TypeError, Exception):
            pass

        # --- Strategy 2: direct forward pass ---
        try:
            probs = self._forward_pass(df)
            if probs is not None:
                return probs
        except Exception:
            pass

        raise RuntimeError(
            f"SuperNNova inference failed for {object_id}. "
            f"Ensure the model in {self.model_dir} is compatible with the "
            f"installed supernnova version. Expected: torch.save(model) object "
            f"with a forward(X, lengths) signature."
        )

    def _forward_pass(self, df: Any) -> dict[str, float] | None:
        """Direct RNN forward pass — works when the model was saved as a
        complete ``torch.save(model)`` object.

        SuperNNova RNNs expect input shaped (batch, seq_len, n_features).
        Features per timestep are [FLUXCAL, FLUXCALERR] per band, with
        per-lightcurve normalization.
        """
        import torch
        import numpy as np

        if not hasattr(self._model, "forward"):
            return None

        # Group by band and build feature matrix
        bands = sorted(df["FLT"].unique())
        band_to_idx = {b: i for i, b in enumerate(bands)}
        n_bands = len(bands)

        # Sort by MJD and normalize flux per-lightcurve
        df = df.sort_values("MJD").copy()
        flux_median = df["FLUXCAL"].median()
        flux_scale = max(df["FLUXCAL"].abs().max(), 1e-10)
        df["FLUXCAL_norm"] = (df["FLUXCAL"] - flux_median) / flux_scale
        df["FLUXCALERR_norm"] = df["FLUXCALERR"] / flux_scale

        # Build sequence: each timestep has (flux_b0, fluxerr_b0, flux_b1, ...)
        n_features = 2 * n_bands
        seq_len = len(df)
        X = np.zeros((1, seq_len, n_features), dtype=np.float32)

        for row_idx, (_, row) in enumerate(df.iterrows()):
            b_idx = band_to_idx[row["FLT"]]
            X[0, row_idx, 2 * b_idx] = row["FLUXCAL_norm"]
            X[0, row_idx, 2 * b_idx + 1] = row["FLUXCALERR_norm"]

        X_tensor = torch.tensor(X, device=self._device)
        lengths = torch.tensor([seq_len], dtype=torch.long, device=self._device)

        with torch.no_grad():
            # SuperNNova models accept (X, lengths) or just (X)
            try:
                output = self._model(X_tensor, lengths)
            except TypeError:
                output = self._model(X_tensor)

            if output.dim() == 1:
                logits = output.unsqueeze(0)
            else:
                logits = output

            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        if len(probs) >= 2:
            return {"SN Ia": float(probs[0]), "non-Ia": float(probs[1])}

        # Single-output model (sigmoid convention)
        p = float(probs) if probs.ndim == 0 else float(probs[0])
        return {"SN Ia": p, "non-Ia": 1.0 - p}
