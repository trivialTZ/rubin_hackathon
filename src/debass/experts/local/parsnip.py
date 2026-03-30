"""ParSNIP local expert wrapper.

Generative model producing well-calibrated multiclass posteriors.
Slower than SuperNNova but richer taxonomy (6 SN classes).

Requires: pip install parsnip
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import LocalExpert, ExpertOutput

_DEFAULT_MODEL_DIR = Path("artifacts/local_experts/parsnip")


class ParSNIPExpert(LocalExpert):
    name = "parsnip"
    semantic_type = "probability"
    CLASSES = ["SN Ia", "SN II", "SN IIn", "SN Ibc", "SN IIb", "SLSN-I"]

    def __init__(self, model_dir: Path = _DEFAULT_MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        self._model = None
        self._parsnip_available = False
        self._load_model()

    def _load_model(self) -> None:
        try:
            import parsnip  # noqa: F401
            self._parsnip_available = True
        except ImportError:
            self._parsnip_available = False
        weights = self.model_dir / "model.pt"
        if self._parsnip_available and weights.exists():
            try:
                import parsnip
                self._model = parsnip.load_model(str(weights))
            except Exception:
                self._model = None

    # ------------------------------------------------------------------ #
    # Interface                                                            #
    # ------------------------------------------------------------------ #

    def fit(self, lightcurves: Any, labels: Any) -> None:
        if not self._parsnip_available:
            raise RuntimeError("parsnip is not installed")
        raise NotImplementedError(
            "ParSNIP training should be run via scripts/train.py on SCC. "
            "Place trained model weights in artifacts/local_experts/parsnip/."
        )

    def predict_epoch(self, object_id: str, lightcurve: Any, epoch_jd: float) -> ExpertOutput:
        truncated = self._truncate(lightcurve, epoch_jd)

        if self._model is not None and self._parsnip_available:
            probs = self._run_parsnip(object_id, truncated)
        else:
            probs = {cls: 1.0 / len(self.CLASSES) for cls in self.CLASSES}

        return ExpertOutput(
            expert=self.name,
            object_id=object_id,
            epoch_jd=epoch_jd,
            class_probabilities=probs,
            raw_output={"n_det": len(truncated) if isinstance(truncated, list) else 0},
            model_version="stub" if self._model is None else "loaded",
            available=self._parsnip_available,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "semantic_type": self.semantic_type,
            "classes": self.CLASSES,
            "model_dir": str(self.model_dir),
            "available": self._parsnip_available,
            "model_loaded": self._model is not None,
            "requires": "parsnip>=0.4, multi-band photometry",
            "epoch_aware": True,
            "calibration": "generative — well-calibrated by design",
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

    def _run_parsnip(self, object_id: str, lightcurve: Any) -> dict[str, float]:
        raise NotImplementedError("Load model weights first via fit() or model_dir")
