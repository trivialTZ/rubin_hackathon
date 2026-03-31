"""SuperNNova local expert wrapper.

Scores objects at arbitrary epochs by truncating the lightcurve
to detections up to epoch_jd before inference.

Requires: pip install supernnova (torch backend)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import LocalExpert, ExpertOutput

_DEFAULT_MODEL_DIR = Path("artifacts/local_experts/supernnova")


class SuperNNovaExpert(LocalExpert):
    name = "supernnova"
    semantic_type = "probability"
    # Output classes
    CLASSES = ["SN Ia", "non-Ia"]

    def __init__(self, model_dir: Path = _DEFAULT_MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            # SuperNNova exposes a Settings + TrainedModel object
            from supernnova.validation.validate_rnn import get_predictions  # noqa: F401
            self._snn_available = True
        except ImportError:
            self._snn_available = False
        # Actual model loading requires trained weights; stub returns None
        self._model = None  # replaced by trained weights in fit()

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
            "inference_implemented": False,
            "requires": "PyTorch, supernnova>=2.0",
            "epoch_aware": True,
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

    def _run_snn(self, object_id: str, lightcurve: Any) -> dict[str, float]:
        """Call SuperNNova inference. Implement when weights are available."""
        raise NotImplementedError("Load model weights first via fit() or model_dir")
