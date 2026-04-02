"""ALeRCE light-curve classifier — local expert wrapper.

Replaces the ALeRCE API for LC classification scores.  Runs on CPU
(LightGBM multiclass) so it slots into the CPU pipeline stage.
Consistent with all other DEBASS models: native NaN handling,
regularisation, is_unbalance=True.

Two feature-extraction tiers:
  1. **Full mode** — uses ``lc_classifier`` (turbofats + P4J + mhps)
     for ~50+ features, matching the published ALeRCE pipeline.
  2. **Lite fallback** — uses the 24 features already computed by
     ``debass_meta.features.lightcurve`` if ``lc_classifier`` is
     not installed.

Trained model artifact:
    artifacts/local_experts/alerce_lc/model.pkl   (LightGBM via train_alerce_lc.py)

Training script: ``scripts/train_alerce_lc.py``
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .base import ExpertOutput, LocalExpert

_DEFAULT_MODEL_DIR = Path("artifacts/local_experts/alerce_lc")

# ALeRCE LC classifier output classes
ALERCE_LC_CLASSES = [
    "SNIa", "SNIbc", "SNII", "SLSN",
    "AGN", "VS", "asteroid", "bogus",
]

# Transient sub-classes (sum → alerce_top_Transient)
_TRANSIENT_CLASSES = {"SNIa", "SNIbc", "SNII", "SLSN"}


class AlerceLCExpert(LocalExpert):
    name = "alerce_lc"
    semantic_type = "probability"
    requires_gpu = False
    CLASSES = ALERCE_LC_CLASSES

    def __init__(self, model_dir: Path = _DEFAULT_MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        self._model: Any = None
        self._feature_names: list[str] = []
        self._use_full_features = False
        self._lc_classifier_available = False
        self._load_model()

    # ------------------------------------------------------------------ #
    # Model loading                                                       #
    # ------------------------------------------------------------------ #

    def _load_model(self) -> None:
        # Check if lc_classifier is available for full feature extraction
        try:
            from lc_classifier.features import CustomHierarchicalExtractor  # noqa: F401
            self._lc_classifier_available = True
        except ImportError:
            self._lc_classifier_available = False

        model_path = self.model_dir / "model.pkl"
        if not model_path.exists():
            return
        try:
            with open(model_path, "rb") as fh:
                blob = pickle.load(fh)
            if isinstance(blob, dict):
                self._model = blob["model"]
                self._feature_names = blob.get("feature_names", [])
                self._use_full_features = blob.get("full_features", False)
            else:
                # Assume bare model — feature names unknown
                self._model = blob
                self._feature_names = []
        except Exception as exc:
            print(f"[alerce_lc] model load error: {exc}")
            self._model = None

    # ------------------------------------------------------------------ #
    # Feature extraction                                                   #
    # ------------------------------------------------------------------ #

    def _extract_features_lite(self, detections: list[dict]) -> dict[str, float]:
        """Extract the 24 built-in LC features (always available)."""
        from debass_meta.features.lightcurve import extract_features
        return extract_features(detections)

    def _extract_features_full(self, detections: list[dict]) -> dict[str, float]:
        """Extract ~50+ features using lc_classifier (turbofats/P4J/mhps).

        Falls back to lite features if lc_classifier is not installed or
        the lightcurve is too short for period-finding.
        """
        if not self._lc_classifier_available:
            return self._extract_features_lite(detections)

        try:
            import pandas as pd
            from lc_classifier.features import CustomHierarchicalExtractor

            # Build DataFrame in the format lc_classifier expects
            rows = []
            for det in detections:
                mjd = det.get("mjd") or det.get("jd", 0.0)
                mag = det.get("magpsf") or det.get("magpsf_corr")
                err = det.get("sigmapsf") or det.get("sigmapsf_corr", 0.1)
                fid = det.get("fid", 1)
                if mag is None:
                    continue
                rows.append({
                    "mjd": float(mjd),
                    "magpsf_corr": float(mag),
                    "sigmapsf_corr": float(err) if err else 0.1,
                    "fid": int(fid) if str(fid).isdigit() else 1,
                })

            if len(rows) < 3:
                return self._extract_features_lite(detections)

            df = pd.DataFrame(rows)
            extractor = CustomHierarchicalExtractor()
            features = extractor.compute_features(df)

            # Merge with lite features to fill any gaps
            lite = self._extract_features_lite(detections)
            for key, val in lite.items():
                if key not in features:
                    features[key] = val

            return features
        except Exception:
            return self._extract_features_lite(detections)

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def predict_epoch(
        self, object_id: str, lightcurve: Any, epoch_jd: float,
    ) -> ExpertOutput:
        # Truncate lightcurve to detections up to epoch_jd
        if isinstance(lightcurve, list):
            truncated = [
                d for d in lightcurve
                if (d.get("mjd") or d.get("jd") or 0.0) <= epoch_jd
            ]
        else:
            truncated = lightcurve

        # Stub mode if model not loaded
        if self._model is None:
            n_classes = len(self.CLASSES)
            stub_probs = {c: 1.0 / n_classes for c in self.CLASSES}
            return ExpertOutput(
                expert=self.name,
                object_id=object_id,
                epoch_jd=epoch_jd,
                class_probabilities=stub_probs,
                model_version="stub",
                available=False,
            )

        # Extract features
        if self._use_full_features and self._lc_classifier_available:
            feats = self._extract_features_full(truncated)
        else:
            feats = self._extract_features_lite(truncated)

        # Build feature vector in the same order the model was trained on
        if self._feature_names:
            X = np.array([[feats.get(f, np.nan) for f in self._feature_names]])
        else:
            # Fall back to all numeric features sorted by name
            X = np.array([[feats.get(f, np.nan) for f in sorted(feats.keys())]])

        try:
            proba = self._model.predict_proba(X)[0]
            classes = list(self._model.classes_)
            probs = {str(c): float(p) for c, p in zip(classes, proba)}
            # Ensure all ALERCE_LC_CLASSES are present
            for c in self.CLASSES:
                if c not in probs:
                    probs[c] = 0.0
        except Exception as exc:
            n_classes = len(self.CLASSES)
            probs = {c: 1.0 / n_classes for c in self.CLASSES}
            print(f"[alerce_lc] predict error for {object_id}: {exc}")

        return ExpertOutput(
            expert=self.name,
            object_id=object_id,
            epoch_jd=epoch_jd,
            class_probabilities=probs,
            model_version="full" if self._use_full_features else "lite",
            available=True,
        )

    def fit(self, lightcurves: Any, labels: Any) -> None:
        raise NotImplementedError("Use scripts/train_alerce_lc.py")

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self._model is not None,
            "model_loaded": self._model is not None,
            "model_dir": str(self.model_dir),
            "lc_classifier_installed": self._lc_classifier_available,
            "feature_mode": "full" if self._use_full_features else "lite",
            "n_features": len(self._feature_names),
            "classes": self.CLASSES,
            "requires_gpu": self.requires_gpu,
            "inference_implemented": True,
        }
