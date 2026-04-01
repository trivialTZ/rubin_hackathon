"""Baseline early-epoch classifier.

Uses LightGBM gradient-boosted trees.  Key advantages over the previous
RandomForest implementation:
  - Native NaN handling (no median imputation — LightGBM learns optimal
    split direction for missing values, critical when Fink/Lasair scores
    are sparse).
  - Leaf-wise tree growth captures broker-score × lightcurve interactions
    more effectively than depth-wise RF.
  - Early stopping prevents overfitting when a validation set is provided.

References:
  Ke et al. (2017) "LightGBM: A Highly Efficient Gradient Boosting
  Decision Tree", NeurIPS.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from pandas.api.types import is_numeric_dtype

LABEL_NAMES = ["snia", "nonIa_snlike", "other"]
LABEL_MAP = {label: idx for idx, label in enumerate(LABEL_NAMES)}
N_CLASSES = len(LABEL_NAMES)

DEFAULT_FEATURES = [
    "n_det",
    "n_det_g",
    "n_det_r",
    "t_since_first",
    "t_baseline",
    "mag_first",
    "mag_last",
    "mag_min",
    "mag_mean",
    "mag_std",
    "mag_range",
    "dmag_dt",
    "dmag_first_last",
    "color_gr",
    "color_gr_slope",
    "mean_rb",
    "alerce_SNIa",
    "alerce_SNIbc",
    "alerce_SNII",
    "alerce_SLSN",
    "alerce_top_Transient",
    "alerce_stamp_SN",
    "fink_rf_snia",
    "fink_snn_snia",
    "fink_snn_sn",
    "snn_prob_ia",
    "parsnip_prob_ia",
]


class EarlyMetaClassifier:
    def __init__(
        self,
        *,
        n_estimators: int = 1000,
        random_state: int = 42,
        feature_cols: list[str] | None = None,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)
        self.feature_cols = feature_cols[:] if feature_cols is not None else None
        self.training_weight_summary: dict[str, Any] | None = None
        self._calibrator = None
        self._fill_values: dict[str, float] = {}  # kept for save/load compat
        self._model = None
        self._class_labels = LABEL_NAMES[:]

    def fit(self, df, df_eval=None) -> "EarlyMetaClassifier":
        """Train the LightGBM multiclass classifier.

        Parameters
        ----------
        df : DataFrame
            Training epoch table (must contain ``target_label`` column).
        df_eval : DataFrame, optional
            Held-out validation data for early stopping.  When provided,
            training stops if ``multi_logloss`` does not improve for 50
            consecutive rounds, and the best iteration is kept.
        """
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation

        labelled = df[df["target_label"].notna()].copy()
        if len(labelled) == 0:
            raise ValueError("No labelled rows available for baseline training")

        self.feature_cols = self._resolve_feature_cols(labelled)
        X = self._prepare_features(labelled)
        y = labelled["target_label"].map(LABEL_MAP).astype(int).to_numpy()
        weights = self._object_balanced_weights(labelled)

        self._model = LGBMClassifier(
            objective="multiclass",
            num_class=N_CLASSES,
            n_estimators=self.n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=5,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=self.random_state,
            verbose=-1,
            importance_type="gain",
        )

        fit_kwargs: dict[str, Any] = {"sample_weight": weights}
        callbacks = [log_evaluation(0)]

        if df_eval is not None:
            labelled_eval = df_eval[df_eval["target_label"].notna()]
            if len(labelled_eval) > 0:
                X_eval = self._prepare_features(labelled_eval)
                y_eval = labelled_eval["target_label"].map(LABEL_MAP).astype(int).to_numpy()
                fit_kwargs["eval_set"] = [(X_eval, y_eval)]
                callbacks.append(early_stopping(50))

        fit_kwargs["callbacks"] = callbacks
        self._model.fit(X, y, **fit_kwargs)
        return self

    def predict_proba(self, df, *, calibrated: bool = True) -> np.ndarray:
        if self._model is None or self.feature_cols is None:
            raise RuntimeError("Model not fit")
        X = self._prepare_features(df)
        probs = self._model.predict_proba(X)
        classes = list(getattr(self._model, "classes_", []))
        full = np.zeros((len(X), N_CLASSES), dtype=float)
        for src_idx, cls in enumerate(classes):
            full[:, int(cls)] = probs[:, src_idx]
        row_sums = full.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        full /= row_sums
        if calibrated and self._calibrator is not None:
            return self._calibrator.transform(full)
        return full

    def feature_importances(self) -> dict[str, float]:
        """Return feature name → importance (gain) mapping."""
        if self._model is None or self.feature_cols is None:
            return {}
        importances = self._model.feature_importances_
        return {
            name: float(imp)
            for name, imp in zip(self.feature_cols, importances)
        }

    def set_calibrator(self, calibrator) -> None:
        self._calibrator = calibrator

    def score_object_at_n_det(self, object_id: str, n_det: int, df) -> dict[str, Any]:
        row_df = df[(df["object_id"] == object_id) & (df["n_det"].astype(int) == int(n_det))]
        if len(row_df) == 0:
            raise ValueError(f"No row found for {object_id} at n_det={n_det}")
        row = row_df.iloc[[0]]
        probs = self.predict_proba(row, calibrated=True)[0]
        pred_idx = int(np.argmax(probs))
        return {
            "object_id": object_id,
            "n_det": int(n_det),
            "alert_mjd": float(row.iloc[0].get("alert_mjd", np.nan)) if row.iloc[0].get("alert_mjd") is not None else None,
            "p_snia": float(probs[LABEL_MAP["snia"]]),
            "p_nonIa_snlike": float(probs[LABEL_MAP["nonIa_snlike"]]),
            "p_other": float(probs[LABEL_MAP["other"]]),
            "predicted_class": LABEL_NAMES[pred_idx],
            "follow_up_priority": float(probs[LABEL_MAP["snia"]]),
        }

    def save(self, model_dir: Path) -> Path:
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / "early_meta.pkl"
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "n_estimators": self.n_estimators,
                    "random_state": self.random_state,
                    "feature_cols": self.feature_cols,
                    "training_weight_summary": self.training_weight_summary,
                    "fill_values": self._fill_values,
                    "calibrator": self._calibrator,
                    "model": self._model,
                },
                fh,
            )
        return path

    @classmethod
    def load(cls, model_dir: Path) -> "EarlyMetaClassifier":
        with open(Path(model_dir) / "early_meta.pkl", "rb") as fh:
            payload = pickle.load(fh)
        clf = cls(
            n_estimators=payload["n_estimators"],
            random_state=payload["random_state"],
            feature_cols=payload["feature_cols"],
        )
        clf.training_weight_summary = payload["training_weight_summary"]
        clf._fill_values = payload.get("fill_values", {})
        clf._calibrator = payload["calibrator"]
        clf._model = payload["model"]
        return clf

    def _resolve_feature_cols(self, df) -> list[str]:
        import warnings

        numeric_cols = [
            column
            for column in df.columns
            if column not in {"object_id", "target_label"}
            and is_numeric_dtype(df[column])
        ]
        preferred = [column for column in DEFAULT_FEATURES if column in numeric_cols]
        missing = [column for column in DEFAULT_FEATURES if column not in numeric_cols]
        if missing:
            warnings.warn(
                f"EarlyMetaClassifier: {len(missing)} expected features missing from "
                f"epoch table: {missing}. These will be filled with NaN.",
                stacklevel=2,
            )
        if not preferred:
            warnings.warn(
                "EarlyMetaClassifier: NONE of the expected DEFAULT_FEATURES found. "
                f"Falling back to all {len(numeric_cols)} numeric columns. "
                "This likely means the epoch table was built incorrectly.",
                stacklevel=2,
            )
            return sorted(numeric_cols)
        return preferred

    def _prepare_features(self, df) -> np.ndarray:
        """Build the feature matrix.  LightGBM handles NaN natively."""
        import pandas as pd

        if self.feature_cols is None:
            raise RuntimeError("Feature columns unavailable")
        frame = pd.DataFrame(df).copy()
        for column in self.feature_cols:
            if column not in frame.columns:
                frame[column] = np.nan
        frame = frame[self.feature_cols].apply(pd.to_numeric, errors="coerce")
        return frame.to_numpy(dtype=float)

    def _object_balanced_weights(self, df) -> np.ndarray:
        object_labels = df.groupby("object_id")["target_label"].first()
        class_counts = object_labels.value_counts().sort_index().to_dict()
        n_objects = int(len(object_labels))
        n_classes = max(1, len(class_counts))
        object_weights: dict[str, float] = {}
        for object_id, label in object_labels.items():
            count = int(class_counts[str(label)])
            object_weights[str(object_id)] = n_objects / (n_classes * count)
        weights = df["object_id"].map(object_weights).astype(float).to_numpy()
        self.training_weight_summary = {
            "n_objects": n_objects,
            "class_object_counts": {str(key): int(value) for key, value in class_counts.items()},
            "mean_weight": float(np.mean(weights)) if len(weights) > 0 else 0.0,
        }
        return weights
