"""Trust-aware follow-up proxy model."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def _numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    blocked = {"object_id", "target_follow_proxy", "target_class", "label_source", "label_quality"}
    return sorted(
        column
        for column in df.columns
        if column not in blocked
        and is_numeric_dtype(df[column])
        # Exclude absolute timestamps — won't generalize to future observations
        and column != "alert_jd"
        and not column.startswith("source_event_time_jd__")
        # Exclude association metadata — not a classification signal
        and column != "lightcurve_association_sep_arcsec"
    )


def _prepare_frame(df: pd.DataFrame, feature_cols: list[str], fill_values: dict[str, float] | None = None):
    frame = df.copy()
    for column in feature_cols:
        if column not in frame.columns:
            frame[column] = np.nan
    numeric = frame[feature_cols].apply(pd.to_numeric, errors="coerce")
    # LightGBM handles NaN natively — no imputation needed.
    if fill_values is None:
        fill_values = {}
    return numeric, fill_values


def _fit_binary_model(X: pd.DataFrame, y: np.ndarray, *, n_estimators: int = 500, n_jobs: int = 1):
    from lightgbm import LGBMClassifier

    unique = np.unique(y)
    if len(unique) < 2:
        return {"kind": "constant", "prob": float(unique[0]) if len(unique) else 0.0}
    model = LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=5,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        reg_alpha=0.1,
        reg_lambda=0.1,
        is_unbalance=True,
        random_state=42,
        n_jobs=n_jobs,
        verbose=-1,
    )
    model.fit(X.to_numpy(dtype=float), y)
    return {"kind": "sklearn", "model": model}  # sklearn-compat API


def _predict_binary_model(model_bundle, X: pd.DataFrame) -> np.ndarray:
    if model_bundle["kind"] == "constant":
        return np.full(len(X), float(model_bundle["prob"]), dtype=float)
    model = model_bundle["model"]
    probs = model.predict_proba(X.to_numpy(dtype=float))
    if probs.shape[1] == 1:
        cls = int(model.classes_[0])
        return np.full(len(X), float(cls), dtype=float)
    class_index = list(model.classes_).index(1)
    return probs[:, class_index].astype(float)


@dataclass
class FollowupArtifact:
    feature_cols: list[str]
    fill_values: dict[str, float]
    model_bundle: Any

    def predict_followup(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = _prepare_frame(df, self.feature_cols, fill_values=self.fill_values)
        return _predict_binary_model(self.model_bundle, X)

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "model.pkl", "wb") as fh:
            pickle.dump(self.model_bundle, fh)
        with open(model_dir / "metadata.json", "w") as fh:
            json.dump(
                {
                    "feature_cols": self.feature_cols,
                    "fill_values": self.fill_values,
                },
                fh,
                indent=2,
            )

    @classmethod
    def load(cls, model_dir: Path) -> "FollowupArtifact":
        with open(model_dir / "metadata.json") as fh:
            metadata = json.load(fh)
        with open(model_dir / "model.pkl", "rb") as fh:
            model_bundle = pickle.load(fh)
        return cls(
            feature_cols=[str(column) for column in metadata["feature_cols"]],
            fill_values={str(key): float(value) for key, value in metadata["fill_values"].items()},
            model_bundle=model_bundle,
        )


def train_followup_model(
    snapshot_df: pd.DataFrame,
    *,
    split,
    model_dir: Path,
    n_estimators: int = 200,
    n_jobs: int = 1,
):
    from sklearn.metrics import brier_score_loss, roc_auc_score

    labelled = snapshot_df[snapshot_df["target_follow_proxy"].notna()].copy()
    feature_cols = _numeric_feature_cols(labelled)

    # Train on train+cal (cal was only needed by expert_trust for OOF); evaluate on test
    train_cal_ids = split.train_ids | split.cal_ids
    train_df = labelled[labelled["object_id"].isin(train_cal_ids)].copy()
    test_df = labelled[labelled["object_id"].isin(split.test_ids)].copy()
    if len(train_df) == 0:
        raise ValueError("No follow-up training rows available")

    X_train, fill_values = _prepare_frame(train_df, feature_cols)
    y_train = train_df["target_follow_proxy"].astype(int).to_numpy()
    model_bundle = _fit_binary_model(X_train, y_train, n_estimators=n_estimators, n_jobs=n_jobs)

    artifact = FollowupArtifact(feature_cols=feature_cols, fill_values=fill_values, model_bundle=model_bundle)
    artifact.save(model_dir)

    X_test, _ = _prepare_frame(test_df, feature_cols, fill_values=fill_values)
    y_test = test_df["target_follow_proxy"].astype(int).to_numpy()
    y_prob = _predict_binary_model(model_bundle, X_test)

    metrics: dict[str, Any] = {"n_rows": int(len(test_df))}
    if len(test_df) > 0:
        metrics["positive_rate"] = float(np.mean(y_test))
        metrics["brier"] = float(brier_score_loss(y_test, y_prob))
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["positive_rate"] = None
        metrics["brier"] = None
        metrics["roc_auc"] = None

    return artifact, metrics
