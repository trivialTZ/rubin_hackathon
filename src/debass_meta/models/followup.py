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

from .calibrate import IsotonicCalibrator


def _expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def _numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    blocked = {"object_id", "target_follow_proxy", "target_class", "label_source", "label_quality"}
    cols = []
    for column in df.columns:
        if column in blocked:
            continue
        if not is_numeric_dtype(df[column]):
            continue
        # Exclude absolute timestamps — won't generalize to future observations
        if column == "alert_jd" or column.startswith("source_event_time_jd__"):
            continue
        # Exclude association metadata — not a classification signal
        if column == "lightcurve_association_sep_arcsec":
            continue
        # Drop columns that are all-NaN or constant — zero signal
        if df[column].isna().all():
            continue
        if df[column].nunique(dropna=True) <= 1:
            continue
        cols.append(column)
    return sorted(cols)


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


def _fit_binary_model(X: pd.DataFrame, y: np.ndarray, *, n_estimators: int = 500, n_jobs: int = 1, sample_weight: np.ndarray | None = None):
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
    if sample_weight is not None:
        model.fit(X, y, sample_weight=sample_weight)
    else:
        model.fit(X, y)
    return {"kind": "sklearn", "model": model}  # sklearn-compat API


def _predict_binary_model(model_bundle, X: pd.DataFrame) -> np.ndarray:
    if model_bundle["kind"] == "constant":
        return np.full(len(X), float(model_bundle["prob"]), dtype=float)
    model = model_bundle["model"]
    probs = model.predict_proba(X)
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
    calibrator: Any = None  # Optional IsotonicCalibrator (Phase 0.1)

    def predict_followup(self, df: pd.DataFrame) -> np.ndarray:
        raw = self.predict_followup_raw(df)
        if self.calibrator is None:
            return raw
        return np.asarray(self.calibrator.transform(raw), dtype=float)

    def predict_followup_raw(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = _prepare_frame(df, self.feature_cols, fill_values=self.fill_values)
        return _predict_binary_model(self.model_bundle, X)

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "model.pkl", "wb") as fh:
            pickle.dump(self.model_bundle, fh)
        if self.calibrator is not None:
            with open(model_dir / "calibrator.pkl", "wb") as fh:
                pickle.dump(self.calibrator, fh)
        with open(model_dir / "metadata.json", "w") as fh:
            json.dump(
                {
                    "feature_cols": self.feature_cols,
                    "fill_values": self.fill_values,
                    "has_calibrator": self.calibrator is not None,
                    "calibrator_kind": (
                        getattr(self.calibrator, "name", None)
                        if self.calibrator is not None
                        else None
                    ),
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
        calibrator = None
        calibrator_path = model_dir / "calibrator.pkl"
        if calibrator_path.exists():
            with open(calibrator_path, "rb") as fh:
                calibrator = pickle.load(fh)
        return cls(
            feature_cols=[str(column) for column in metadata["feature_cols"]],
            fill_values={str(key): float(value) for key, value in metadata["fill_values"].items()},
            model_bundle=model_bundle,
            calibrator=calibrator,
        )


def train_followup_model(
    snapshot_df: pd.DataFrame,
    *,
    split,
    model_dir: Path,
    n_estimators: int = 200,
    n_jobs: int = 1,
    weak_weight: float = 1.0,
):
    from sklearn.metrics import brier_score_loss, roc_auc_score

    # --- Split integrity validation ---
    assert len(split.train_ids & split.cal_ids) == 0, "train/cal overlap"
    assert len(split.train_ids & split.test_ids) == 0, "train/test overlap"
    assert len(split.cal_ids & split.test_ids) == 0, "cal/test overlap"

    labelled = snapshot_df[snapshot_df["target_follow_proxy"].notna()].copy()
    feature_cols = _numeric_feature_cols(labelled)

    # Phase 0.2: Train on train-only. Cal reserved for isotonic calibration.
    train_df = labelled[labelled["object_id"].isin(split.train_ids)].copy()
    cal_df = labelled[labelled["object_id"].isin(split.cal_ids)].copy()
    test_df = labelled[labelled["object_id"].isin(split.test_ids)].copy()
    if len(train_df) == 0:
        raise ValueError("No follow-up training rows available")

    X_train, fill_values = _prepare_frame(train_df, feature_cols)
    y_train = train_df["target_follow_proxy"].astype(int).to_numpy()

    # Downweight weak / context labels to prevent them from dominating the decision
    # boundary at the Ia vs non-Ia cut. Weak rows still teach "not Ia", but with
    # reduced leverage so spectroscopic positives drive the hard boundary.
    if weak_weight != 1.0 and "label_quality" in train_df.columns:
        lq = train_df["label_quality"].astype(str)
        sample_weight = np.where(lq.isin(["weak", "context"]), float(weak_weight), 1.0).astype(float)
        n_weak = int((lq.isin(["weak", "context"])).sum())
        print(f"  weak_weight={weak_weight} applied to {n_weak}/{len(train_df)} train rows")
    else:
        sample_weight = None

    train_pos_rate = float(np.mean(y_train)) if len(y_train) > 0 else 0.0
    print(f"  followup: {len(train_df):,} train rows, {len(cal_df):,} cal, "
          f"{len(test_df):,} test, {len(feature_cols)} features, "
          f"train_pos_rate={train_pos_rate:.3f}")

    model_bundle = _fit_binary_model(X_train, y_train, n_estimators=n_estimators, n_jobs=n_jobs, sample_weight=sample_weight)

    # Phase 0.1: Fit IsotonicCalibrator on cal-set raw predictions vs cal y.
    calibrator: Any = None
    if len(cal_df) > 0:
        X_cal, _ = _prepare_frame(cal_df, feature_cols, fill_values=fill_values)
        y_cal = cal_df["target_follow_proxy"].astype(int).to_numpy()
        cal_raw = _predict_binary_model(model_bundle, X_cal)
        valid = ~np.isnan(cal_raw) & np.isin(y_cal, [0, 1])
        if valid.sum() >= 10 and len(np.unique(y_cal[valid])) == 2:
            calibrator = IsotonicCalibrator()
            calibrator.fit(cal_raw[valid], y_cal[valid])

    artifact = FollowupArtifact(
        feature_cols=feature_cols,
        fill_values=fill_values,
        model_bundle=model_bundle,
        calibrator=calibrator,
    )
    artifact.save(model_dir)

    # --- Train metrics (raw) ---
    y_train_pred = _predict_binary_model(model_bundle, X_train)
    train_metrics: dict[str, Any] = {
        "n_rows": int(len(train_df)),
        "positive_rate": float(np.mean(y_train)),
        "ece": _expected_calibration_error(y_train, y_train_pred),
    }
    try:
        train_metrics["roc_auc"] = float(roc_auc_score(y_train, y_train_pred))
    except ValueError:
        train_metrics["roc_auc"] = None
    train_metrics["brier"] = float(brier_score_loss(y_train, y_train_pred))

    # --- Test metrics: raw + calibrated ---
    X_test, _ = _prepare_frame(test_df, feature_cols, fill_values=fill_values)
    y_test = test_df["target_follow_proxy"].astype(int).to_numpy()
    y_test_raw = _predict_binary_model(model_bundle, X_test)
    if calibrator is not None and len(y_test_raw) > 0:
        y_test_cal = np.asarray(calibrator.transform(y_test_raw), dtype=float)
    else:
        y_test_cal = y_test_raw

    def _mk_test_metrics(y_prob_arr: np.ndarray) -> dict[str, Any]:
        m: dict[str, Any] = {"n_rows": int(len(test_df))}
        if len(test_df) == 0:
            m["positive_rate"] = None
            m["brier"] = None
            m["roc_auc"] = None
            m["ece"] = None
            return m
        m["positive_rate"] = float(np.mean(y_test))
        m["brier"] = float(brier_score_loss(y_test, y_prob_arr))
        m["ece"] = _expected_calibration_error(y_test, y_prob_arr)
        try:
            m["roc_auc"] = float(roc_auc_score(y_test, y_prob_arr))
        except ValueError:
            m["roc_auc"] = None
        return m

    test_raw_metrics = _mk_test_metrics(y_test_raw)
    test_cal_metrics = _mk_test_metrics(y_test_cal)

    metrics: dict[str, Any] = {
        "train": train_metrics,
        "test": test_raw_metrics,  # Back-compat: `test` = raw
        "test_raw": test_raw_metrics,
        "test_calibrated": test_cal_metrics,
        "has_calibrator": calibrator is not None,
        # Back-compat top-level keys from raw test
        "n_rows": test_raw_metrics["n_rows"],
        "positive_rate": test_raw_metrics["positive_rate"],
        "brier": test_raw_metrics["brier"],
        "roc_auc": test_raw_metrics["roc_auc"],
    }

    # Overfitting check (raw train vs raw test)
    if train_metrics.get("roc_auc") and test_raw_metrics.get("roc_auc"):
        gap = train_metrics["roc_auc"] - test_raw_metrics["roc_auc"]
        if gap > 0.05:
            print(f"  WARNING: followup train/test AUC gap = {gap:.4f} "
                  f"(train={train_metrics['roc_auc']:.4f}, "
                  f"test={test_raw_metrics['roc_auc']:.4f})")

    return artifact, metrics
