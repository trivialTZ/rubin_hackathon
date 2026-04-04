"""Per-expert trust model training and inference."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from debass_meta.projectors import sanitize_expert_key


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import brier_score_loss, roc_auc_score

    metrics: dict[str, Any] = {"n_rows": int(len(y_true))}
    if len(y_true) == 0:
        metrics["roc_auc"] = None
        metrics["brier"] = None
        return metrics
    metrics["positive_rate"] = float(np.mean(y_true))
    metrics["brier"] = float(brier_score_loss(y_true, y_prob))
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def _prepare_numeric_frame(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    fill_values: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    frame = df.copy()
    for column in feature_cols:
        if column not in frame.columns:
            frame[column] = np.nan
    numeric = frame[feature_cols].apply(pd.to_numeric, errors="coerce")
    # LightGBM handles NaN natively — no imputation needed.
    if fill_values is None:
        fill_values = {}
    return numeric, fill_values


def _fit_binary_classifier(X: pd.DataFrame, y: np.ndarray, *, n_estimators: int = 500, n_jobs: int = 1):
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


def _predict_binary_classifier(model_bundle, X: pd.DataFrame) -> np.ndarray:
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
class ExpertTrustArtifact:
    expert_key: str
    feature_cols: list[str]
    fill_values: dict[str, float]
    model_bundle: Any

    def predict_trust(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = _prepare_numeric_frame(df, self.feature_cols, fill_values=self.fill_values)
        return _predict_binary_classifier(self.model_bundle, X)

    def save(self, expert_dir: Path) -> None:
        expert_dir.mkdir(parents=True, exist_ok=True)
        with open(expert_dir / "model.pkl", "wb") as fh:
            pickle.dump(self.model_bundle, fh)
        with open(expert_dir / "metadata.json", "w") as fh:
            json.dump(
                {
                    "expert_key": self.expert_key,
                    "feature_cols": self.feature_cols,
                    "fill_values": self.fill_values,
                },
                fh,
                indent=2,
            )

    @classmethod
    def load(cls, expert_dir: Path) -> "ExpertTrustArtifact":
        with open(expert_dir / "metadata.json") as fh:
            metadata = json.load(fh)
        with open(expert_dir / "model.pkl", "rb") as fh:
            model_bundle = pickle.load(fh)
        return cls(
            expert_key=str(metadata["expert_key"]),
            feature_cols=[str(column) for column in metadata["feature_cols"]],
            fill_values={str(key): float(value) for key, value in metadata["fill_values"].items()},
            model_bundle=model_bundle,
        )


def _expert_feature_cols(df: pd.DataFrame, expert_key: str) -> list[str]:
    san = sanitize_expert_key(expert_key)
    blocked = {
        "object_id",
        "expert_key",
        "target_class",
        "target_follow_proxy",
        "mapped_pred_class",
        "prediction_type",
        "reason",
        "available",
        "label_source",
        "label_quality",
        "is_topclass_correct",
        "is_helpful_for_follow_proxy",
        "mapped_p_true_class",
        "temporal_exactness",
    }
    feature_cols: list[str] = []
    for column in df.columns:
        if column in blocked:
            continue
        # Exclude absolute timestamps — won't generalize to future observations
        if column == "alert_jd" or column.startswith("source_event_time_jd__"):
            continue
        if column.startswith("proj__") and not column.startswith(f"proj__{san}__"):
            continue
        if column.startswith("q__") or column.startswith("trust_source__"):
            continue
        if df[column].dtype == object and column not in {f"mapped_pred_class__{san}", f"prediction_type__{san}"}:
            continue
        if is_numeric_dtype(df[column]):
            feature_cols.append(column)
    return sorted(set(feature_cols))


def _merge_predictions_into_snapshots(
    snapshot_df: pd.DataFrame,
    helpfulness_df: pd.DataFrame,
    *,
    expert_key: str,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    san = sanitize_expert_key(expert_key)
    key_cols = ["object_id", "n_det", "alert_jd"]
    merged = snapshot_df.merge(
        predictions[key_cols + [f"q__{san}", f"trust_source__{san}"]],
        on=key_cols,
        how="left",
    )
    available_col = f"avail__{san}"
    if available_col in merged.columns:
        merged[f"q__{san}"] = merged[f"q__{san}"].where(merged[available_col].fillna(0).astype(bool), 0.0)
        merged[f"trust_source__{san}"] = merged[f"trust_source__{san}"].where(
            merged[available_col].fillna(0).astype(bool),
            "unavailable",
        )
    else:
        merged[f"q__{san}"] = merged[f"q__{san}"].fillna(0.0)
        merged[f"trust_source__{san}"] = merged[f"trust_source__{san}"].fillna("not_joined")
    return merged


def train_expert_trust_suite(
    *,
    helpfulness_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    split,
    models_dir: Path,
    output_snapshot_path: Path,
    n_splits: int = 5,
    allow_unsafe_latest_snapshot: bool = False,
    n_jobs: int = 1,
):
    from sklearn.model_selection import GroupKFold

    output_df = snapshot_df.copy()
    metrics_report: dict[str, Any] = {}
    trained_experts: list[str] = []
    models_dir.mkdir(parents=True, exist_ok=True)

    for expert_key in sorted(str(key) for key in helpfulness_df["expert_key"].dropna().unique()):
        san = sanitize_expert_key(expert_key)
        expert_rows = helpfulness_df[helpfulness_df["expert_key"] == expert_key].copy()
        if not allow_unsafe_latest_snapshot:
            expert_rows = expert_rows[expert_rows["temporal_exactness"] != "latest_object_unsafe"]
        if len(expert_rows) == 0:
            continue
        if "is_topclass_correct" not in expert_rows.columns or expert_rows["is_topclass_correct"].dropna().empty:
            continue

        expert_rows = expert_rows[expert_rows["is_topclass_correct"].notna()].copy()
        if len(expert_rows) == 0:
            continue

        feature_cols = _expert_feature_cols(expert_rows, expert_key)
        X_all, fill_values = _prepare_numeric_frame(expert_rows, feature_cols)
        y_all = expert_rows["is_topclass_correct"].astype(int).to_numpy()

        train_mask = expert_rows["object_id"].isin(split.train_ids)
        cal_mask = expert_rows["object_id"].isin(split.cal_ids)
        test_mask = expert_rows["object_id"].isin(split.test_ids)

        train_rows = expert_rows[train_mask].copy()
        cal_rows = expert_rows[cal_mask].copy()
        test_rows = expert_rows[test_mask].copy()

        if len(train_rows) == 0:
            continue

        train_X, fill_values = _prepare_numeric_frame(train_rows, feature_cols, fill_values=fill_values)
        train_y = train_rows["is_topclass_correct"].astype(int).to_numpy()

        oof_pred = np.zeros(len(train_rows), dtype=float)
        groups = train_rows["object_id"].astype(str).to_numpy()
        unique_groups = np.unique(groups)
        splits = min(int(n_splits), len(unique_groups))
        if splits >= 2:
            group_kfold = GroupKFold(n_splits=splits)
            for fit_idx, pred_idx in group_kfold.split(train_X, train_y, groups=groups):
                fold_model = _fit_binary_classifier(train_X.iloc[fit_idx], train_y[fit_idx], n_jobs=n_jobs)
                oof_pred[pred_idx] = _predict_binary_classifier(fold_model, train_X.iloc[pred_idx])
        else:
            base_model = _fit_binary_classifier(train_X, train_y, n_jobs=n_jobs)
            oof_pred[:] = _predict_binary_classifier(base_model, train_X)

        train_rows[f"q__{san}"] = oof_pred
        train_rows[f"trust_source__{san}"] = "oof"

        train_fit_model = _fit_binary_classifier(train_X, train_y, n_jobs=n_jobs)

        def _predict_rows(rows: pd.DataFrame, source: str) -> pd.DataFrame:
            rows = rows.copy()
            if len(rows) == 0:
                rows[f"q__{san}"] = []
                rows[f"trust_source__{san}"] = []
                return rows
            X_rows, _ = _prepare_numeric_frame(rows, feature_cols, fill_values=fill_values)
            rows[f"q__{san}"] = _predict_binary_classifier(train_fit_model, X_rows)
            rows[f"trust_source__{san}"] = source
            return rows

        cal_rows = _predict_rows(cal_rows, "train_model")
        test_rows = _predict_rows(test_rows, "train_model")

        merged_predictions = pd.concat([train_rows, cal_rows, test_rows], ignore_index=True, sort=False)
        output_df = _merge_predictions_into_snapshots(
            output_df,
            helpfulness_df,
            expert_key=expert_key,
            predictions=merged_predictions,
        )

        deploy_rows = pd.concat([
            train_rows.drop(columns=[f"q__{san}", f"trust_source__{san}"], errors="ignore"),
            cal_rows.drop(columns=[f"q__{san}", f"trust_source__{san}"], errors="ignore"),
        ], ignore_index=True, sort=False)
        deploy_X, _ = _prepare_numeric_frame(
            deploy_rows if len(deploy_rows) > 0 else train_rows,
            feature_cols,
            fill_values=fill_values,  # reuse training fill_values for consistency
        )
        deploy_y = deploy_rows["is_topclass_correct"].astype(int).to_numpy() if len(deploy_rows) > 0 else train_y
        deploy_model = _fit_binary_classifier(deploy_X, deploy_y, n_jobs=n_jobs)
        artifact = ExpertTrustArtifact(
            expert_key=expert_key,
            feature_cols=feature_cols,
            fill_values=fill_values,
            model_bundle=deploy_model,
        )
        artifact.save(models_dir / san)

        test_metrics = _binary_metrics(
            test_rows["is_topclass_correct"].astype(int).to_numpy(),
            test_rows[f"q__{san}"].astype(float).to_numpy(),
        )
        test_metrics["n_train_rows"] = int(len(train_rows))
        test_metrics["n_cal_rows"] = int(len(cal_rows))
        test_metrics["n_test_rows"] = int(len(test_rows))
        metrics_report[expert_key] = test_metrics
        trained_experts.append(expert_key)

    output_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(output_snapshot_path, index=False)
    metadata = {
        "experts": trained_experts,
        "train_ids": sorted(str(object_id) for object_id in split.train_ids),
        "cal_ids": sorted(str(object_id) for object_id in split.cal_ids),
        "test_ids": sorted(str(object_id) for object_id in split.test_ids),
    }
    return output_df, metrics_report, metadata
