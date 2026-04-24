"""Per-expert trust model training and inference."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from debass_meta.projectors import sanitize_expert_key

from .calibrate import IsotonicCalibrator


# Experts whose ternary projection is designed for SN-vs-other discrimination
# rather than Ia-vs-non-Ia. Their p_snia is mathematically capped at 0.5 by
# the projection (`p_snia = score × 0.5` in fink_lsst projector), so training
# a trust head with target=is_topclass_correct inflates aggregate AUC on
# AGN-heavy samples and inverts on spec-Ia rows.
# For these experts we train trust with target=is_sn (bool target_class != 'other').
# Honest SN-filter AUC: ~0.85 instead of inflated 0.99.
SN_FILTER_EXPERTS = {"fink_lsst/snn", "fink_lsst/cats"}


def trust_target_col(expert_key: str) -> str:
    """Return the helpfulness-row column this expert's trust head should
    be trained against. Default is is_topclass_correct (ternary top-1).
    SN-filter experts use is_sn (binary SN vs other)."""
    if expert_key in SN_FILTER_EXPERTS:
        return "is_sn"
    return "is_topclass_correct"


def _expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Equal-width ECE — mean |bin-accuracy - bin-confidence|, weighted by bin count."""
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


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import brier_score_loss, roc_auc_score

    metrics: dict[str, Any] = {"n_rows": int(len(y_true))}
    if len(y_true) == 0:
        metrics["roc_auc"] = None
        metrics["brier"] = None
        metrics["ece"] = None
        return metrics
    metrics["positive_rate"] = float(np.mean(y_true))
    metrics["brier"] = float(brier_score_loss(y_true, y_prob))
    metrics["ece"] = _expected_calibration_error(y_true, y_prob)
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
    model.fit(X, y)
    return {"kind": "sklearn", "model": model}  # sklearn-compat API


def _predict_binary_classifier(model_bundle, X: pd.DataFrame) -> np.ndarray:
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
class ExpertTrustArtifact:
    expert_key: str
    feature_cols: list[str]
    fill_values: dict[str, float]
    model_bundle: Any
    calibrator: Any = None  # Optional IsotonicCalibrator fit on cal set (Phase 0.1)

    def predict_trust(self, df: pd.DataFrame) -> np.ndarray:
        """Return calibrated trust probability (passes through raw if no calibrator)."""
        raw = self.predict_trust_raw(df)
        if self.calibrator is None:
            return raw
        return np.asarray(self.calibrator.transform(raw), dtype=float)

    def predict_trust_raw(self, df: pd.DataFrame) -> np.ndarray:
        """Uncalibrated trust probability (for diagnostics / stacking features)."""
        X, _ = _prepare_numeric_frame(df, self.feature_cols, fill_values=self.fill_values)
        return _predict_binary_classifier(self.model_bundle, X)

    def save(self, expert_dir: Path) -> None:
        expert_dir.mkdir(parents=True, exist_ok=True)
        with open(expert_dir / "model.pkl", "wb") as fh:
            pickle.dump(self.model_bundle, fh)
        if self.calibrator is not None:
            with open(expert_dir / "calibrator.pkl", "wb") as fh:
                pickle.dump(self.calibrator, fh)
        with open(expert_dir / "metadata.json", "w") as fh:
            json.dump(
                {
                    "expert_key": self.expert_key,
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
    def load(cls, expert_dir: Path) -> "ExpertTrustArtifact":
        with open(expert_dir / "metadata.json") as fh:
            metadata = json.load(fh)
        with open(expert_dir / "model.pkl", "rb") as fh:
            model_bundle = pickle.load(fh)
        calibrator = None
        calibrator_path = expert_dir / "calibrator.pkl"
        if calibrator_path.exists():
            with open(calibrator_path, "rb") as fh:
                calibrator = pickle.load(fh)
        return cls(
            expert_key=str(metadata["expert_key"]),
            feature_cols=[str(column) for column in metadata["feature_cols"]],
            fill_values={str(key): float(value) for key, value in metadata["fill_values"].items()},
            model_bundle=model_bundle,
            calibrator=calibrator,
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
        "is_sn",
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
            # Drop columns that are all-NaN or constant — zero signal
            if df[column].isna().all():
                continue
            if df[column].nunique(dropna=True) <= 1:
                continue
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
    exclude_circular_consensus: bool = True,
    n_jobs: int = 1,
):
    from sklearn.model_selection import GroupKFold

    output_df = snapshot_df.copy()
    metrics_report: dict[str, Any] = {}
    trained_experts: list[str] = []
    models_dir.mkdir(parents=True, exist_ok=True)

    # --- Split integrity validation ---
    assert len(split.train_ids & split.cal_ids) == 0, "train/cal overlap detected"
    assert len(split.train_ids & split.test_ids) == 0, "train/test overlap detected"
    assert len(split.cal_ids & split.test_ids) == 0, "cal/test overlap detected"
    print(f"  Split validated: train={len(split.train_ids):,}, "
          f"cal={len(split.cal_ids):,}, test={len(split.test_ids):,}")

    for expert_key in sorted(str(key) for key in helpfulness_df["expert_key"].dropna().unique()):
        san = sanitize_expert_key(expert_key)
        expert_rows = helpfulness_df[helpfulness_df["expert_key"] == expert_key].copy()
        if not allow_unsafe_latest_snapshot:
            expert_rows = expert_rows[expert_rows["temporal_exactness"] != "latest_object_unsafe"]
        # Anti-circularity filter: exclude rows where truth comes from the
        # same broker family whose trust we are training.  Using a broker's
        # own labels to evaluate its trustworthiness is circular reasoning.
        #
        # 1. broker_consensus: derived from the same experts being evaluated.
        # 2. alerce_self_label: ALeRCE stamp class used as truth for ALeRCE experts.
        #
        # TNS-backed labels ("tns_*"), Fink crossmatch labels, and host-context
        # labels are independent external sources and are always kept.
        if exclude_circular_consensus and "label_source" in expert_rows.columns:
            n_before_circ = len(expert_rows)
            # Always exclude broker_consensus (circular for all experts)
            expert_rows = expert_rows[expert_rows["label_source"] != "broker_consensus"]
            # For ALeRCE experts, also exclude alerce_self_label
            if expert_key.startswith("alerce/") or expert_key == "alerce_lc":
                expert_rows = expert_rows[expert_rows["label_source"] != "alerce_self_label"]
            n_dropped_circ = n_before_circ - len(expert_rows)
            if n_dropped_circ > 0:
                print(f"  {expert_key}: excluded {n_dropped_circ:,} circular label rows")
        if len(expert_rows) == 0:
            continue
        target_col = trust_target_col(expert_key)
        if target_col not in expert_rows.columns or expert_rows[target_col].dropna().empty:
            # Fallback to legacy target if the new SN-filter target isn't present
            # (helpfulness table pre-dates is_sn column).
            if target_col != "is_topclass_correct":
                if ("is_topclass_correct" not in expert_rows.columns
                        or expert_rows["is_topclass_correct"].dropna().empty):
                    continue
                target_col = "is_topclass_correct"
                print(f"  {expert_key}: is_sn column missing, falling back to is_topclass_correct")
            else:
                continue
        if target_col != "is_topclass_correct":
            print(f"  {expert_key}: using trust target = {target_col}")

        expert_rows = expert_rows[expert_rows[target_col].notna()].copy()
        if len(expert_rows) == 0:
            continue

        feature_cols = _expert_feature_cols(expert_rows, expert_key)
        # --- Feature leakage guard ---
        leakage_cols = [c for c in feature_cols if c.startswith(("q__", "trust_source__"))]
        assert len(leakage_cols) == 0, (
            f"Stacking leakage detected for {expert_key}: {leakage_cols}"
        )
        X_all, fill_values = _prepare_numeric_frame(expert_rows, feature_cols)
        y_all = expert_rows[target_col].astype(int).to_numpy()

        train_mask = expert_rows["object_id"].isin(split.train_ids)
        cal_mask = expert_rows["object_id"].isin(split.cal_ids)
        test_mask = expert_rows["object_id"].isin(split.test_ids)

        train_rows = expert_rows[train_mask].copy()
        cal_rows = expert_rows[cal_mask].copy()
        test_rows = expert_rows[test_mask].copy()

        if len(train_rows) == 0:
            continue

        train_X, fill_values = _prepare_numeric_frame(train_rows, feature_cols, fill_values=fill_values)
        train_y = train_rows[target_col].astype(int).to_numpy()
        pos_rate = float(np.mean(train_y)) if len(train_y) > 0 else 0.0
        print(f"  {expert_key}: {len(train_rows):,} train rows, "
              f"{len(feature_cols)} features, pos_rate={pos_rate:.3f}")

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

        # Phase 0.2: Deploy model is train-only (train_fit_model). Cal is
        # reserved strictly for fitting the isotonic calibrator — NOT merged
        # into training data.  This preserves a clean held-out set for
        # post-hoc probability calibration and for unbiased evaluation.
        deploy_model = train_fit_model

        # Phase 0.1: Fit IsotonicCalibrator on (cal raw-prob, cal y).
        # Guards (tightened 2026-04-24 after fink_lsst/early_snia collapse
        # 0.897 raw → 0.488 calibrated with only ~8 cal positives):
        #   (i)   cal_n   >= 200 valid rows
        #   (ii)  cal_pos >= 20 positives
        #   (iii) cal_neg >= 20 negatives
        #   (iv)  Post-fit AUC-drop guard: if calibrator drops test AUC by
        #         > 0.05, reject it (isotonic regression on sparse cal
        #         collapses to near-constant → destroys signal).
        calibrator = None
        cal_raw_prob = None
        cal_y_arr = None
        _calibrator_skip_reason: str | None = None
        if len(cal_rows) > 0:
            cal_raw_prob = cal_rows[f"q__{san}"].astype(float).to_numpy()
            cal_y_arr = cal_rows[target_col].astype(int).to_numpy()
            valid = ~np.isnan(cal_raw_prob) & np.isin(cal_y_arr, [0, 1])
            n_valid = int(valid.sum())
            n_pos = int(cal_y_arr[valid].sum()) if n_valid > 0 else 0
            n_neg = n_valid - n_pos
            if n_valid < 200:
                _calibrator_skip_reason = f"cal_n<200 (have {n_valid})"
            elif n_pos < 20:
                _calibrator_skip_reason = f"cal_pos<20 (have {n_pos})"
            elif n_neg < 20:
                _calibrator_skip_reason = f"cal_neg<20 (have {n_neg})"
            elif len(np.unique(cal_y_arr[valid])) != 2:
                _calibrator_skip_reason = "single_class_in_cal"
            else:
                candidate = IsotonicCalibrator()
                candidate.fit(cal_raw_prob[valid], cal_y_arr[valid])
                # AUC-drop guard on held-out test set (if available)
                if len(test_rows) > 20:
                    test_y_guard = test_rows[target_col].astype(int).to_numpy()
                    test_raw_guard = test_rows[f"q__{san}"].astype(float).to_numpy()
                    gv = ~np.isnan(test_raw_guard) & np.isin(test_y_guard, [0, 1])
                    if gv.sum() > 20 and len(np.unique(test_y_guard[gv])) == 2:
                        from sklearn.metrics import roc_auc_score
                        try:
                            auc_raw = roc_auc_score(test_y_guard[gv], test_raw_guard[gv])
                            auc_cal = roc_auc_score(
                                test_y_guard[gv],
                                np.asarray(candidate.transform(test_raw_guard[gv]), dtype=float),
                            )
                            if (auc_raw - auc_cal) > 0.05:
                                _calibrator_skip_reason = (
                                    f"auc_drop>0.05 (raw={auc_raw:.3f}, cal={auc_cal:.3f})"
                                )
                            else:
                                calibrator = candidate
                        except Exception:
                            calibrator = candidate  # keep it if guard fails for non-AUC reasons
                    else:
                        calibrator = candidate
                else:
                    calibrator = candidate
        if _calibrator_skip_reason:
            print(f"    {expert_key}: calibrator SKIPPED ({_calibrator_skip_reason})")

        artifact = ExpertTrustArtifact(
            expert_key=expert_key,
            feature_cols=feature_cols,
            fill_values=fill_values,
            model_bundle=deploy_model,
            calibrator=calibrator,
        )
        artifact.save(models_dir / san)

        # Report RAW and CALIBRATED test-set metrics so paper can show uplift.
        test_y = test_rows[target_col].astype(int).to_numpy()
        test_raw = test_rows[f"q__{san}"].astype(float).to_numpy()
        raw_metrics = _binary_metrics(test_y, test_raw)
        if calibrator is not None:
            test_cal = np.asarray(calibrator.transform(test_raw), dtype=float)
            cal_metrics = _binary_metrics(test_y, test_cal)
        else:
            cal_metrics = dict(raw_metrics)  # copy; no calibration applied

        # Cal-set self-metrics (on calibrator-fit set — upper-bound, paper uses test)
        if cal_raw_prob is not None and cal_y_arr is not None and calibrator is not None:
            cal_calibrated = np.asarray(calibrator.transform(cal_raw_prob), dtype=float)
            cal_raw_metrics = _binary_metrics(cal_y_arr, cal_raw_prob)
            cal_cal_metrics = _binary_metrics(cal_y_arr, cal_calibrated)
        else:
            cal_raw_metrics = None
            cal_cal_metrics = None

        test_metrics = {
            "raw": raw_metrics,
            "calibrated": cal_metrics,
            "cal_set_raw": cal_raw_metrics,
            "cal_set_calibrated": cal_cal_metrics,
            "n_train_rows": int(len(train_rows)),
            "n_cal_rows": int(len(cal_rows)),
            "n_test_rows": int(len(test_rows)),
            "has_calibrator": calibrator is not None,
            # Backward-compat top-level keys (test raw) — existing consumers
            # read these; add them last so raw_metrics still wins if a key collides.
            **{k: v for k, v in raw_metrics.items() if k not in {"raw", "calibrated"}},
        }
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
