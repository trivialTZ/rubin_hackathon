"""scripts/train_early.py — Train the baseline early-epoch meta-classifier.

This script is kept for benchmark comparisons only.
It is not the primary trust-aware science path.

Trains on (object_id, n_det) epoch feature table.
Outputs: models/early_meta/early_meta.pkl + metrics summary.

Usage:
    python scripts/train_early.py
    python scripts/train_early.py --max-n-det 5   # train only on early epochs
    python scripts/train_early.py --val-frac 0.2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

# Suppress LAPACK/OpenBLAS diagnostic messages from LightGBM internals
os.environ.setdefault("KMP_WARNINGS", "0")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC_ROOT))

import numpy as np

try:
    from debass_meta.models.calibrate import TemperatureScaler
    from debass_meta.models.early_meta import EarlyMetaClassifier, LABEL_MAP, LABEL_NAMES, N_CLASSES
except ModuleNotFoundError as exc:
    print(f"ERROR: failed to import DEBASS package modules from {_SRC_ROOT}: {exc}")
    print("This usually means the SCC checkout is incomplete or stale. Re-sync the repository and retry.")
    sys.exit(1)


def _load_epoch_table(path: Path):
    import pandas as pd
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run build_epoch_table_from_lc.py first")
    return pd.read_parquet(path)


def _metrics(clf, df_val, *, calibrated: bool) -> dict:
    from sklearn.metrics import f1_score, balanced_accuracy_score

    labelled = df_val[df_val["target_label"].notna()]
    if len(labelled) == 0:
        return {}

    y_true = labelled["target_label"].map(LABEL_MAP).values
    probs = clf.predict_proba(labelled, calibrated=calibrated)
    y_pred = probs.argmax(axis=1)

    probs_clip = np.clip(probs, 1e-7, 1)
    nll = float(-np.mean(np.log(probs_clip[np.arange(len(y_true)), y_true])))
    y_oh = np.eye(N_CLASSES)[y_true]
    brier = float(np.mean(np.sum((probs - y_oh) ** 2, axis=1)))

    return {
        "n_val": int(len(labelled)),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "bal_acc": round(balanced_accuracy_score(y_true, y_pred), 4),
        "nll": round(nll, 4),
        "brier": round(brier, 4),
        "probabilities": "calibrated" if calibrated else "raw",
    }


def _early_epoch_metrics(clf, df, max_early: int = 5, *, calibrated: bool) -> dict:
    """Metrics restricted to n_det <= max_early (the critical window)."""
    early = df[df["n_det"] <= max_early]
    m = _metrics(clf, early, calibrated=calibrated)
    m["n_det_max"] = max_early
    return m


def _stratified_object_split(
    df, val_frac: float, seed: int = 42,
) -> tuple[set[str], set[str], set[str]]:
    """Split by object_id into train/cal/test while preserving class balance.

    Returns (train_ids, cal_ids, test_ids).  cal_frac and test_frac each get
    half of val_frac so that total held-out data equals val_frac.
    """
    object_labels = (
        df[df["target_label"].notna()]
        .groupby("object_id")["target_label"]
        .first()
    )
    rng = np.random.default_rng(seed)
    train_ids: set[str] = set()
    cal_ids: set[str] = set()
    test_ids: set[str] = set()

    cal_frac = val_frac / 2.0
    test_frac = val_frac / 2.0

    for label in sorted(object_labels.unique()):
        label_ids = object_labels[object_labels == label].index.to_numpy(copy=True)
        rng.shuffle(label_ids)
        n = len(label_ids)
        if n <= 2:
            train_ids.update(label_ids.tolist())
            continue
        n_cal = max(1, int(round(n * cal_frac)))
        n_test = max(1, int(round(n * test_frac)))
        # Ensure at least 1 training sample
        if n_cal + n_test >= n:
            n_cal = max(1, (n - 1) // 2)
            n_test = max(1, n - 1 - n_cal)
        cal_ids.update(label_ids[:n_cal].tolist())
        test_ids.update(label_ids[n_cal:n_cal + n_test].tolist())
        train_ids.update(label_ids[n_cal + n_test:].tolist())

    if not test_ids:
        object_ids = object_labels.index.to_numpy(copy=True)
        rng.shuffle(object_ids)
        n = len(object_ids)
        n_cal = max(1, int(n * cal_frac))
        n_test = max(1, int(n * test_frac))
        cal_ids = set(object_ids[:n_cal].tolist())
        test_ids = set(object_ids[n_cal:n_cal + n_test].tolist())
        train_ids = set(object_ids[n_cal + n_test:].tolist())

    return train_ids, cal_ids, test_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline early-epoch meta-classifier")
    parser.add_argument("--epochs-dir", default="data/epochs")
    parser.add_argument("--model-dir",  default="models/early_meta")
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument("--max-n-det",  type=int, default=20,
                        help="Only train on rows with n_det <= this value")
    parser.add_argument("--val-frac",   type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()

    import pandas as pd

    epoch_path = Path(args.epochs_dir) / "epoch_features.parquet"
    print("NOTE: train_early.py is the baseline benchmark path, not the primary trust-aware pipeline.")
    print(f"Loading epoch table from {epoch_path}...")
    df = _load_epoch_table(epoch_path)

    df = df[df["n_det"] <= args.max_n_det]
    labelled = df[df["target_label"].notna()]
    print(f"  Total rows: {len(df)} | Labelled: {len(labelled)} | Objects: {df['object_id'].nunique()}")
    print(f"  Label dist: {labelled['target_label'].value_counts().to_dict()}")
    print(f"  n_det dist (labelled): {labelled['n_det'].value_counts().sort_index().to_dict()}")

    if len(labelled) < 3:
        print("Not enough labelled baseline data.")
        print("Run the seed-data path first: download_alerce_training.py → backfill.py → normalize.py → build_epoch_table_from_lc.py")
        sys.exit(1)

    train_ids, cal_ids, test_ids = _stratified_object_split(labelled, args.val_frac, seed=42)

    df_train = df[df["object_id"].isin(train_ids)]
    df_cal   = df[df["object_id"].isin(cal_ids)]
    df_test  = df[df["object_id"].isin(test_ids)]
    print(f"  Train objects: {len(train_ids)} | Cal objects: {len(cal_ids)} | Test objects: {len(test_ids)}")
    print(
        "  Test label dist (objects): "
        f"{df_test[df_test['target_label'].notna()].groupby('target_label')['object_id'].nunique().to_dict()}"
    )

    clf = EarlyMetaClassifier(n_estimators=args.n_estimators)
    clf.fit(df_train)

    raw_all_metrics = _metrics(clf, df_test, calibrated=False)
    raw_early_metrics = _early_epoch_metrics(clf, df_test, max_early=5, calibrated=False)

    calibration_report = {"enabled": False}
    active_all_metrics = raw_all_metrics
    active_early_metrics = raw_early_metrics

    # Fit calibrator on cal set, evaluate on separate test set (no leakage)
    labelled_cal = df_cal[df_cal["target_label"].notna()]
    if len(labelled_cal) > 0:
        y_cal = labelled_cal["target_label"].map(LABEL_MAP).values
        raw_cal_probs = clf.predict_proba(labelled_cal, calibrated=False)
        calibrator = TemperatureScaler()
        calibrator.fit(raw_cal_probs, y_cal)
        clf.set_calibrator(calibrator)
        calibration_report = {
            "enabled": True,
            "method": calibrator.name,
            "temperature": round(float(calibrator.temperature), 6),
            "fit_split": "calibration",
            "note": "Calibrator fit on held-out cal split; metrics evaluated on separate test split.",
        }
        active_all_metrics = _metrics(clf, df_test, calibrated=True)
        active_early_metrics = _early_epoch_metrics(clf, df_test, max_early=5, calibrated=True)

    print(f"\nAll epochs metrics (raw):         {raw_all_metrics}")
    print(f"Early (n_det≤5) metrics (raw):   {raw_early_metrics}")
    if calibration_report["enabled"]:
        print(f"Calibration: {calibration_report}")
        print(f"All epochs metrics (calibrated):  {active_all_metrics}")
        print(f"Early (n_det≤5) metrics (cal'd):  {active_early_metrics}")

    model_dir = Path(args.model_dir)
    clf.save(model_dir)
    print(f"\nModel saved → {model_dir}/early_meta.pkl")

    report = {
        "all_epochs": active_all_metrics,
        "early_epochs": active_early_metrics,
        "raw_all_epochs": raw_all_metrics,
        "raw_early_epochs": raw_early_metrics,
        "calibration": calibration_report,
        "training_weights": clf.training_weight_summary,
        "train_cal_test_split": {
            "train_objects": len(train_ids),
            "cal_objects": len(cal_ids),
            "test_objects": len(test_ids),
            "val_frac": args.val_frac,
            "max_n_det": args.max_n_det,
        },
    }
    metrics_out = Path(args.metrics_out) if args.metrics_out else model_dir.parent.parent / "reports/metrics/early_meta_metrics.json"
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Metrics saved → {metrics_out}")


if __name__ == "__main__":
    main()
