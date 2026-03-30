"""scripts/train_early.py — Train early-epoch meta-classifier.

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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from debass.models.early_meta import EarlyMetaClassifier, LABEL_MAP, LABEL_NAMES, N_CLASSES


def _load_epoch_table(path: Path):
    import pandas as pd
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run build_epoch_table.py first")
    return pd.read_parquet(path)


def _metrics(clf, df_val) -> dict:
    from sklearn.metrics import f1_score, balanced_accuracy_score

    labelled = df_val[df_val["target_label"].notna()]
    if len(labelled) == 0:
        return {}

    y_true = labelled["target_label"].map(LABEL_MAP).values
    probs = clf.predict_proba(labelled)
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
    }


def _early_epoch_metrics(clf, df, max_early: int = 5) -> dict:
    """Metrics restricted to n_det <= max_early (the critical window)."""
    early = df[df["n_det"] <= max_early]
    m = _metrics(clf, early)
    m["n_det_max"] = max_early
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="Train early-epoch meta-classifier")
    parser.add_argument("--epochs-dir", default="data/epochs")
    parser.add_argument("--model-dir",  default="models/early_meta")
    parser.add_argument("--max-n-det",  type=int, default=20,
                        help="Only train on rows with n_det <= this value")
    parser.add_argument("--val-frac",   type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()

    import pandas as pd

    epoch_path = Path(args.epochs_dir) / "epoch_features.parquet"
    print(f"Loading epoch table from {epoch_path}...")
    df = _load_epoch_table(epoch_path)

    # Filter to requested max epoch depth
    df = df[df["n_det"] <= args.max_n_det]
    labelled = df[df["target_label"].notna()]
    print(f"  Total rows: {len(df)} | Labelled: {len(labelled)} | Objects: {df['object_id'].nunique()}")
    print(f"  Label dist: {labelled['target_label'].value_counts().to_dict()}")
    print(f"  n_det dist (labelled): {labelled['n_det'].value_counts().sort_index().to_dict()}")

    if len(labelled) < 3:
        print("Not enough labelled data. Run fetch_training_data.py --limit 500 first.")
        sys.exit(1)

    # Train/val split by object (not by row — prevents leakage)
    object_ids = np.array(list(labelled["object_id"].unique()))
    rng = np.random.default_rng(42)
    rng.shuffle(object_ids)
    n_val = max(1, int(len(object_ids) * args.val_frac))
    val_ids = set(object_ids[:n_val])
    train_ids = set(object_ids[n_val:])

    df_train = df[df["object_id"].isin(train_ids)]
    df_val   = df[df["object_id"].isin(val_ids)]
    print(f"  Train objects: {len(train_ids)} | Val objects: {len(val_ids)}")

    # Train
    clf = EarlyMetaClassifier(n_estimators=args.n_estimators)
    clf.fit(df_train)

    # Evaluate
    all_metrics = _metrics(clf, df_val)
    early_metrics = _early_epoch_metrics(clf, df_val, max_early=5)

    print(f"\nAll epochs metrics:   {all_metrics}")
    print(f"Early (n_det≤5) metrics: {early_metrics}")

    # Save
    model_dir = Path(args.model_dir)
    clf.save(model_dir)
    print(f"\nModel saved → {model_dir}/early_meta.pkl")

    # Save metrics
    report = {"all_epochs": all_metrics, "early_epochs": early_metrics}
    reports_dir = Path("reports/metrics")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "early_meta_metrics.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Metrics saved → {reports_dir}/early_meta_metrics.json")


if __name__ == "__main__":
    main()
