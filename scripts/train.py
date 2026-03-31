"""scripts/train.py — Train baselines and meta-classifier on gold snapshots.

Usage:
    python scripts/train.py
    python scripts/train.py --method lgbm --val-frac 0.2
    python scripts/train.py --method logistic
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from debass_meta.models.baselines import (
    AverageEnsemble, WeightedEnsemble, StackedEnsemble, LABEL_MAP, LABEL_NAMES
)
from debass_meta.models.meta import MetaClassifier
from debass_meta.models.calibrate import TemperatureScaler

_GOLD_DIR = Path("data/gold")
_MODEL_DIR = Path("models")
_BROKERS = ["alerce", "fink", "lasair", "supernnova", "parsnip", "ampel"]


def _load_gold(gold_dir: Path):
    import pandas as pd
    path = gold_dir / "snapshots.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Gold snapshots not found: {path}")
    return pd.read_parquet(path)


def _build_feature_matrix(df, brokers: list[str]):
    """Extract flat feature matrix X and availability masks from gold snapshots."""
    import pandas as pd
    rows = []
    for _, r in df.iterrows():
        scores = r.get("broker_scores") or {}
        avail = r.get("availability_mask") or {}
        row = {"epoch_index": r["epoch_index"]}
        for b in brokers:
            b_scores = {k: v for k, v in scores.items() if k.startswith(b + "__")}
            prob_vals = [
                v["canonical_projection"]
                for v in b_scores.values()
                if isinstance(v, dict) and v.get("semantic_type") == "probability"
                and v.get("canonical_projection") is not None
            ]
            row[f"{b}_mean_prob"] = float(np.mean(prob_vals)) if prob_vals else np.nan
            row[f"{b}_available"] = float(avail.get(b, False))
        rows.append(row)
    feat_df = pd.DataFrame(rows)
    return feat_df


def _encode_labels(df) -> np.ndarray:
    labels = []
    for lbl in df["target_label"]:
        if lbl in LABEL_MAP:
            labels.append(LABEL_MAP[lbl])
        else:
            labels.append(-1)
    return np.array(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DEBASS meta-classifier")
    parser.add_argument("--method", default="lgbm", choices=["lgbm", "logistic"])
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--gold-dir", default="data/gold")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()

    gold_dir = Path(args.gold_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Loading gold snapshots...")
    df = _load_gold(gold_dir)
    print(f"  {len(df)} snapshot rows")

    labels = _encode_labels(df)
    labelled_mask = labels >= 0
    df_lab = df[labelled_mask].copy()
    y = labels[labelled_mask]

    if len(df_lab) == 0:
        print("No labelled snapshots found. Add labels to data/gold/snapshots.parquet.")
        sys.exit(1)

    feat_df = _build_feature_matrix(df_lab, _BROKERS)
    prob_cols = [c for c in feat_df.columns if c.endswith("_mean_prob")]
    avail_cols = [c for c in feat_df.columns if c.endswith("_available")]
    X = feat_df[prob_cols].values
    epoch_idx = feat_df["epoch_index"].values

    # Train/val split
    n = len(X)
    n_val = max(1, int(n * args.val_frac))
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    ep_tr, ep_val = epoch_idx[train_idx], epoch_idx[val_idx]

    print(f"  Train: {len(X_tr)}  Val: {len(X_val)}")

    # --- Baselines ---
    baselines = [
        AverageEnsemble(),
        WeightedEnsemble(),
        StackedEnsemble("logistic"),
        StackedEnsemble("lgbm"),
    ]
    bl_dir = model_dir / "baselines"
    bl_dir.mkdir(exist_ok=True)
    for bl in baselines:
        bl.fit(X_tr, y_tr, feature_names=prob_cols)
        with open(bl_dir / f"{bl.name}.pkl", "wb") as fh:
            pickle.dump(bl, fh)
        print(f"  [baseline] {bl.name} saved")

    # --- Meta-classifier ---
    X_brokers_tr = {b: X_tr[:, [i]] for i, b in enumerate(prob_cols)}
    avail_tr = {b: (X_tr[:, i] > 0).astype(float) for i, b in enumerate(prob_cols)}
    X_brokers_val = {b: X_val[:, [i]] for i, b in enumerate(prob_cols)}
    avail_val = {b: (X_val[:, i] > 0).astype(float) for i, b in enumerate(prob_cols)}

    meta = MetaClassifier(broker_names=prob_cols, method=args.method)
    meta.fit(X_brokers_tr, avail_tr, ep_tr, y_tr)

    meta_dir = model_dir / "meta"
    meta.save(meta_dir)
    # Also pickle the full MetaClassifier so report.py can reload it
    with open(meta_dir / "meta_full.pkl", "wb") as fh:
        pickle.dump(meta, fh)
    print(f"  [meta] {meta.name} saved → {meta_dir}")

    # --- Calibration on val set ---
    probs_val = meta.predict_proba(X_brokers_val, avail_val, ep_val)
    cal = TemperatureScaler()
    cal.fit(probs_val, y_val)
    print(f"  [calibration] temperature={cal.temperature:.4f}")
    with open(meta_dir / "calibrator.pkl", "wb") as fh:
        pickle.dump(cal, fh)

    print("Done. Run scripts/report.py to evaluate.")


if __name__ == "__main__":
    main()
