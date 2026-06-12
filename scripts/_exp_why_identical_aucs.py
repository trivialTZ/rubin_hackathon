"""Diagnose why all 5 A/B/C variants produced identical AUCs.

Hypothesis (a): my retrain silently failed - q values unchanged
Hypothesis (b): LSST trust/proj features have low importance in baseline
                followup, so changing them doesn't affect predictions
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from debass_meta.projectors import sanitize_expert_key


def main() -> None:
    snap = pd.read_parquet("data/gold/object_epoch_snapshots_trust_safe_v5c.parquet")
    help_df = pd.read_parquet("data/gold/expert_helpfulness_safe_v5c.parquet")
    snap["object_id"] = snap["object_id"].astype(str)
    help_df["object_id"] = help_df["object_id"].astype(str)

    md = json.loads(Path("models/trust_safe_v5c/metadata.json").read_text())
    train_ids = set(str(x) for x in md["train_ids"])
    test_ids = set(str(x) for x in md["test_ids"])
    cal_ids = set(str(x) for x in md.get("cal_ids", []))
    print(f"Split: train={len(train_ids):,}  cal={len(cal_ids):,}  test={len(test_ids):,}")

    for expert in ["fink_lsst/cats", "fink_lsst/snn",
                   "fink_lsst/early_snia", "pittgoogle/supernnova_lsst"]:
        rows = help_df[help_df.expert_key == expert]
        rows = rows[rows.target_class.notna()]
        in_train = rows[rows.object_id.isin(train_ids)]
        in_test = rows[rows.object_id.isin(test_ids)]
        print(f"\n{expert}:")
        print(f"  helpfulness with target: {len(rows):,}")
        print(f"  train: n_rows={len(in_train):,}  n_obj={in_train.object_id.nunique()}")
        print(f"  test : n_rows={len(in_test):,}  n_obj={in_test.object_id.nunique()}")
        if len(in_train) > 0:
            print("  train target_class:", in_train.target_class.value_counts().to_dict())
            print(f"  train is_topclass_correct pos rate:"
                  f" {in_train.is_topclass_correct.mean():.3f}")

    # Feature importance
    with open("models/followup_safe_v5c/model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    md_fu = json.loads(Path("models/followup_safe_v5c/metadata.json").read_text())
    feat_cols = md_fu["feature_cols"]
    imps = model.feature_importances_
    total = imps.sum()
    rank = sorted(zip(imps, feat_cols), reverse=True)
    print(f"\nTotal feature importance: {total}")
    print("\nTop 20 followup features by importance:")
    for imp, c in rank[:20]:
        print(f"  {imp:>8d} ({100*imp/total:5.1f}%)  {c}")

    print("\nLSST-specific feature ranks (q__ and proj__):")
    for target_key in ["fink_lsst__cats", "fink_lsst__snn",
                       "fink_lsst__early_snia", "pittgoogle__supernnova_lsst"]:
        hits = [(i, imp, c) for i, (imp, c) in enumerate(rank, 1)
                if target_key in c and ("q__" in c or "proj__" in c)]
        print(f"\n  {target_key}:")
        for i, imp, c in hits[:8]:
            print(f"    #{i:3d}  imp={imp:>6d} ({100*imp/total:4.1f}%)  {c}")
        if not hits:
            print("    (no features match)")


if __name__ == "__main__":
    main()
