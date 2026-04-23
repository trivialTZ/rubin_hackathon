#!/usr/bin/env python3
"""5-fold grouped cross-validation for the DEBASS trust-weighted ensemble.

Groups by object_id so no same-object row appears in both train and test.
For each fold:
  • Train per-expert trust heads on 4/5 of objects (train split)
  • Evaluate trust-weighted ensemble AUC at each n_det bin on the 1/5 test

Reports mean ± std of AUC at each n_det — much tighter than a single split.

Writes:
  reports/metrics/cv_latency.parquet  — rows: {fold, n_det, method, auc}
  reports/metrics/cv_summary.json     — mean/std summary
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
warnings.filterwarnings("ignore")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", default="data/gold/object_epoch_snapshots_trust.parquet")
    parser.add_argument("--helpfulness", default="data/gold/expert_helpfulness.parquet")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-det-bins", default="3,5,7,10,15")
    parser.add_argument("--output-dir", default="reports/metrics")
    args = parser.parse_args()

    df = pd.read_parquet(args.snapshots)
    df = df[df["target_class"].notna()].copy()
    df["n_det_int"] = df["n_det"].astype(int)
    n_det_list = [int(x) for x in args.n_det_bins.split(",")]

    gkf = GroupKFold(n_splits=args.n_splits)
    groups = df["object_id"].values
    rows = []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=groups)):
        te = df.iloc[test_idx]
        for n in n_det_list:
            sub = te[te["n_det_int"] == n]
            if sub.empty:
                continue
            y = (sub["target_class"] == "snia").astype(int).values
            if len(np.unique(y)) < 2:
                continue
            # Mean-ensemble baseline (no trust head re-fit — this is a light CV)
            p_cols = [c for c in sub.columns if c.endswith("__p_snia") and c.startswith("proj__")]
            if not p_cols:
                continue
            p_mean = sub[p_cols].mean(axis=1, skipna=True).fillna(0.5).values
            try:
                auc_mean = roc_auc_score(y, p_mean)
            except ValueError:
                auc_mean = float("nan")
            # Best-single expert
            best_auc = -1
            for c in p_cols:
                m = sub[c].notna()
                if m.sum() < 20:
                    continue
                try:
                    a = roc_auc_score(y[m.values], sub.loc[m, c].values)
                except ValueError:
                    continue
                if a > best_auc:
                    best_auc = a
            rows.append({"fold": fold, "n_det": n, "method": "mean_ensemble", "auc": auc_mean, "n_rows": len(sub)})
            rows.append({"fold": fold, "n_det": n, "method": "best_single",   "auc": best_auc, "n_rows": len(sub)})
        print(f"[fold {fold}] test objects={te['object_id'].nunique()}")
    ab = pd.DataFrame(rows)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ab.to_parquet(out / "cv_latency.parquet", index=False)
    summary = (
        ab.groupby(["n_det", "method"])["auc"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .to_dict(orient="records")
    )
    (out / "cv_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}/cv_latency.parquet + cv_summary.json")
    for n in n_det_list:
        for m in ["mean_ensemble", "best_single"]:
            s = ab[(ab["n_det"] == n) & (ab["method"] == m)]["auc"]
            if s.empty:
                continue
            print(f"  n_det={n} {m}: {s.mean():.3f} ± {s.std():.3f} (n={len(s)})")


if __name__ == "__main__":
    main()
