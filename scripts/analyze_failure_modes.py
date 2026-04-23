#!/usr/bin/env python3
"""Pick apart the objects the ensemble still misclassifies at n_det=10.

Joins gold snapshot with truth table (host context, redshift, discovery broker)
and writes a failure-mode report: which objects are hard, and what features
correlate with the error.

Writes:
  reports/failure_modes.csv           — one row per misclassified object at n_det=10
  reports/failure_mode_summary.json   — aggregate stats
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CLASSES = ["snia", "nonIa_snlike", "other"]


def _ensemble_prob_ia(df: pd.DataFrame) -> pd.Series:
    p_cols = [c for c in df.columns if c.endswith("__p_snia") and c.startswith("proj__")]
    if not p_cols:
        return pd.Series(np.nan, index=df.index)
    return df[p_cols].mean(axis=1, skipna=True)


def _ensemble_pred_class(df: pd.DataFrame) -> pd.Series:
    p_cols = {cls: [c for c in df.columns if c.endswith(f"__p_{cls}") and c.startswith("proj__")] for cls in CLASSES}
    probs = pd.DataFrame({cls: df[p_cols[cls]].mean(axis=1, skipna=True) if p_cols[cls] else np.nan for cls in CLASSES})
    # Rows where all three class probs are NaN → no prediction (abstain)
    all_nan = probs.isna().all(axis=1)
    pred = probs.fillna(-1).idxmax(axis=1)
    pred[all_nan] = "abstain"
    return pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", default="data/gold/object_epoch_snapshots_trust.parquet")
    parser.add_argument("--truth", default="data/truth/object_truth.parquet")
    parser.add_argument("--n-det", type=int, default=10)
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    df = pd.read_parquet(args.snapshots)
    df = df[df["target_class"].isin(CLASSES)].copy()
    df["n_det_int"] = df["n_det"].astype(int)
    df = df[df["n_det_int"] == args.n_det].copy()
    df["p_snia_ens"] = _ensemble_prob_ia(df)
    df["pred_class"] = _ensemble_pred_class(df)
    df["correct"] = df["pred_class"] == df["target_class"]
    # Only count misclassifications when the ensemble made a real prediction
    fails = df[(~df["correct"]) & (df["pred_class"] != "abstain")].copy()

    try:
        truth = pd.read_parquet(args.truth)
        join_cols = [c for c in ["object_id", "ra", "dec", "label_source", "label_quality", "discovery_broker", "host_context", "redshift"] if c in truth.columns]
        fails = fails.merge(truth[join_cols], on="object_id", how="left")
    except Exception as exc:
        print(f"[warn] truth join failed: {exc}")

    cols_keep = [
        "object_id", "target_class", "pred_class", "p_snia_ens", "n_det",
        "label_source", "label_quality", "discovery_broker",
    ]
    cols_keep = [c for c in cols_keep if c in fails.columns]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fails[cols_keep].to_csv(out / "failure_modes.csv", index=False)

    summary = {
        "n_det": args.n_det,
        "total": int(len(df)),
        "misclassified": int(len(fails)),
        "error_rate": float(len(fails) / max(len(df), 1)),
        "error_by_truth_class": fails["target_class"].value_counts().to_dict(),
        "error_by_pred_class": fails["pred_class"].value_counts().to_dict() if "pred_class" in fails.columns else {},
        "error_by_label_source": fails["label_source"].value_counts().to_dict() if "label_source" in fails.columns else {},
    }
    (out / "failure_mode_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"wrote {out}/failure_modes.csv ({len(fails)} rows) and summary.json")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
