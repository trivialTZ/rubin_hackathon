#!/usr/bin/env python3
"""Top-K unlabeled follow-up candidates from the v6 DP1 production run.

This is the method's actual output: a ranked list of DP1 objects an observer
would take to a spectrograph tonight. It drops known contaminants (Gaia
stars, Gaia variables, SIMBAD-typed objects, already-published transients)
and reports what's left at the top of the v5d follow-up score.

Per object we keep the highest-n_det snapshot (the latest/most-informed
score v5d produced before the reliability-filtered light curve ran out).

Outputs
-------
- reports/v6_dp1_50k/topk_unlabeled.csv         Full ranked table (top-200)
- reports/v6_dp1_50k/topk_unlabeled_top20.md    Paper-ready Markdown table
- reports/v6_dp1_50k/topk_filter_breakdown.txt  How many rows each filter removes
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PRED = REPO / "reports/v6_dp1_50k/predictions.parquet"
DEFAULT_OUT_DIR = REPO / "reports/v6_dp1_50k"


def load_latest_snapshot_per_object(df: pd.DataFrame) -> pd.DataFrame:
    """For each object_id take the row with highest n_det (most-informed score)."""
    df = df.sort_values(["object_id", "n_det"])
    latest = df.groupby("object_id", as_index=False).tail(1)
    return latest.reset_index(drop=True)


def filter_unlabeled(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop anything with a known label from Gaia / SIMBAD / published-SN lists.

    Returns (filtered_df, per_filter_counts).
    """
    n0 = len(df)
    mask_gaia_star = df["is_gaia_known_star"].fillna(False).astype(bool)
    mask_gaia_var = df["is_known_variable"].fillna(False).astype(bool)
    mask_simbad = df["simbad_main_type"].notna()
    mask_pub = df["is_published_sn"].fillna(False).astype(bool)
    mask_any = mask_gaia_star | mask_gaia_var | mask_simbad | mask_pub

    counts = {
        "total_objects": n0,
        "gaia_known_star": int(mask_gaia_star.sum()),
        "gaia_known_variable": int(mask_gaia_var.sum()),
        "simbad_typed": int(mask_simbad.sum()),
        "published_sn": int(mask_pub.sum()),
        "any_label": int(mask_any.sum()),
        "unlabeled_remaining": int((~mask_any).sum()),
    }
    return df.loc[~mask_any].reset_index(drop=True), counts


def _fmt_mag(x: float | None) -> str:
    try:
        return f"{float(x):.2f}"
    except (TypeError, ValueError):
        return "–"


def build_markdown_top20(df: pd.DataFrame) -> str:
    """Paper-ready table: rank, object, n_det, p_follow, ensemble, LC summary."""
    lines = [
        "| Rank | diaObjectId | Field | n_det | p_follow | p(Ia) | p(non-Ia) | p(other) | bands | Δt (d) | mag_min |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for i, r in enumerate(df.itertuples(index=False), start=1):
        bands = ""
        for b in "ugrizy":
            n = getattr(r, f"n_det_{b}", np.nan)
            if pd.notna(n) and n > 0:
                bands += b
        lines.append(
            f"| {i} | {r.diaObjectId} | {r.dp1_field} | {int(r.n_det)} "
            f"| {r.p_follow_proxy:.3f} | {r.ensemble_p_snia:.2f} "
            f"| {r.ensemble_p_nonIa_snlike:.2f} | {r.ensemble_p_other:.2f} "
            f"| {bands or '–'} | {_fmt_mag(r.t_baseline)} | {_fmt_mag(r.mag_min)} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", type=Path, default=DEFAULT_PRED)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    pred_path = args.pred
    out_dir = args.out_dir

    if not pred_path.exists():
        raise SystemExit(f"Missing predictions parquet: {pred_path}")

    pred = pd.read_parquet(pred_path)
    n_snap = len(pred)
    latest = load_latest_snapshot_per_object(pred)
    n_obj = len(latest)
    unl, counts = filter_unlabeled(latest)

    unl_sorted = unl.sort_values(
        ["p_follow_proxy", "n_det"], ascending=[False, False]
    ).reset_index(drop=True)

    # Keep a compact column set for the CSV / paper table
    keep_cols = [
        "object_id", "diaObjectId", "dp1_field", "survey", "n_det",
        "n_det_u", "n_det_g", "n_det_r", "n_det_i", "n_det_z", "n_det_y",
        "t_baseline", "mag_first", "mag_last", "mag_min", "mag_range",
        "color_gr", "color_ri", "dmag_dt", "mean_quality",
        "p_follow_proxy",
        "ensemble_p_snia", "ensemble_p_nonIa_snlike", "ensemble_p_other",
        "ensemble_n_trusted_experts",
    ]
    keep = [c for c in keep_cols if c in unl_sorted.columns]
    top200 = unl_sorted[keep].head(200).copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    top200.to_csv(out_dir / "topk_unlabeled.csv", index=False)

    md_table = build_markdown_top20(top200.head(20))
    (out_dir / "topk_unlabeled_top20.md").write_text(md_table + "\n")

    break_path = out_dir / "topk_filter_breakdown.txt"
    tau_topk_pct = 0.05
    tau_row_idx = max(int(len(unl_sorted) * tau_topk_pct) - 1, 0)
    tau_threshold_unlabeled = float(unl_sorted.iloc[tau_row_idx]["p_follow_proxy"])
    with break_path.open("w") as fh:
        fh.write(f"Snapshots in predictions.parquet : {n_snap:,}\n")
        fh.write(f"Unique objects (latest snapshot)  : {n_obj:,}\n")
        fh.write("\n")
        fh.write("Filter breakdown on latest-snapshot population:\n")
        for k, v in counts.items():
            fh.write(f"  {k:24s}: {v:>6,}\n")
        fh.write("\n")
        fh.write(
            f"Top-5% threshold within UNLABELED pool "
            f"(n={len(unl_sorted):,}): p_follow_proxy = {tau_threshold_unlabeled:.4f}\n"
        )
        fh.write(
            f"  → objects above τ : "
            f"{int((unl_sorted['p_follow_proxy'] >= tau_threshold_unlabeled).sum()):,}\n"
        )

    print(f"Wrote: {out_dir/'topk_unlabeled.csv'} ({len(top200)} rows)")
    print(f"Wrote: {out_dir/'topk_unlabeled_top20.md'}")
    print(f"Wrote: {break_path}")
    print()
    print("─── TOP 20 UNLABELED FOLLOW-UP CANDIDATES ───")
    print(md_table)
    print()
    print("─── FILTER BREAKDOWN ───")
    print(break_path.read_text())


if __name__ == "__main__":
    main()
