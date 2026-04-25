#!/usr/bin/env python3
"""Enrichment-factor vs follow-up threshold — paper figure.

Companion to plot_contamination_vs_threshold.py but with the y-axis
reframed as enrichment factor (EF = leakage / K). EF=1 is the random
baseline; EF<1 means the ranker SUPPRESSES the class; EF>1 means the
ranker UP-RANKS the class.

The figure makes the strong / weak / catastrophic regions of v5d's
behavior visually obvious: galaxies and Gaia stars sit below 1×,
periodic variables and AGN sit far above.

Outputs (reports/v6_dp1_50k/)
-----------------------------
  enrichment_curves.png      — paper figure (300 dpi)
  enrichment_curves.pdf      — paper figure (vector)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PRED = REPO / "reports/v6_dp1_50k/predictions.parquet"
DEFAULT_OUT_DIR = REPO / "reports/v6_dp1_50k"

K_GRID = np.logspace(np.log10(0.001), np.log10(0.50), 200)
HEADLINE_K = 0.01
RANKER_COL = "p_follow_proxy"


def latest_per_object(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["object_id", "n_det"])
        .groupby("object_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def class_masks(latest: pd.DataFrame) -> dict[str, np.ndarray]:
    simbad = latest["simbad_main_type"].fillna("").astype(str)
    gs = latest["is_gaia_known_star"].fillna(False).astype(bool).to_numpy()
    gv = latest["is_known_variable"].fillna(False).astype(bool).to_numpy()
    pub = latest["is_published_sn"].fillna(False).astype(bool).to_numpy()
    simbad_typed = simbad.str.len().gt(0).to_numpy()
    return {
        "Gaia known stars":     gs,
        "Gaia known variables": gv,
        "SIMBAD Galaxy":        (simbad == "Galaxy").to_numpy(),
        "SIMBAD QSO/AGN":       simbad.isin(["QSO", "AGN"]).to_numpy(),
        "EclBin + RRLyrae":     simbad.isin(["EclBin", "RRLyrae"]).to_numpy(),
        "Published SNe":        pub,
        "Unlabeled":            (~(gs | gv | simbad_typed | pub)),
    }


def enrichment_curve(
    scores: np.ndarray, mask: np.ndarray, k_grid: np.ndarray
) -> np.ndarray:
    n = len(scores); n_class = int(mask.sum())
    if n_class == 0:
        return np.full_like(k_grid, np.nan)
    order = np.argsort(-scores, kind="stable")
    cum_hits = np.cumsum(mask[order])
    cutoffs = np.clip(np.round(k_grid * n).astype(int), 1, n)
    leak = cum_hits[cutoffs - 1] / n_class
    return leak / k_grid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", type=Path, default=DEFAULT_PRED)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    pred_path = args.pred
    out_dir = args.out_dir

    if not pred_path.exists():
        raise SystemExit(f"Missing: {pred_path}")
    pred = pd.read_parquet(pred_path)
    latest = latest_per_object(pred)
    scores = np.nan_to_num(latest[RANKER_COL].to_numpy(float), nan=0.0)
    masks = class_masks(latest)
    n_pool = len(latest)
    counts = {n: int(m.sum()) for n, m in masks.items()}

    curves = {n: enrichment_curve(scores, m, K_GRID) for n, m in masks.items()}

    styles = {
        "Gaia known stars":      dict(color="#d62728", ls="-",  lw=2.0),
        "Gaia known variables":  dict(color="#ff7f0e", ls="-",  lw=1.8),
        "SIMBAD Galaxy":         dict(color="#9467bd", ls="-",  lw=1.7),
        "SIMBAD QSO/AGN":        dict(color="#8c564b", ls="-",  lw=1.5),
        "EclBin + RRLyrae":      dict(color="#e377c2", ls="-",  lw=1.7),
        "Published SNe":         dict(color="#2ca02c", ls="--", lw=2.4),
        "Unlabeled":             dict(color="#1f77b4", ls="--", lw=1.8),
    }

    fig, ax = plt.subplots(figsize=(8.0, 5.4))

    # background bands: shade up-rank vs suppress regions
    ax.axhspan(1.0, 100.0, color="#fdecea", alpha=0.55, zorder=0)
    ax.axhspan(0.001, 1.0, color="#eafaf1", alpha=0.55, zorder=0)

    # y=1 random reference
    ax.axhline(1.0, color="grey", lw=1.0, ls=":", zorder=1)

    for name, curve in curves.items():
        label = f"{name} (n={counts[name]:,})"
        ax.plot(100.0 * K_GRID, curve, label=label, zorder=3, **styles[name])

    # operating point
    ax.axvline(HEADLINE_K * 100, color="black", lw=0.9, ls=":", zorder=2)
    ax.text(HEADLINE_K * 100 * 1.08, 0.013, f"τ = top {HEADLINE_K*100:.0f}%",
            rotation=90, va="bottom", ha="left", fontsize=9, color="black")

    # zone labels in the LEFT margin so legend can sit upper-right
    ax.text(0.105, 25, "UP-RANK\n(worse than random)",
            fontsize=8, color="#a93226", alpha=0.85, va="top", ha="left")
    ax.text(0.105, 0.012, "SUPPRESS\n(better than random)",
            fontsize=8, color="#1d7d3f", alpha=0.85, va="bottom", ha="left")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 50.0)
    ax.set_ylim(0.01, 50.0)
    ax.set_xlabel("Follow-up threshold K  (top-K%  of  scored pool)")
    ax.set_ylabel("Enrichment factor  EF = (fraction of class in top-K%) / K")
    ax.set_title(
        f"DP1 enrichment vs random ranking — ranker: {RANKER_COL}  (N = {n_pool:,})"
    )
    ax.grid(True, which="both", alpha=0.25, zorder=1)
    leg = ax.legend(loc="upper right", fontsize=8, frameon=True, ncol=1,
                    framealpha=0.92, edgecolor="#cccccc")
    leg.set_zorder(10)

    fig.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.10)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "enrichment_curves.png"
    pdf = out_dir / "enrichment_curves.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)

    print(f"Wrote: {png}")
    print(f"Wrote: {pdf}")
    print(f"Pool N = {n_pool:,}, ranker = {RANKER_COL}, headline K = {HEADLINE_K*100:.1f}%")


if __name__ == "__main__":
    main()
