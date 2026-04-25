#!/usr/bin/env python3
"""Contamination-vs-threshold curve — the operational paper figure.

For each class in the scored DP1 pool, plot what fraction of its members
lands in the top-K% of p_follow_proxy as K varies. Contaminants we want
LOW (Gaia stars, variables, galaxies, periodic variables). Positives we
want HIGH (published SNe, the unlabeled discovery pool).

The headline point is the vertical line at K=5%: it shows the leakage /
retention rates at the follow-up threshold chosen in the paper.

Outputs (reports/v6_dp1_50k/):
  contamination_vs_threshold.png      — paper figure (300 dpi)
  contamination_vs_threshold.pdf      — paper figure (vector)
  contamination_vs_threshold.csv      — numeric table at K ∈ {0.5,1,2,5,10,20,50}%
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PRED = REPO / "reports/v6_dp1_50k/predictions.parquet"
OUT_DIR = REPO / "reports/v6_dp1_50k"

K_GRID = np.logspace(np.log10(0.001), np.log10(0.50), 200)  # 0.1% → 50%
K_TABLE = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]        # table rows


def latest_per_object(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["object_id", "n_det"])
    return df.groupby("object_id", as_index=False).tail(1).reset_index(drop=True)


def class_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    simbad = df["simbad_main_type"].fillna("").astype(str)
    gs = df["is_gaia_known_star"].fillna(False).astype(bool).to_numpy()
    gv = df["is_known_variable"].fillna(False).astype(bool).to_numpy()
    pub = df["is_published_sn"].fillna(False).astype(bool).to_numpy()
    simbad_typed = simbad.str.len().gt(0).to_numpy()
    return {
        "Gaia known stars":        gs,
        "Gaia known variables":    gv,
        "SIMBAD Galaxy":           (simbad == "Galaxy").to_numpy(),
        "SIMBAD QSO/AGN":          simbad.isin(["QSO", "AGN"]).to_numpy(),
        "EclBin + RRLyrae":        simbad.isin(["EclBin", "RRLyrae"]).to_numpy(),
        "Published SNe":           pub,
        "Unlabeled":               (~(gs | gv | simbad_typed | pub)),
    }


def fraction_in_topk(
    scores: np.ndarray, class_mask: np.ndarray, k_grid: np.ndarray
) -> np.ndarray:
    """At each K, return fraction of class members above the top-K% threshold."""
    n = len(scores)
    order = np.argsort(-scores, kind="stable")  # highest first
    ranked_mask = class_mask[order]
    cum_hits = np.cumsum(ranked_mask)
    total = ranked_mask.sum()
    if total == 0:
        return np.full_like(k_grid, np.nan)
    cutoffs = np.clip(np.round(k_grid * n).astype(int), 1, n)
    return cum_hits[cutoffs - 1] / total


def main() -> None:
    if not PRED.exists():
        raise SystemExit(f"Missing: {PRED}")
    pred = pd.read_parquet(PRED)
    latest = latest_per_object(pred)
    scores = latest["p_follow_proxy"].to_numpy(dtype=float)
    scores = np.nan_to_num(scores, nan=0.0)
    masks = class_masks(latest)
    n_objects = len(latest)

    curves: dict[str, np.ndarray] = {
        name: fraction_in_topk(scores, m, K_GRID) for name, m in masks.items()
    }
    counts: dict[str, int] = {name: int(m.sum()) for name, m in masks.items()}

    styles = {
        "Gaia known stars":       dict(color="#d62728", ls="-",  lw=2.0),   # red
        "Gaia known variables":   dict(color="#ff7f0e", ls="-",  lw=1.8),   # orange
        "SIMBAD Galaxy":          dict(color="#9467bd", ls="-",  lw=1.5),   # purple
        "SIMBAD QSO/AGN":         dict(color="#8c564b", ls="-",  lw=1.5),   # brown
        "EclBin + RRLyrae":       dict(color="#e377c2", ls="-",  lw=1.5),   # pink
        "Published SNe":          dict(color="#2ca02c", ls="--", lw=2.2),   # green dashed
        "Unlabeled":              dict(color="#1f77b4", ls="--", lw=1.8),   # blue dashed
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for name, curve in curves.items():
        label = f"{name} (n={counts[name]:,})"
        ax.plot(100.0 * K_GRID, 100.0 * curve, label=label, **styles[name])

    # τ = 5% vertical callout
    ax.axvline(5.0, color="k", lw=0.8, ls=":")
    ax.text(
        5.0, 2.5, "  τ = top 5%", rotation=0, va="bottom", ha="left",
        fontsize=9, color="k",
    )

    # Diagonal reference: random ranking → class fraction = K
    k_pct = 100.0 * K_GRID
    ax.plot(k_pct, k_pct, color="grey", lw=0.6, alpha=0.5, label="random ranking")

    ax.set_xscale("log")
    ax.set_xlim(0.1, 50.0)
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel("Follow-up threshold K  (top-K%  of  scored pool)")
    ax.set_ylabel("Fraction of class in top-K%  [%]")
    ax.set_title(
        f"DP1 prioritization leakage / retention  (N = {n_objects:,} objects)"
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=1)

    fig.tight_layout()
    png = OUT_DIR / "contamination_vs_threshold.png"
    pdf = OUT_DIR / "contamination_vs_threshold.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)

    # Numeric table at canonical K
    rows = []
    for k in K_TABLE:
        row = {"K_percent": 100.0 * k}
        for name in masks:
            frac = fraction_in_topk(scores, masks[name], np.array([k]))[0]
            row[name] = float(frac) if not np.isnan(frac) else None
        rows.append(row)
    tbl = pd.DataFrame(rows).round(4)
    tbl.to_csv(OUT_DIR / "contamination_vs_threshold.csv", index=False)

    print(f"Wrote: {png}")
    print(f"Wrote: {pdf}")
    print(f"Wrote: {OUT_DIR/'contamination_vs_threshold.csv'}")
    print(f"Scored pool: N = {n_objects:,}")
    for name, n in counts.items():
        print(f"  {name:24s} n = {n:>6,}")
    print()
    print(f"Leakage / retention at canonical K  (% of class in top-K%):")
    print(tbl.to_string(index=False))


if __name__ == "__main__":
    main()
