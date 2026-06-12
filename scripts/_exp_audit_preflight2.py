#!/usr/bin/env python3
"""Audit preflight v2 — enrichment factors across top-K% and rankers.

The first preflight revealed that "94.8% Gaia rejection at top-5%" is only
~1pp above the 95% random baseline (because Gaia stars are 63% of the pool).
The right metric is ENRICHMENT relative to random, computed across multiple
top-K% operating points and contaminant classes.

A method that's 1.2× better than random at K=5% may be 5× better at K=0.5%.
The paper's headline operating point may be wrong.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PRED = REPO / "reports/v6_dp1_50k/predictions.parquet"


def latest_per_object(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["object_id", "n_det"])
        .groupby("object_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def class_leakage_at_K(scores: np.ndarray, mask: np.ndarray, k_frac: float) -> float:
    """Fraction of class members in top-K%."""
    n = len(scores)
    n_class = int(mask.sum())
    if n_class == 0:
        return np.nan
    cutoff = max(int(round(k_frac * n)), 1)
    order = np.argsort(-scores, kind="stable")
    top_set = np.zeros(n, dtype=bool)
    top_set[order[:cutoff]] = True
    return int((mask & top_set).sum()) / n_class


def best_single_broker_score(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in [
        "proj__supernnova__p_snia",
        "proj__alerce_lc__p_snia",
        "proj__lc_features_bv__p_snia",
    ] if c in df.columns]
    arr = df[cols].to_numpy(dtype=float)
    s = np.nanmax(arr, axis=1)
    return np.where(np.isnan(s), 0.0, s)


def main() -> None:
    df = pd.read_parquet(PRED)
    latest = latest_per_object(df)
    n = len(latest)

    pf = latest["p_follow_proxy"].to_numpy(float)
    ep = latest["ensemble_p_snia"].to_numpy(float)
    bb = best_single_broker_score(latest)
    rng = np.random.default_rng(0)
    rd = rng.random(n)

    simbad = latest["simbad_main_type"].fillna("").astype(str)
    masks = {
        "Gaia stars":      latest["is_gaia_known_star"].fillna(False).astype(bool).to_numpy(),
        "Gaia variables":  latest["is_known_variable"].fillna(False).astype(bool).to_numpy(),
        "SIMBAD Galaxy":   (simbad == "Galaxy").to_numpy(),
        "SIMBAD AGN/QSO":  simbad.isin(["AGN","QSO","AGN_Candidate"]).to_numpy(),
        "EclBin/RRLyrae":  simbad.isin(["EclBin","RRLyrae"]).to_numpy(),
        "Published SNe":   latest["is_published_sn"].fillna(False).astype(bool).to_numpy(),
    }

    rankers = {
        "p_follow":  pf,
        "ens_snia":  ep,
        "best_loc":  bb,
        "random":    rd,
    }

    K_GRID = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]

    print("=" * 90)
    print("ENRICHMENT FACTORS  (leakage / k_frac).  >1 = up-ranked relative to random.")
    print(f"  pool n={n:,}.  ranker columns: {list(rankers)}")
    print("=" * 90)

    for cls, m in masks.items():
        n_cls = int(m.sum())
        if n_cls == 0:
            continue
        print(f"\n--- {cls}  (n={n_cls:,}) ---")
        print(f"{'K%':>5s} | " + "  ".join(f"{r:>9s}" for r in rankers))
        for k in K_GRID:
            row = [f"{100*k:>4.1f} |"]
            for rname, rscore in rankers.items():
                leak = class_leakage_at_K(rscore, m, k)
                ef = leak / k if k > 0 else np.nan
                row.append(f"{ef:>9.2f}×")
            print(" ".join(row))

    print()
    print("=" * 90)
    print("READING")
    print("=" * 90)
    print("""
For SNe / Unlabeled (positives):  enrichment >> 1 means the ranker concentrates them at top.
For Gaia / Galaxy / Var / EclBin (negatives): enrichment < 1 means the ranker SUPPRESSES.
                                              enrichment > 1 means the ranker UPRANKS them.

A method paper wants:
  - SNe enrichment ≥ 3× at the chosen K%
  - Gaia/Galaxy/AGN suppression ≤ 0.5× at the chosen K%
  - Periodic variables (EclBin/RRLyrae) NOT being uplifted

If `best_loc` consistently equals or beats `p_follow`/`ens_snia`, then the followup
head + trust ensemble is NOT defensible — drop it, or re-define the operating point.
""")


if __name__ == "__main__":
    main()
