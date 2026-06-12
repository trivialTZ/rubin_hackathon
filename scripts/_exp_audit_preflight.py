#!/usr/bin/env python3
"""Audit preflight — verify two paper experiments are feasible from existing artifacts.

EXPERIMENT 1 — "Does the followup head earn its keep?"
   Compare top-5% Gaia rejection under three rankers:
     A) p_follow_proxy            (current operational score)
     B) ensemble_p_snia           (trust-weighted SNIa, no followup head)
     C) max(per-broker p_snia)    (best single available broker per object)
     D) random                    (sanity floor)

EXPERIMENT 2 — "Does it work in the early-follow-up regime?"
   Per-n_det bin, recompute top-5% Gaia rejection under each ranker.
   Bins: [1-2], [3-5], [6-10], [11-20].

If both produce distinguishable, paper-quality numbers from the existing
predictions.parquet, the audit's recommended next-step is feasible without
re-scoring on SCC.
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


def topk_rejection(scores: np.ndarray, mask: np.ndarray, k_frac: float = 0.05) -> dict:
    """At top-K%, what fraction of `mask`-class members are NOT in the cut?"""
    n = len(scores)
    order = np.argsort(-scores, kind="stable")
    cutoff = max(int(round(k_frac * n)), 1)
    top_set = np.zeros(n, dtype=bool)
    top_set[order[:cutoff]] = True
    n_class = int(mask.sum())
    if n_class == 0:
        return {"n": 0, "leaked": 0, "rejection": np.nan}
    leaked = int((mask & top_set).sum())
    return {"n": n_class, "leaked": leaked, "rejection": 1 - leaked / n_class}


def random_rejection(mask: np.ndarray, k_frac: float, n_rounds: int = 200) -> float:
    """Mean rejection if we ranked uniformly at random."""
    n = len(mask); n_class = int(mask.sum())
    if n_class == 0:
        return np.nan
    cutoff = max(int(round(k_frac * n)), 1)
    rng = np.random.default_rng(0)
    leaks = []
    for _ in range(n_rounds):
        order = rng.permutation(n)
        top_set = np.zeros(n, dtype=bool); top_set[order[:cutoff]] = True
        leaks.append(int((mask & top_set).sum()) / n_class)
    return 1 - float(np.mean(leaks))


def best_single_broker_score(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Per row, max of available per-broker p_snia projections.
    Falls back to NaN where none available — those rows tie-rank at the bottom.
    """
    cands = [
        "proj__supernnova__p_snia",
        "proj__alerce_lc__p_snia",
        "proj__lc_features_bv__p_snia",
    ]
    cols = [c for c in cands if c in df.columns]
    arr = df[cols].to_numpy(dtype=float)
    s = np.nanmax(arr, axis=1)
    s = np.where(np.isnan(s), 0.0, s)
    return s, "+".join(cols)


def main() -> None:
    if not PRED.exists():
        raise SystemExit(f"Missing: {PRED}")
    df = pd.read_parquet(PRED)
    latest = latest_per_object(df)

    pf = latest["p_follow_proxy"].to_numpy(float)
    ep = latest["ensemble_p_snia"].to_numpy(float)
    bb, bb_src = best_single_broker_score(latest)

    gaia = latest["is_gaia_known_star"].fillna(False).astype(bool).to_numpy()
    var = latest["is_known_variable"].fillna(False).astype(bool).to_numpy()
    pub = latest["is_published_sn"].fillna(False).astype(bool).to_numpy()

    rankers = {
        "p_follow_proxy   (followup head)":  pf,
        "ensemble_p_snia  (trust SNIa)":     ep,
        "best_local_p_snia (max of 3)":      bb,
    }

    print("=" * 72)
    print("EXPERIMENT 1 — top-5% Gaia rejection by ranker (latest-snapshot pool)")
    print("=" * 72)
    print(f"pool n_objects = {len(latest):,}   single-broker source: {bb_src}")
    print()
    print(f"{'ranker':38s} | {'Gaia(★)':>8s} {'Var':>8s} {'PubSN':>8s}  {'topK%':>6s}")
    print("-" * 72)
    for name, scores in rankers.items():
        rg = topk_rejection(scores, gaia, 0.05)
        rv = topk_rejection(scores, var, 0.05)
        rp = 1 - topk_rejection(scores, pub, 0.05)["rejection"]  # SN: want HIGH retention
        print(f"{name:38s} | {rg['rejection']*100:>7.1f}% {rv['rejection']*100:>7.1f}% {rp*100:>7.1f}%  {'5':>5}%")
    rg_rand = random_rejection(gaia, 0.05)
    rv_rand = random_rejection(var, 0.05)
    print(f"{'random_baseline':38s} | {rg_rand*100:>7.1f}% {rv_rand*100:>7.1f}% {'~5.0':>7s}%  {'5':>5}%")

    print()
    print("=" * 72)
    print("EXPERIMENT 2 — per-n_det Gaia rejection at top-5%")
    print("(uses ALL snapshots, not latest — early-follow-up regime)")
    print("=" * 72)
    bins = [(1, 2), (3, 5), (6, 10), (11, 20)]
    print(f"{'n_det bin':>10s}  {'rows':>6s}  {'Gaia n':>7s}  | {'p_follow':>10s}  {'ens_snia':>10s}  {'best_loc':>10s}  {'random':>8s}")
    print("-" * 80)
    for lo, hi in bins:
        mb = (df["n_det"] >= lo) & (df["n_det"] <= hi)
        sub = df.loc[mb].reset_index(drop=True)
        if len(sub) == 0:
            continue
        pf_b = sub["p_follow_proxy"].to_numpy(float)
        ep_b = sub["ensemble_p_snia"].to_numpy(float)
        bb_b, _ = best_single_broker_score(sub)
        gaia_b = sub["is_gaia_known_star"].fillna(False).astype(bool).to_numpy()
        rg_pf = topk_rejection(pf_b, gaia_b, 0.05)["rejection"]
        rg_ep = topk_rejection(ep_b, gaia_b, 0.05)["rejection"]
        rg_bb = topk_rejection(bb_b, gaia_b, 0.05)["rejection"]
        rg_rd = random_rejection(gaia_b, 0.05, n_rounds=100)
        n_g = int(gaia_b.sum())
        print(f"  [{lo:>2},{hi:>2}]  {len(sub):>6,}  {n_g:>7,}  | "
              f"{rg_pf*100:>9.1f}%  {rg_ep*100:>9.1f}%  {rg_bb*100:>9.1f}%  {rg_rd*100:>7.1f}%")

    print()
    print("=" * 72)
    print("INTERPRETATION KEY")
    print("=" * 72)
    print("EXP1 question: does p_follow_proxy beat a pure SNIa-ensemble or single broker?")
    print("              if YES → the followup head EARNS the model-card line that says it.")
    print("              if NO  → drop it; just rank by ensemble_p_snia. Simpler, defensible.")
    print()
    print("EXP2 question: does the method work at n_det ≤ 5 (early-follow-up regime)?")
    print("              if Gaia rejection holds at [3-5] → thesis is supported.")
    print("              if it collapses at low n_det → paper must scope-down.")


if __name__ == "__main__":
    main()
