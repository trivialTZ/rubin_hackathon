#!/usr/bin/env python3
"""Per-expert ablation on the v6 DP1 production run — operational reality.

Why this script exists
----------------------
The v5d architecture has 11 trust heads. On the DP1 pool, however, eight of
them have `avail__*=0` (the public LSST brokers don't index DP1 today). The
ENSEMBLE that ranks DP1 follow-ups is therefore a 3-expert local-rerunnable
classifier mix:

    alerce_lc         (100 % available)
    supernnova        (100 % available)
    lc_features_bv    (16 % available — Bazin/Villar fits)

A reviewer asking "which expert is doing the work?" deserves an answer.
This script measures top-K% enrichment factor under five rankers, with
bootstrap CI95:

    1.  p_follow_proxy             — current operational followup head
    2.  ensemble_p_snia            — current trust-weighted SNIa probability
    3.  alerce_lc only             — single-expert ablation
    4.  supernnova only            — single-expert ablation
    5.  lc_features_bv only        — single-expert ablation (sparse)
    6.  naive_mean (no trust)      — equal-weight average of available
    7.  random                     — sanity floor

The 8-broker ablation on the v5d *training* set (where brokers are populated)
requires SCC artifacts and is tracked as a separate task.

Outputs (reports/v6_dp1_50k/)
-----------------------------
  ablation_3expert.json
  ablation_3expert_per_class.csv
  ablation_3expert.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PRED = REPO / "reports/v6_dp1_50k/predictions.parquet"
OUT_DIR = REPO / "reports/v6_dp1_50k"

K_GRID = [0.005, 0.01, 0.02, 0.05]
HEADLINE_K = 0.01
N_BOOT = 1000
RNG_SEED = 20260425


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
        "Gaia stars":      gs,
        "Gaia variables":  gv,
        "SIMBAD Galaxy":   (simbad == "Galaxy").to_numpy(),
        "SIMBAD AGN/QSO":  simbad.isin(["AGN", "QSO", "AGN_Candidate"]).to_numpy(),
        "EclBin+RRLyrae":  simbad.isin(["EclBin", "RRLyrae"]).to_numpy(),
        "Published SNe":   pub,
        "Unlabeled":       (~(gs | gv | simbad_typed | pub)),
    }


def naive_mean_score(df: pd.DataFrame) -> np.ndarray:
    """Equal-weight mean of available proj p_snia (no trust weights)."""
    cols = [c for c in [
        "proj__alerce_lc__p_snia",
        "proj__supernnova__p_snia",
        "proj__lc_features_bv__p_snia",
    ] if c in df.columns]
    arr = df[cols].to_numpy(dtype=float)
    s = np.nanmean(arr, axis=1)
    return np.where(np.isnan(s), 0.0, s)


def enrichment_at_K(scores: np.ndarray, mask: np.ndarray, k_frac: float) -> float:
    n = len(scores); n_class = int(mask.sum())
    if n_class == 0:
        return float("nan")
    cutoff = max(int(round(k_frac * n)), 1)
    order = np.argsort(-scores, kind="stable")
    top = np.zeros(n, dtype=bool)
    top[order[:cutoff]] = True
    leak = int((mask & top).sum()) / n_class
    return leak / k_frac


def bootstrap_ef(
    scores: np.ndarray, mask: np.ndarray, k_frac: float, n_boot: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    n = len(scores)
    if int(mask.sum()) == 0:
        return float("nan"), float("nan"), float("nan")
    efs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s_b = scores[idx]; m_b = mask[idx]
        if int(m_b.sum()) == 0:
            efs[i] = np.nan; continue
        efs[i] = enrichment_at_K(s_b, m_b, k_frac)
    return (
        float(np.nanmean(efs)),
        float(np.nanpercentile(efs, 2.5)),
        float(np.nanpercentile(efs, 97.5)),
    )


def fmt(mean: float, lo: float, hi: float) -> str:
    if np.isnan(mean): return "—"
    return f"{mean:.2f} [{lo:.2f}, {hi:.2f}]"


def main() -> None:
    if not PRED.exists():
        raise SystemExit(f"Missing: {PRED}")
    df = pd.read_parquet(PRED)
    latest = latest_per_object(df)
    n_pool = len(latest)
    rng = np.random.default_rng(RNG_SEED)

    # Per-expert single-broker scores
    sc_alerce = np.where(np.isnan(latest["proj__alerce_lc__p_snia"]), 0.0,
                         latest["proj__alerce_lc__p_snia"]).astype(float)
    sc_snn = np.where(np.isnan(latest["proj__supernnova__p_snia"]), 0.0,
                      latest["proj__supernnova__p_snia"]).astype(float)
    if "proj__lc_features_bv__p_snia" in latest.columns:
        sc_bv = np.where(np.isnan(latest["proj__lc_features_bv__p_snia"]), 0.0,
                         latest["proj__lc_features_bv__p_snia"]).astype(float)
    else:
        sc_bv = np.zeros(n_pool, float)

    rankers = {
        "p_follow_proxy":     latest["p_follow_proxy"].to_numpy(float),
        "ensemble_p_snia":    latest["ensemble_p_snia"].to_numpy(float),
        "alerce_lc_only":     sc_alerce,
        "supernnova_only":    sc_snn,
        "lc_features_bv_only": sc_bv,
        "naive_mean_3expert": naive_mean_score(latest),
        "random":             rng.random(n_pool),
    }
    masks = class_masks(latest)

    out = {
        "n_pool": n_pool,
        "operating_points": K_GRID,
        "headline_K": HEADLINE_K,
        "rankers": list(rankers),
        "n_bootstrap": N_BOOT,
        "rng_seed": RNG_SEED,
        "by_class": {},
        "expert_availability_on_dp1": {
            "alerce_lc":     "100% (local rerunnable)",
            "supernnova":    "100% (local rerunnable)",
            "lc_features_bv": "16%  (Bazin/Villar fits — needs n_det≥5 with quality)",
            "8 broker experts": "0% — brokers don't index DP1",
        },
    }
    flat_rows = []
    for cname, mask in masks.items():
        n_cls = int(mask.sum())
        out["by_class"][cname] = {"n": n_cls, "by_K": {}}
        for k in K_GRID:
            row = {"class": cname, "n": n_cls, "K_pct": 100 * k}
            cell = {}
            for rname, scores in rankers.items():
                cell_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
                ef_mean, lo, hi = bootstrap_ef(scores, mask, k, N_BOOT, cell_rng)
                cell[rname] = {
                    "ef_point": enrichment_at_K(scores, mask, k),
                    "ef_mean": ef_mean,
                    "ef_ci95_lo": lo,
                    "ef_ci95_hi": hi,
                }
                row[f"{rname}_ef"] = round(ef_mean, 4)
                row[f"{rname}_ef_lo"] = round(lo, 4)
                row[f"{rname}_ef_hi"] = round(hi, 4)
            out["by_class"][cname]["by_K"][f"{k:.4f}"] = cell
            flat_rows.append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "ablation_3expert.json"
    csv_path = OUT_DIR / "ablation_3expert_per_class.csv"
    md_path = OUT_DIR / "ablation_3expert.md"

    json_path.write_text(json.dumps(out, indent=2))
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)

    # Markdown
    headline_classes = ["Gaia stars", "SIMBAD Galaxy", "EclBin+RRLyrae",
                        "Gaia variables", "Published SNe"]
    lines = [
        f"## Per-expert ablation on DP1 — top-{HEADLINE_K*100:.0f}% (N={n_pool:,})",
        "",
        "### Operational availability on DP1",
        "",
        "| Expert | Availability | Source |",
        "|---|---|---|",
        "| alerce_lc | 100% | local rerunnable LightGBM (15,428-label train) |",
        "| supernnova | 100% | local rerunnable RNN |",
        "| lc_features_bv | 16% | local Bazin/Villar fits (sparse: needs n_det≥5 quality) |",
        "| 8 broker experts (fink, fink_lsst, alerce broker, ampel, antares, pittgoogle, lasair, salt3) | 0% | LSST brokers don't index DP1 |",
        "",
        "The trust-weighted ensemble on DP1 reduces operationally to a 3-expert",
        "local-classifier ensemble. The ablation below isolates which expert(s)",
        "drive the headline galaxy-suppression result.",
        "",
        "### Enrichment factor at top-1 % (lower for negatives = better)",
        "",
        "| Class | n |" + "|".join(f" {r} " for r in rankers) + "|",
        "|---|---:|" + "|".join("---" for _ in rankers) + "|",
    ]
    for cname in headline_classes:
        cell = out["by_class"][cname]["by_K"][f"{HEADLINE_K:.4f}"]
        n = out["by_class"][cname]["n"]
        row = [f"| {cname} | {n:,}"]
        for rname in rankers:
            c = cell[rname]
            row.append(f" {fmt(c['ef_mean'], c['ef_ci95_lo'], c['ef_ci95_hi'])}")
        lines.append("|".join(row) + " |")

    # Reading
    def _ef(cls, rk):
        c = out["by_class"][cls]["by_K"][f"{HEADLINE_K:.4f}"][rk]
        return fmt(c["ef_mean"], c["ef_ci95_lo"], c["ef_ci95_hi"])

    lines += [
        "",
        "### Reading",
        "",
        f"**SIMBAD Galaxy** (n=402): trust-weighted ensemble EF = {_ef('SIMBAD Galaxy', 'ensemble_p_snia')}; "
        f"naive 3-expert mean EF = {_ef('SIMBAD Galaxy', 'naive_mean_3expert')}; "
        f"alerce_lc alone EF = {_ef('SIMBAD Galaxy', 'alerce_lc_only')}; "
        f"supernnova alone EF = {_ef('SIMBAD Galaxy', 'supernnova_only')}.",
        "",
        f"**Gaia stars** (n=9,937): trust-weighted EF = {_ef('Gaia stars', 'ensemble_p_snia')}; "
        f"alerce_lc alone EF = {_ef('Gaia stars', 'alerce_lc_only')}; "
        f"supernnova alone EF = {_ef('Gaia stars', 'supernnova_only')}.",
        "",
        f"**EclBin+RRLyrae** (n=52, the up-rank limitation): "
        f"p_follow EF = {_ef('EclBin+RRLyrae', 'p_follow_proxy')}; "
        f"trust-weighted EF = {_ef('EclBin+RRLyrae', 'ensemble_p_snia')}; "
        f"alerce_lc alone EF = {_ef('EclBin+RRLyrae', 'alerce_lc_only')}; "
        f"supernnova alone EF = {_ef('EclBin+RRLyrae', 'supernnova_only')}.",
        "",
        "### Caveat — full 11-expert ablation requires the v5d training set",
        "",
        "The 8 broker experts have `avail=0` on DP1, so this ablation cannot test",
        "their contribution. The full 11-expert ablation against the v5d training",
        "set (where brokers are populated) is deferred and tracked separately.",
    ]
    md_path.write_text("\n".join(lines) + "\n")

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print()
    print(f"=== HEADLINE @ top-{HEADLINE_K*100:.0f}% ===")
    print(f"  pool n_objects = {n_pool:,}")
    print()
    for cname in headline_classes:
        cell = out["by_class"][cname]["by_K"][f"{HEADLINE_K:.4f}"]
        n = out["by_class"][cname]["n"]
        print(f"--- {cname}  (n={n:,}) ---")
        for rname in rankers:
            c = cell[rname]
            print(f"  {rname:25s}  EF = {fmt(c['ef_mean'], c['ef_ci95_lo'], c['ef_ci95_hi'])}")
        print()


if __name__ == "__main__":
    main()
