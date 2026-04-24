#!/usr/bin/env python3
"""Score DP1 snapshots with frozen v5d models and emit validation metrics.

Loads:
  - data/gold/dp1_snapshots.parquet  (built by scripts/build_dp1_snapshots.py)
  - models/trust_safe_v5d/<expert>/ (11 trust heads; on SCC)
  - models/followup_safe_v5d/       (followup model)

Emits:
  - reports/v6_dp1/predictions.parquet  per-row (object, n_det, p_follow_proxy,
                                        per-expert trust, ensemble ternary)
  - reports/v6_dp1/metrics.json         rejection-rate-by-class summary
  - stdout report

Per plan §5 Stage 5:
  1. rejection_rate on Gaia-known stars  (expected >95%)
  2. rejection_rate stratified by SIMBAD / Gaia-var class
  3. top-K ranked by p_follow_proxy  (candidates for human inspection)
  4. score histogram by truth class

Usage
-----
    # SCC (default v5d paths):
    python scripts/score_dp1_v5d.py \\
        --trust-dir /project/pi-brout/rubin_hackathon/models/trust_safe_v5d \\
        --followup-dir /project/pi-brout/rubin_hackathon/models/followup_safe_v5d

    # Local (if you've rsynced v5d artifacts):
    python scripts/score_dp1_v5d.py --trust-dir models/trust_safe_v5d \\
        --followup-dir models/followup_safe_v5d
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from debass_meta.models.expert_trust import ExpertTrustArtifact
from debass_meta.models.followup import FollowupArtifact
from debass_meta.projectors import ALL_EXPERT_KEYS, sanitize_expert_key

REPO = Path(__file__).resolve().parents[1]


def _load_trust_models(trust_dir: Path) -> dict[str, ExpertTrustArtifact]:
    models: dict[str, ExpertTrustArtifact] = {}
    if not trust_dir.exists():
        print(f"WARNING: trust dir {trust_dir} does not exist — scoring without trust heads")
        return models
    for child in sorted(trust_dir.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "metadata.json").exists():
            continue
        art = ExpertTrustArtifact.load(child)
        models[art.expert_key] = art
    return models


def _followup(followup_dir: Path) -> FollowupArtifact | None:
    if not (followup_dir / "metadata.json").exists():
        print(f"WARNING: followup model {followup_dir} not found — ensemble score only")
        return None
    return FollowupArtifact.load(followup_dir)


def _bootstrap_ci(values: np.ndarray, n_boot: int = 1000, rng_seed: int = 0) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(rng_seed)
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def score(
    snapshot_path: Path,
    trust_dir: Path,
    followup_dir: Path,
    out_dir: Path,
    *,
    top_percentile: float = 95.0,
) -> None:
    df = pd.read_parquet(snapshot_path)
    print(f"Loaded {len(df):,} snapshot rows ({df['object_id'].nunique():,} objects)")

    trust_models = _load_trust_models(trust_dir)
    print(f"Loaded {len(trust_models)} trust heads: {sorted(trust_models.keys())}")
    follow_model = _followup(followup_dir)

    # Score followup
    if follow_model is not None:
        p_follow = follow_model.predict_followup(df)
        df["p_follow_proxy"] = p_follow
    else:
        df["p_follow_proxy"] = np.nan

    # Per-expert trust scores
    for expert_key, art in trust_models.items():
        san = sanitize_expert_key(expert_key)
        try:
            df[f"trust__{san}"] = art.predict_trust(df)
        except Exception as exc:
            print(f"  [trust:{expert_key}] error {type(exc).__name__}: {exc}")
            df[f"trust__{san}"] = np.nan

    # Trust-weighted ternary ensemble (matches score_nightly.py logic)
    tw_weights = np.zeros(len(df), dtype=float)
    tw_p_snia = np.zeros(len(df), dtype=float)
    tw_p_nonia = np.zeros(len(df), dtype=float)
    tw_p_other = np.zeros(len(df), dtype=float)
    for expert_key in ALL_EXPERT_KEYS:
        san = sanitize_expert_key(expert_key)
        avail_col = f"avail__{san}"
        if avail_col not in df.columns:
            continue
        trust_col = f"trust__{san}"
        if trust_col not in df.columns:
            continue
        snia_col = f"proj__{san}__p_snia"
        nonia_col = f"proj__{san}__p_nonIa_snlike"
        other_col = f"proj__{san}__p_other"
        if snia_col not in df.columns or nonia_col not in df.columns or other_col not in df.columns:
            continue
        p_s_arr = df[snia_col].to_numpy(dtype=float)
        p_n_arr = df[nonia_col].to_numpy(dtype=float)
        p_o_arr = df[other_col].to_numpy(dtype=float)
        avail = df[avail_col].fillna(0).astype(float).to_numpy()
        trust = df[trust_col].fillna(0).astype(float).to_numpy()
        mask = np.isfinite(p_s_arr) & np.isfinite(p_n_arr) & np.isfinite(p_o_arr)
        w = avail * trust * mask.astype(float)
        tw_weights += w
        tw_p_snia += w * np.where(mask, p_s_arr, 0.0)
        tw_p_nonia += w * np.where(mask, p_n_arr, 0.0)
        tw_p_other += w * np.where(mask, p_o_arr, 0.0)
    safe_w = np.where(tw_weights > 0, tw_weights, np.nan)
    df["ensemble_p_snia"] = tw_p_snia / safe_w
    df["ensemble_p_nonIa_snlike"] = tw_p_nonia / safe_w
    df["ensemble_p_other"] = tw_p_other / safe_w
    df["ensemble_n_trusted_experts"] = (tw_weights > 0).astype(int)

    # --- Metrics ---
    # Work on last-epoch snapshot per object (max n_det)
    df_last = df.sort_values(["object_id", "n_det"]).drop_duplicates("object_id", keep="last").copy()
    print(f"\nComputing metrics on last-epoch per object: N={len(df_last):,}")

    p_fu = df_last["p_follow_proxy"].to_numpy()
    finite = np.isfinite(p_fu)
    if finite.sum() == 0:
        print("WARNING: all p_follow_proxy are NaN — skipping rejection metrics")
        tau = np.nan
    else:
        # τ = top-5% threshold over *this DP1 pool* (conservative: uses in-pool
        # quantile; an alternative is the threshold from the v5d training
        # distribution, noted in the metadata for reproducibility).
        tau = float(np.nanpercentile(p_fu, top_percentile))

    metrics: dict[str, object] = {
        "n_objects": int(len(df_last)),
        "tau_topN_percentile": top_percentile,
        "tau_threshold": float(tau) if np.isfinite(tau) else None,
        "median_p_follow_proxy": float(np.nanmedian(p_fu)) if finite.any() else None,
    }

    def _group_metric(name: str, mask: pd.Series) -> None:
        sel = mask.fillna(False).to_numpy() & finite
        if sel.sum() == 0:
            return
        subset = p_fu[sel]
        rejected = (subset < tau) if np.isfinite(tau) else np.zeros_like(subset, dtype=bool)
        lo, hi = _bootstrap_ci(rejected.astype(float), n_boot=500)
        metrics[name] = {
            "n": int(sel.sum()),
            "mean_p_follow_proxy": float(np.nanmean(subset)),
            "median_p_follow_proxy": float(np.nanmedian(subset)),
            "n_rejected": int(rejected.sum()),
            "rejection_rate": float(rejected.mean()),
            "rejection_ci95": [lo, hi],
        }

    # Negative class — Gaia stars
    _group_metric("gaia_known_star", df_last["is_gaia_known_star"] == True)
    # Negative class — known variables
    _group_metric("known_variable", df_last["is_known_variable"] == True)
    # Per-SIMBAD-class stratification
    per_simbad: dict[str, object] = {}
    for simbad_type, grp in df_last.groupby("simbad_main_type", dropna=True):
        n = len(grp)
        if n < 3:
            continue
        sub_p = grp["p_follow_proxy"].to_numpy()
        sub_finite = np.isfinite(sub_p)
        if sub_finite.sum() == 0:
            continue
        rej = (sub_p[sub_finite] < tau) if np.isfinite(tau) else np.array([False])
        per_simbad[str(simbad_type)] = {
            "n": int(n),
            "rejection_rate": float(rej.mean()),
            "mean_p_follow_proxy": float(np.nanmean(sub_p)),
        }
    metrics["per_simbad_class"] = per_simbad

    # Positive class — published SNe
    pub_mask = df_last["is_published_sn"] == True
    if pub_mask.any():
        sub = df_last.loc[pub_mask, ["object_id", "published_sn_name", "published_sn_type",
                                     "published_sn_conf", "p_follow_proxy",
                                     "ensemble_p_snia", "n_det"]].copy()
        sub["above_tau"] = sub["p_follow_proxy"] >= tau if np.isfinite(tau) else False
        hi_conf = sub["published_sn_conf"] >= 0.95
        metrics["published_sn"] = {
            "n_total": int(len(sub)),
            "n_above_tau": int(sub["above_tau"].sum()),
            "recall_above_tau": float(sub["above_tau"].mean()) if len(sub) else None,
            "n_high_conf (>=0.95)": int(hi_conf.sum()),
            "recall_above_tau_high_conf": (
                float(sub.loc[hi_conf, "above_tau"].mean()) if hi_conf.any() else None
            ),
            "rows": sub.sort_values("p_follow_proxy", ascending=False).to_dict(orient="records"),
        }

    # Top-K by followup score (candidates for inspection)
    top_k = 20
    top = df_last.sort_values("p_follow_proxy", ascending=False).head(top_k)
    metrics["top_k"] = top[[
        "object_id", "n_det", "p_follow_proxy",
        "ensemble_p_snia", "ensemble_p_nonIa_snlike", "ensemble_p_other",
        "is_gaia_known_star", "simbad_main_type", "gaia_var_class",
        "is_published_sn", "published_sn_name", "published_sn_type",
    ]].to_dict(orient="records")

    # --- Write ---
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.parquet"
    df.to_parquet(pred_path, index=False)
    print(f"Wrote {len(df):,} scored rows → {pred_path}")

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    print(f"Wrote metrics → {metrics_path}")

    # Pretty-print headline numbers
    print("\n=== DP1 v5d validation headlines ===")
    print(f"  N objects (last epoch) : {metrics['n_objects']:,}")
    print(f"  τ (top-{top_percentile}%)       : "
          f"{metrics['tau_threshold']:.4f}" if metrics['tau_threshold'] is not None else "τ: n/a")
    for key in ("gaia_known_star", "known_variable"):
        if key not in metrics:
            continue
        m = metrics[key]
        lo, hi = m["rejection_ci95"]
        print(f"  {key:20s}: n={m['n']:,}, "
              f"rejection={m['rejection_rate']*100:.1f}% "
              f"[{lo*100:.1f}, {hi*100:.1f}]")
    if "published_sn" in metrics:
        p = metrics["published_sn"]
        n_hi = p.get("n_high_conf (>=0.95)", 0)
        rec_hi = p.get("recall_above_tau_high_conf")
        rec_hi_str = f"{rec_hi*100:.1f}%" if rec_hi is not None else "n/a"
        print(f"  published_sn recall  : {p['n_above_tau']}/{p['n_total']} overall, "
              f"hi-conf {rec_hi_str} of {n_hi}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--snapshots", default=str(REPO / "data/gold/dp1_snapshots.parquet"))
    ap.add_argument("--trust-dir", default=str(REPO / "models/trust_safe_v5d"),
                    help="Dir of per-expert subdirs with metadata.json + model.pkl.")
    ap.add_argument("--followup-dir", default=str(REPO / "models/followup_safe_v5d"))
    ap.add_argument("--out-dir", default=str(REPO / "reports/v6_dp1"))
    ap.add_argument("--top-percentile", type=float, default=95.0,
                    help="Percentile of in-pool p_follow_proxy used as τ (default 95).")
    args = ap.parse_args()

    score(
        Path(args.snapshots),
        Path(args.trust_dir),
        Path(args.followup_dir),
        Path(args.out_dir),
        top_percentile=args.top_percentile,
    )


if __name__ == "__main__":
    main()
