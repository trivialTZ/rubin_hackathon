#!/usr/bin/env python3
"""fusion_v8 scorer — predictions parquet + per-goal budget-aware priority lists.

Scores ANY snapshot parquet (gold trust-augmented, or DP1 via --dp1) with the
fusion_v8 artifact stack and emits:

  data/scores/predictions_fusion_v8[ _dp1 ].parquet
    object_id, n_det, alert_jd, survey,
    p_snia/p_nonia/p_other (Dirichlet-calibrated) + *_raw,
    q__<expert> (NaN-absent), q_prior__<expert> (expected helpfulness if fetched,
    spec correction #4c), conformal set cols (set_snia/set_nonia/set_other, set_size),
    s_ia/s_nonia/s_other (= u·p̂ for the three presets),
    compat: p_follow_proxy = calibrated p_snia,
            ensemble_p_snia = NaN-aware trust-weighted linear pool (design §1.5).

  data/scores/priority_fusion_v8[ _dp1 ]_<goal>.parquet  (goal ∈ ia/nonia/other)
    latest-epoch-per-object ranking via selection.rank_and_select, with
    selected_at_{20,50,100} membership flags and abstention demotion.

DP1 contract (spec correction #2): the DP1 predictions parquet MUST carry
simbad_main_type, is_gaia_known_star, is_known_variable, is_published_sn,
gaia_var_class so scripts/build_enrichment_metrics.py:class_masks can run
unchanged.  They are carried from the DP1 snapshot, or joined per object_id
from data/gold/dp1_snapshots_50k.parquet when the fusion DP1 snapshot lacks them.

No-leakage argument: scoring is pure inference — every feature in the input
snapshot was built from detections 1..n_det only (gold builder contract), and q
columns computed here use only the trained Stage-A artifact (fit on train, never
on the rows being scored when those rows are cal/test/DP1).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

try:
    from debass_meta.models.multiclass_followup import CLASSES
except Exception:  # pragma: no cover — G4 module built in parallel
    CLASSES = ("snia", "nonIa_snlike", "other")

DP1_CATALOG_COLS = [
    "simbad_main_type", "is_gaia_known_star", "is_known_variable",
    "is_published_sn", "gaia_var_class",
]
GOALS = ("ia", "nonia", "other")
FALLBACK_UTILITY_PRESETS = {"ia": (1, 0, 0), "nonia": (0, 1, 0), "other": (0, 0, 1)}
# proj__ p_snia columns kept for build_enrichment_metrics.best_local_score
LOCAL_PSNIA_COLS = [
    "proj__supernnova__p_snia", "proj__alerce_lc__p_snia", "proj__lc_features_bv__p_snia",
]


def ndet_bucket(n_det: np.ndarray) -> np.ndarray:
    n = np.asarray(n_det, dtype=float)
    out = np.full(n.shape, "10-20", dtype=object)
    out[(n >= 1) & (n <= 4)] = "1-4"
    out[(n >= 5) & (n <= 9)] = "5-9"
    return out


def _latest_per_object(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["object_id", "n_det"])
        .groupby("object_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


# ── trust (q / q_prior) at score time ────────────────────────────────────────

def _expert_dirs(trust_dir: Path) -> list[Path]:
    if not trust_dir.is_dir():
        return []
    return sorted(
        d for d in trust_dir.iterdir()
        if d.is_dir() and d.name != "pooled" and not d.name.startswith("_")
        and (d / "metadata.json").exists()
    )


def attach_trust_columns(df: pd.DataFrame, trust_dir: Path) -> pd.DataFrame:
    """Ensure q__<san> and q_prior__<san> columns exist for every trained expert.

    q semantics (design §1.6): NaN where the expert is absent — NOT 0.0.
    q_prior (spec correction #4c): pooled model scored with the expert's own
    prediction slots NaN'd out, emitted for ALL rows regardless of availability
    ("expected helpfulness if we fetched this broker").
    """
    dirs = _expert_dirs(trust_dir)
    if not dirs:
        print(f"  [warn] no expert dirs under {trust_dir} — q columns left as-is")
        return df
    try:
        from debass_meta.models.pooled_trust import PooledTrustView
    except Exception as exc:
        print(f"  [warn] pooled_trust not importable ({exc}) — q columns left as-is")
        return df

    df = df.copy()
    for expert_dir in dirs:
        san = expert_dir.name
        q_col, qp_col, src_col = f"q__{san}", f"q_prior__{san}", f"trust_source__{san}"
        avail_col = f"avail__{san}"
        own_proj_cols = [c for c in df.columns if c.startswith(f"proj__{san}__")]
        try:
            view = PooledTrustView.load(str(expert_dir))
        except Exception as exc:
            print(f"  [warn] PooledTrustView.load failed for {san}: {exc}")
            continue

        avail = (
            df[avail_col].fillna(0).astype(bool).to_numpy()
            if avail_col in df.columns
            else np.zeros(len(df), dtype=bool)
        )
        if q_col not in df.columns or df[q_col].isna().all():
            q = np.full(len(df), np.nan)
            if avail.any():
                try:
                    q_pred = np.asarray(view.predict_trust(df), dtype=float)
                    q[avail] = q_pred[avail]
                except Exception as exc:
                    print(f"  [warn] predict_trust failed for {san}: {exc}")
            df[q_col] = q
            df[src_col] = np.where(avail, "score_time", "unavailable")
        if qp_col not in df.columns or df[qp_col].isna().all():
            try:
                masked = df.copy()
                for c in own_proj_cols:
                    masked[c] = np.nan
                df[qp_col] = np.asarray(view.predict_trust(masked), dtype=float)
            except Exception as exc:
                print(f"  [warn] q_prior failed for {san}: {exc}")
                df[qp_col] = np.nan
    return df


# ── NaN-aware trust-weighted linear pool (compat ensemble_p_snia) ────────────

def trust_weighted_p_snia(df: pd.DataFrame) -> np.ndarray:
    """p̃_snia = Σ_e avail_e·q_e·p_snia^e / Σ_e avail_e·q_e with NaN-aware q.

    An absent expert (avail=0 or NaN q/proj) contributes NOTHING — the v6e2
    conflation of "absent" with "untrustworthy q=0" is removed (design §1.5/§1.6).
    """
    sans = sorted({
        m.group(1) for c in df.columns
        for m in [re.match(r"proj__(.+)__p_snia$", c)] if m
    })
    num = np.zeros(len(df), dtype=float)
    den = np.zeros(len(df), dtype=float)
    for san in sans:
        p = pd.to_numeric(df[f"proj__{san}__p_snia"], errors="coerce").to_numpy(dtype=float)
        avail_col, q_col = f"avail__{san}", f"q__{san}"
        avail = (
            df[avail_col].fillna(0).astype(bool).to_numpy()
            if avail_col in df.columns else np.isfinite(p)
        )
        q = (
            pd.to_numeric(df[q_col], errors="coerce").to_numpy(dtype=float)
            if q_col in df.columns else np.ones(len(df), dtype=float)
        )
        usable = avail & np.isfinite(p) & np.isfinite(q) & (q > 0)
        num[usable] += p[usable] * q[usable]
        den[usable] += q[usable]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Score snapshots with fusion_v8 artifacts")
    parser.add_argument("--snapshots", default=None,
                        help="Snapshot parquet (default: gold trust snapshot, or DP1 with --dp1)")
    parser.add_argument("--dp1", action="store_true",
                        help="Score DP1 snapshots; joins truth-catalog columns (spec correction #2)")
    parser.add_argument("--dp1-catalog", default="data/gold/dp1_snapshots_50k.parquet",
                        help="Source of DP1 truth-catalog columns when missing from snapshot")
    parser.add_argument("--followup-dir", default="models/followup_fusion_v8")
    parser.add_argument("--trust-dir", default="models/trust_fusion_v8")
    parser.add_argument("--conformal", default="models/conformal_fusion_v8/mondrian_aps.pkl")
    parser.add_argument("--out", default=None,
                        help="Predictions parquet (default: data/scores/predictions_fusion_v8[_dp1].parquet)")
    parser.add_argument("--scores-dir", default="data/scores")
    parser.add_argument("--budgets", default="20,50,100")
    parser.add_argument("--no-priority", action="store_true", help="Skip priority lists")
    parser.add_argument("--smoke", action="store_true", help="Score only ~200 objects")
    args = parser.parse_args()

    suffix = "_dp1" if args.dp1 else ""
    snap_path = Path(args.snapshots) if args.snapshots else Path(
        "data/gold/dp1_snapshots_fusion_v8.parquet" if args.dp1
        else "data/gold/object_epoch_snapshots_fusion_v8_trust.parquet"
    )
    out_path = Path(args.out) if args.out else Path(args.scores_dir) / f"predictions_fusion_v8{suffix}.parquet"
    budgets = [int(b) for b in str(args.budgets).split(",") if b.strip()]

    df = pd.read_parquet(snap_path)
    print(f"Loaded {len(df):,} rows ({df['object_id'].nunique():,} objects) from {snap_path}")
    if args.smoke:
        keep = sorted(df["object_id"].astype(str).unique())[:200]
        df = df[df["object_id"].astype(str).isin(keep)].copy()
        print(f"  --smoke: {len(df):,} rows / {len(keep)} objects")

    # ── q / q_prior columns ──────────────────────────────────────────────────
    df = attach_trust_columns(df, Path(args.trust_dir))

    # ── Stage-B probabilities (raw + Dirichlet-calibrated) ───────────────────
    from debass_meta.models.multiclass_followup import MulticlassFollowupArtifact

    artifact = MulticlassFollowupArtifact.load(str(args.followup_dir))
    p_raw = np.asarray(artifact.predict_proba_raw(df), dtype=float)
    p_cal = np.asarray(artifact.predict_proba(df), dtype=float)
    for i, name in enumerate(("p_snia", "p_nonia", "p_other")):
        df[f"{name}_raw"] = p_raw[:, i]
        df[name] = p_cal[:, i]

    # ── conformal prediction sets ────────────────────────────────────────────
    conformal_path = Path(args.conformal)
    conformal_ok = False
    strata = pd.DataFrame({
        "survey": df["survey"].astype(str).str.lower().to_numpy(),
        "n_det": df["n_det"].astype(int).to_numpy(),
        "n_det_bucket": ndet_bucket(df["n_det"].to_numpy()),
    })
    if conformal_path.exists():
        try:
            from debass_meta.models.conformal import MondrianAPS

            conformal = MondrianAPS.load(str(conformal_path))
            sets = np.asarray(conformal.predict_sets(p_cal, strata), dtype=bool)
            conformal_ok = True
        except Exception as exc:
            print(f"  [warn] conformal scoring failed ({exc}) — emitting full sets")
            sets = np.ones((len(df), 3), dtype=bool)
    else:
        print(f"  [warn] no conformal artifact at {conformal_path} — emitting full sets")
        sets = np.ones((len(df), 3), dtype=bool)
    for i, name in enumerate(("set_snia", "set_nonia", "set_other")):
        df[name] = sets[:, i]
    df["set_size"] = sets.sum(axis=1).astype(int)

    # ── utility scores + compat columns ──────────────────────────────────────
    try:
        from debass_meta.models.selection import UTILITY_PRESETS
    except Exception:
        UTILITY_PRESETS = FALLBACK_UTILITY_PRESETS
    for goal, s_col in (("ia", "s_ia"), ("nonia", "s_nonia"), ("other", "s_other")):
        u = np.asarray(UTILITY_PRESETS[goal], dtype=float)
        df[s_col] = p_cal @ u
    df["p_follow_proxy"] = df["p_snia"]           # compat: Ia-centric consumers
    df["ensemble_p_snia"] = trust_weighted_p_snia(df)

    # ── DP1 truth-catalog columns (spec correction #2) ───────────────────────
    if args.dp1:
        missing = [c for c in DP1_CATALOG_COLS if c not in df.columns]
        if missing:
            catalog_path = Path(args.dp1_catalog)
            if not catalog_path.exists():
                raise SystemExit(
                    f"DP1 predictions need {missing} but {catalog_path} is missing "
                    "(build_enrichment_metrics.class_masks contract)"
                )
            cat = pd.read_parquet(catalog_path, columns=["object_id"] + DP1_CATALOG_COLS)
            cat = cat.drop_duplicates(subset="object_id")
            df = df.merge(cat[["object_id"] + missing], on="object_id", how="left")
            print(f"  joined DP1 catalog columns {missing} from {catalog_path}")
        still = [c for c in DP1_CATALOG_COLS if c not in df.columns]
        assert not still, f"DP1 contract violated — missing {still}"

    # ── select output columns ────────────────────────────────────────────────
    keep_cols = [c for c in (
        "object_id", "diaObjectId", "n_det", "alert_jd", "survey", "survey_is_lsst",
        "target_class", "label_quality", "label_source",
        "p_snia_raw", "p_nonia_raw", "p_other_raw", "p_snia", "p_nonia", "p_other",
        "set_snia", "set_nonia", "set_other", "set_size",
        "s_ia", "s_nonia", "s_other", "p_follow_proxy", "ensemble_p_snia",
        "traj_x__mean_slope",
    ) if c in df.columns]
    keep_cols += [c for c in df.columns if c.startswith(("q__", "q_prior__", "trust_source__"))]
    keep_cols += [c for c in LOCAL_PSNIA_COLS if c in df.columns]
    if args.dp1:
        keep_cols += [c for c in DP1_CATALOG_COLS if c in df.columns]
    pred = df[list(dict.fromkeys(keep_cols))].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(out_path, index=False)
    print(f"Wrote predictions → {out_path} ({len(pred):,} rows, {len(pred.columns)} cols)")

    # ── per-goal priority lists (latest epoch per object; design §3) ─────────
    if args.no_priority:
        return
    try:
        from debass_meta.models.selection import rank_and_select
    except Exception as exc:
        print(f"  [warn] selection module not importable ({exc}) — priority lists skipped")
        return

    latest = _latest_per_object(df)
    if "traj_x__mean_slope" not in latest.columns:
        latest["traj_x__mean_slope"] = np.nan  # tiebreak column contract
    pri_cols = [c for c in (
        "object_id", "diaObjectId", "n_det", "alert_jd", "survey", "target_class",
        "label_quality", "p_snia", "p_nonia", "p_other",
        "set_snia", "set_nonia", "set_other", "set_size", "traj_x__mean_slope",
    ) if c in latest.columns]
    scores_dir = Path(args.scores_dir)
    scores_dir.mkdir(parents=True, exist_ok=True)
    for goal in GOALS:
        u = UTILITY_PRESETS[goal]
        base = latest[pri_cols].copy()
        per_budget: dict[int, pd.DataFrame] = {}
        summary: dict[str, dict] = {}
        failed = False
        for budget in sorted(budgets):
            try:
                res = rank_and_select(base.copy(), u, budget)
            except Exception as exc:
                print(f"  [warn] rank_and_select failed for goal={goal} B={budget}: {exc}")
                failed = True
                break
            per_budget[budget] = res
            attrs = dict(getattr(res, "attrs", {}) or {})
            summary[str(budget)] = {
                "expected_yield": attrs.get("expected_yield"),
                "expected_purity": attrs.get("expected_purity"),
            }
        if failed or not per_budget:
            continue
        # rank/score/flags come from the largest-budget call; smaller budgets
        # contribute their selected membership as selected_at_<B> columns.
        ranked = per_budget[max(per_budget)].copy()
        for budget, res in per_budget.items():
            if "selected" in res.columns:
                flags = res.set_index("object_id")["selected"].astype(bool)
                ranked[f"selected_at_{budget}"] = (
                    ranked["object_id"].map(flags).fillna(False).astype(bool)
                )
        ranked.attrs["budget_summary"] = summary
        goal_path = scores_dir / f"priority_fusion_v8{suffix}_{goal}.parquet"
        ranked.to_parquet(goal_path, index=False)
        print(f"Wrote priority list (goal={goal}) → {goal_path} "
              f"[{json.dumps(summary, default=str)}]")


if __name__ == "__main__":
    main()
