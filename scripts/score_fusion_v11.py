#!/usr/bin/env python3
"""fusion_v11 scorer — extends the fusion_v8 scorer with the anchored blend.

Identical CLI + output schema to ``scripts/score_fusion_v8.py`` (helpers imported,
never reimplemented) plus:

  * ``--blend-dir`` (default ``models/anchor_blend_v11``) — the fitted
    ``BlendSpec`` (``anchor_blend.py``).
  * predictions carry ``p_{snia,nonia,other}_model`` (the hierarchical-head
    calibrated probs), ``p_{snia,nonia,other}_anchor`` (the trust-weighted
    ternary anchor), ``alpha`` and ``alpha_fallback_level``.  ``p_snia/p_nonia/
    p_other`` are the DEPLOYED, post-blend probabilities.

The followup artifact is loaded via ``load_followup_artifact`` (HierarchicalFollowup
if available, else MulticlassFollowupArtifact — both expose ``predict_proba_raw``
+ ``predict_proba``).  Both heads are scored (``predict_proba_raw`` AND
``predict_proba``), the calibrated marginals become the model input to the blend,
and conformal prediction runs on the POST-BLEND deployed probabilities.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# sys.path: repo-root AND src (export_lsst_candidates.py:14-15 pattern) so both
# `scripts.*` and `debass_meta.*` import cleanly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

# Extend by import — never reimplement the v8 helpers/constants.
from scripts.score_fusion_v8 import (  # noqa: E402
    CLASSES,
    DP1_CATALOG_COLS,
    FDR_LABEL_QUALITIES,
    GOALS,
    GOAL_SCORE_COLS,
    LOCAL_PSNIA_COLS,
    attach_trust_columns,
    compute_fdr_thresholds,
    ndet_bucket,
    _latest_per_object,
    trust_weighted_p_snia,
)

from debass_meta.features.availability import BROKER_EXPERTS, LOCAL_EXPERTS  # noqa: E402
from debass_meta.models import anchor_blend  # noqa: E402
from debass_meta.projectors.base import sanitize_expert_key  # noqa: E402


def input_group_counts(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per row: how many broker and local experts fired (avail__<expert> flags) and the serving regime
    (full / no_broker / no_local / none)."""
    def count(experts) -> np.ndarray:
        cols = [f"avail__{sanitize_expert_key(k)}" for k in sorted(experts)]
        cols = [c for c in cols if c in df.columns]
        if not cols:
            return np.zeros(len(df), dtype=int)
        return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).gt(0).sum(axis=1).to_numpy(int)
    nb, nl = count(BROKER_EXPERTS), count(LOCAL_EXPERTS)
    regime = np.select([(nb > 0) & (nl > 0), nl > 0, nb > 0], ["full", "no_broker", "no_local"], default="none")
    return {"n_broker_inputs": nb, "n_local_inputs": nl, "serving_regime": regime}


def load_followup_artifact(followup_dir: str):
    """Load the v11 hierarchical head, falling back to the v8 multiclass head.

    Both classes expose ``predict_proba_raw`` and ``predict_proba`` with a
    classmethod ``load(dir)`` — the scorer only needs that surface.
    """
    try:
        from debass_meta.models.hierarchical_followup import HierarchicalFollowup

        meta = Path(followup_dir) / "metadata.json"
        if meta.exists():
            try:
                if "hierarchical" in json.loads(meta.read_text()).get("kind", "hierarchical"):
                    return HierarchicalFollowup.load(str(followup_dir))
            except Exception:
                return HierarchicalFollowup.load(str(followup_dir))
        return HierarchicalFollowup.load(str(followup_dir))
    except Exception as exc:
        print(f"  [info] HierarchicalFollowup unavailable ({exc}); "
              "loading MulticlassFollowupArtifact")
        from debass_meta.models.multiclass_followup import MulticlassFollowupArtifact

        return MulticlassFollowupArtifact.load(str(followup_dir))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score snapshots with fusion_v11 artifacts (anchored blend)")
    parser.add_argument("--snapshots", default=None,
                        help="Snapshot parquet (default: gold trust snapshot, or DP1 with --dp1)")
    parser.add_argument("--dp1", action="store_true",
                        help="Score DP1 snapshots; joins truth-catalog columns (spec correction #2)")
    parser.add_argument("--dp1-catalog", default="data/gold/dp1_snapshots_50k.parquet",
                        help="Source of DP1 truth-catalog columns when missing from snapshot")
    parser.add_argument("--followup-dir", default="models/followup_fusion_v11")
    parser.add_argument("--trust-dir", default="models/trust_fusion_v11")
    parser.add_argument("--conformal", default="models/conformal_fusion_v11/mondrian_aps.pkl")
    parser.add_argument("--blend-dir", default="models/anchor_blend_v11",
                        help="Fitted BlendSpec dir (anchor_blend.BlendSpec.load)")
    parser.add_argument("--out", default=None,
                        help="Predictions parquet (default: data/scores/predictions_<tag>[_dp1].parquet)")
    parser.add_argument("--scores-dir", default="data/scores")
    parser.add_argument("--budgets", default="20,50,100")
    parser.add_argument("--fdr-gamma", type=float, default=None,
                        help="Add selected_fdr column per goal using the FDR-controlled "
                             "threshold fit on labeled CAL objects (selection.fdr_controlled_threshold)")
    parser.add_argument("--fdr-n-det-max", type=int, default=None,
                        help="Fit FDR thresholds on each cal object's latest epoch with n_det <= cap")
    parser.add_argument("--split", default=None,
                        help="Split manifest providing cal_ids for --fdr-gamma "
                             "(default: data/gold/split_<tag>.json derived from --tag)")
    parser.add_argument("--no-priority", action="store_true", help="Skip priority lists")
    parser.add_argument("--require-local-experts", action="store_true",
                        help="Fail if any row has no local-expert output (default: warn)")
    parser.add_argument("--smoke", action="store_true", help="Score only ~200 objects")
    parser.add_argument("--tag", default="fusion_v11",
                        help="Artifact tag used in default output filenames")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    suffix = "_dp1" if args.dp1 else ""
    snap_path = Path(args.snapshots) if args.snapshots else Path(
        "data/gold/dp1_snapshots_fusion_v11.parquet" if args.dp1
        else "data/gold/object_epoch_snapshots_fusion_v11_trust.parquet"
    )
    out_path = Path(args.out) if args.out else \
        Path(args.scores_dir) / f"predictions_{args.tag}{suffix}.parquet"
    budgets = [int(b) for b in str(args.budgets).split(",") if b.strip()]

    df = pd.read_parquet(snap_path)
    print(f"Loaded {len(df):,} rows ({df['object_id'].nunique():,} objects) from {snap_path}")
    if args.smoke:
        keep = sorted(df["object_id"].astype(str).unique())[:200]
        df = df[df["object_id"].astype(str).isin(keep)].copy()
        print(f"  --smoke: {len(df):,} rows / {len(keep)} objects")

    # ── serving guard: which input groups fired on each row ──────────────────
    # Training rows always carry the local experts; a row without them is out of
    # distribution (P(SN) collapses — docs/fusion_v13_plan.md), so say so loudly.
    df = df.assign(**input_group_counts(df))
    no_local = df["n_local_inputs"].eq(0)
    if no_local.any():
        msg = (f"{int(no_local.sum()):,} of {len(df):,} rows ({df.loc[no_local, 'object_id'].nunique():,} objects) "
               "have no local-expert outputs — run scripts/local_infer.py on their lightcurves first")
        if args.require_local_experts:
            raise SystemExit(f"SERVING GUARD: {msg}")
        print(f"  [warn] {msg}")
    print(f"  serving regimes: {df['serving_regime'].value_counts().to_dict()}")

    # ── q / q_prior columns ──────────────────────────────────────────────────
    df = attach_trust_columns(df, Path(args.trust_dir))

    # ── Stage-B probabilities: raw + calibrated (the calibrated marginals are
    #    the model input to the blend) ─────────────────────────────────────────
    artifact = load_followup_artifact(args.followup_dir)
    p_raw = np.asarray(artifact.predict_proba_raw(df), dtype=float)
    p_cal = np.asarray(artifact.predict_proba(df), dtype=float)
    for i, name in enumerate(("p_snia", "p_nonia", "p_other")):
        df[f"{name}_raw"] = p_raw[:, i]
        df[name] = p_cal[:, i]  # model prob (pre-blend); overwritten post-blend

    # ── anchored blend (fusion_v11) ──────────────────────────────────────────
    blend_dir = Path(args.blend_dir)
    try:
        spec = anchor_blend.BlendSpec.load(blend_dir)
        df = anchor_blend.compute_anchor(
            df, base_rate_by_survey=spec.base_rates,
            default_base_rate=spec.default_base_rate)
        df = anchor_blend.apply(df, spec)
        print(f"  applied anchored blend from {blend_dir} "
              f"(alpha levels: {pd.Series(df['alpha_fallback_level']).value_counts().to_dict()})")
    except FileNotFoundError:
        print(f"  [warn] no BlendSpec at {blend_dir} — anchor=model, alpha=1 (model-only)")
        df = anchor_blend.compute_anchor(df)
        df = anchor_blend.apply(df, anchor_blend.BlendSpec())

    # deployed (post-blend) probability matrix drives conformal + utilities
    p_deploy = df[["p_snia", "p_nonia", "p_other"]].to_numpy(dtype=float)

    # ── conformal prediction sets on POST-BLEND probs ────────────────────────
    conformal_path = Path(args.conformal)
    strata = pd.DataFrame({
        "survey": df["survey"].astype(str).str.lower().to_numpy(),
        "n_det": df["n_det"].astype(int).to_numpy(),
        "n_det_bucket": ndet_bucket(df["n_det"].to_numpy()),
    })
    if conformal_path.exists():
        try:
            from debass_meta.models.conformal import MondrianAPS

            conformal = MondrianAPS.load(str(conformal_path))
            sets = np.asarray(conformal.predict_sets(p_deploy, strata), dtype=bool)
        except Exception as exc:
            print(f"  [warn] conformal scoring failed ({exc}) — emitting full sets")
            sets = np.ones((len(df), 3), dtype=bool)
    else:
        print(f"  [warn] no conformal artifact at {conformal_path} — emitting full sets")
        sets = np.ones((len(df), 3), dtype=bool)
    for i, name in enumerate(("set_snia", "set_nonia", "set_other")):
        df[name] = sets[:, i]
    df["set_size"] = sets.sum(axis=1).astype(int)

    # ── utility scores + compat columns (post-blend) ─────────────────────────
    try:
        from debass_meta.models.selection import UTILITY_PRESETS
    except Exception:
        UTILITY_PRESETS = {"ia": (1, 0, 0), "nonia": (0, 1, 0), "other": (0, 0, 1)}
    for goal, s_col in (("ia", "s_ia"), ("nonia", "s_nonia"), ("other", "s_other")):
        u = np.asarray(UTILITY_PRESETS[goal], dtype=float)
        df[s_col] = p_deploy @ u
    df["p_follow_proxy"] = df["p_snia"]
    df["ensemble_p_snia"] = trust_weighted_p_snia(df)

    # ── DP1 truth-catalog columns (spec correction #2) ───────────────────────
    if args.dp1:
        missing = [c for c in DP1_CATALOG_COLS if c not in df.columns]
        if missing:
            catalog_path = Path(args.dp1_catalog)
            if not catalog_path.exists():
                raise SystemExit(
                    f"DP1 predictions need {missing} but {catalog_path} is missing "
                    "(build_enrichment_metrics.class_masks contract)")
            cat = pd.read_parquet(catalog_path, columns=["object_id"] + DP1_CATALOG_COLS)
            cat = cat.drop_duplicates(subset="object_id")
            df = df.merge(cat[["object_id"] + missing], on="object_id", how="left")
            print(f"  joined DP1 catalog columns {missing} from {catalog_path}")
        still = [c for c in DP1_CATALOG_COLS if c not in df.columns]
        assert not still, f"DP1 contract violated — missing {still}"

    # ── select output columns (v8 schema + blend columns) ────────────────────
    keep_cols = [c for c in (
        "object_id", "diaObjectId", "n_det", "alert_jd", "survey", "survey_is_lsst",
        "target_class", "label_quality", "label_source",
        "p_snia_raw", "p_nonia_raw", "p_other_raw",
        "p_snia_model", "p_nonia_model", "p_other_model",
        "p_snia_anchor", "p_nonia_anchor", "p_other_anchor",
        "n_experts_fired", "n_broker_inputs", "n_local_inputs", "serving_regime", "alpha", "alpha_fallback_level",
        "p_snia", "p_nonia", "p_other",
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

    # ── per-goal priority lists (latest epoch per object) ────────────────────
    if args.no_priority:
        return
    try:
        from debass_meta.models.selection import rank_and_select
    except Exception as exc:
        print(f"  [warn] selection module not importable ({exc}) — priority lists skipped")
        return

    latest = _latest_per_object(df)
    if "traj_x__mean_slope" not in latest.columns:
        latest["traj_x__mean_slope"] = np.nan
    fdr_thresholds: dict[str, dict] = {}
    if args.fdr_gamma is not None:
        split_path = (
            Path(args.split) if args.split
            else Path(f"data/gold/split_{args.tag}.json")
        )
        print(f"  FDR thresholds: cal ids from {split_path}"
              f"{' (derived from --tag)' if not args.split else ''}")
        fdr_thresholds = compute_fdr_thresholds(
            df, split_path, float(args.fdr_gamma), n_det_max=args.fdr_n_det_max)
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
        ranked = per_budget[max(per_budget)].copy()
        for budget, res in per_budget.items():
            if "selected" in res.columns:
                flags = res.set_index("object_id")["selected"].astype(bool)
                ranked[f"selected_at_{budget}"] = (
                    ranked["object_id"].map(flags).fillna(False).astype(bool))
        if goal in fdr_thresholds:
            fit = fdr_thresholds[goal]
            tau = fit["tau"]
            try:
                res_fdr = rank_and_select(
                    base.copy(), u, budget=len(base), score_threshold=tau)
            except Exception as exc:
                print(f"  [warn] FDR selection failed for goal={goal}: {exc}")
            else:
                flags = res_fdr.set_index("object_id")["selected"].astype(bool)
                ranked["selected_fdr"] = (
                    ranked["object_id"].map(flags).fillna(False).astype(bool))
                ranked["fdr_threshold"] = tau
                ranked["fdr_gamma"] = float(args.fdr_gamma)
                ranked["fdr_fit_n_det_median"] = fit["fit_n_det_median"]
                ranked["fdr_fit_n_det_cap"] = (
                    float(fit["fit_n_det_cap"]) if fit["fit_n_det_cap"] is not None else np.nan)
                summary["fdr"] = {
                    "gamma": float(args.fdr_gamma),
                    "threshold": tau if np.isfinite(tau) else None,
                    "n_selected": int(ranked["selected_fdr"].sum()),
                    "fit_n_objects": fit["n_fit"],
                    "fit_n_targets": fit["n_targets"],
                    "fit_n_det_median": fit["fit_n_det_median"],
                    "fit_n_det_max": fit["fit_n_det_max"],
                    "fit_n_det_cap": fit["fit_n_det_cap"],
                }
        ranked.attrs["budget_summary"] = summary
        goal_path = scores_dir / f"priority_{args.tag}{suffix}_{goal}.parquet"
        ranked.to_parquet(goal_path, index=False)
        print(f"Wrote priority list (goal={goal}) → {goal_path} "
              f"[{json.dumps(summary, default=str)}]")


if __name__ == "__main__":
    main()
