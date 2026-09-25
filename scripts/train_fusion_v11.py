#!/usr/bin/env python3
"""fusion_v11 training orchestrator (work package P6).

Integrates the v11 arms behind the pinned interfaces — nothing here reimplements
a shared module; every stage is a call into P1-P5 code or a v8 helper imported
by name:

  Stage A  pooled trust      -> ``pooled_trust.train_pooled_trust`` (v8 pattern;
                                reused verbatim, q__/q_prior__/trust_source__ into
                                the snapshot).
  P3 heads + calibration      -> ``HierarchicalFollowup.fit`` (guard G7 asserts
                                inside ``fit``; per-head/per-survey calibrators fit
                                on CAL; the fit report carries the §5 gate verdicts,
                                the availability audit and the provenance-mask ledger).
  P4 anchor + alpha           -> ``anchor_blend.fit_alpha`` on CAL (grid / 1-SE /
                                honesty filter / fallback ladder; guard G3 recorded
                                per cell + post-fit per-survey pooled verification).
  conformal (POST-BLEND)      -> ``MondrianAPS`` refit on the DEPLOYED (blended)
                                CAL probabilities, strata (survey, n_det, n_det_bucket).
  FDR                         -> ``score_fusion_v8.compute_fdr_thresholds`` on the
                                deployed CAL utilities (recorded; the scorer refits
                                at score time — this is the train-stage ledger).
  guards G2 / G3 / G6 / G7    -> asserted here (G6 after ``load_split_manifest``;
                                G2 on the OOF-train/spec-cal/assoc-spec LSST spec-Ia
                                union; G3 from the BlendSpec; G7 delegated to fit).
  gates (spec §5)             -> surfaced from the head fit report + the seq_v11
                                in/out verdict.

G1 (ZTF locked-test macro AUC) and G4 (LSST-live benchmark) are TEST-TOUCHING and
run in the EVAL stage ONLY (eval_fusion_v8 / the benchmark re-score in the job
scripts) — never consulted during training, so the pre-registered headline stays
honest (spec deviation #16).

All v11 builds/train take ``--truth data/truth/object_truth_v11.parquet`` (B0);
training refuses to fit Head-2 when that file is absent (B0 ordering: P1's
rederive must land first).  v11 writes ONLY ``*_v11``-suffixed artifacts.

fusion v13 flags (docs/fusion_v13_plan.md; all default to the v11/v12 path):
  --head1-exclude-quality lsst:weak   B1  drop a label tier (optionally survey-
                                          scoped) from head 1, its calibrators,
                                          the per-survey gate and the α fit
  --no-lsst-equalization              B2  keep the class-pure ALeRCE-family
                                          equalization off
  --head1-context-mask survey         B2  mask the context family on every row
                                          of a survey with context labels
  --head1-survey-mask lsst:parsnip    B2  mask a named expert on a whole survey
  --g8-max-corr 0.2 / --acknowledge-g8    guard G8 on availability–label corr
  --head1-dropout (+ --dropout-*)     B3  availability-dropout augmentation
  --cross-fit-folds 5                 B4  OOF heads; calibrators on OOF-train ∪
                                          cal; α and G2 on OOF-train ∪ cal
  --head1-cal-weights object          v13b calibrators without label-quality factors
  --stage-a-weak-policy / --stage-a-q-prior-experts   Stage-A passthroughs

Outputs:
  models/trust_fusion_v11/                       (Stage A)
  models/followup_fusion_v11/                    (HierarchicalFollowup)
  models/anchor_blend_v11/blend.json             (BlendSpec)
  models/conformal_fusion_v11/mondrian_aps.pkl   (post-blend Mondrian APS)
  data/gold/object_epoch_snapshots_fusion_v11_trust.parquet
  reports/metrics/fusion_v11_train.json          (guards + gates + ledgers)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# sys.path: repo-root AND src (export_lsst_candidates.py:14-15 pattern) so both
# ``scripts.*`` and ``debass_meta.*`` import cleanly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Extend by import — never reimplement the v8 orchestration helpers/constants.
from scripts.train_fusion_v8 import (  # noqa: E402
    CAL_QUALITIES,
    CLASSES,
    SPEC_QUALITIES,
    _jsonify,
    _macro_ovr_auc,  # noqa: F401 — re-exported for eval-stage reuse / tests
    _paired_delta_ci,  # noqa: F401
    _smoke_subset,
    class_index,
    load_split_manifest,
    ndet_bucket,
    run_component_gates,  # noqa: F401 — surfaced for the seq_v11 component gate
)

from debass_meta.models import anchor_blend  # noqa: E402

_MODEL_COLS = ("p_snia", "p_nonia", "p_other")


# ── locked-test manifest (G6-at-train) ───────────────────────────────────────

def load_locked_test_ids(path: Path) -> set[str]:
    """Read ``lsst_live_locked_test.json`` test_ids (P1-pinned schema).

    Missing file → empty set (mirrors the ``--association-csv`` / ``--lsst-live-locked``
    no-op convention); the caller records that G6 was skipped for want of a manifest.
    """
    if not path.exists():
        return set()
    with open(path) as fh:
        raw = json.load(fh)
    node = raw.get("split", raw) if isinstance(raw, dict) else raw
    ids = node.get("test_ids", node.get("test")) if isinstance(node, dict) else node
    return {str(x) for x in (ids or [])}


# ── guard G2 (spec §4 / B6) ──────────────────────────────────────────────────

def _lsst_spec_ia_frame(
    snap: pd.DataFrame, train_ids: set[str], cal_ids: set[str],
    test_ids: set[str] | None = None,
) -> pd.DataFrame:
    """G2 evaluation union (B6): OOF-train LSST spec-Ia ∪ LSST spec-tier cal ∪
    assoc-spec rows.  One row per object at its latest epoch.

    G2 is an OOF/cal-only guard (spec §4) — it must NEVER touch a test row.
    The unconditional ``is_lsst & assoc_spec`` term could otherwise pull in a
    locked-benchmark/test id that happens to be assoc-spec, so any id in
    ``test_ids`` is excluded from the union."""
    sv = snap["survey"].astype(str).str.lower()
    oid = snap["object_id"].astype(str)
    lq = (
        snap["label_quality"].astype(str).str.lower()
        if "label_quality" in snap.columns
        else pd.Series([""] * len(snap), index=snap.index)
    )
    ls = (
        snap["label_source"].astype(str)
        if "label_source" in snap.columns
        else pd.Series([""] * len(snap), index=snap.index)
    )
    is_lsst = sv == "lsst"
    is_ia = snap["target_class"].astype(str) == "snia"
    spec = lq.isin([q.lower() for q in SPEC_QUALITIES])
    assoc_spec = ls == "ztf_assoc_spec"

    test_ids = test_ids or set()
    not_test = ~oid.isin(test_ids)
    train_ia = is_lsst & is_ia & spec & oid.isin(train_ids)
    cal_spec = is_lsst & spec & (oid.isin(cal_ids) | assoc_spec)
    keep = ((train_ia | cal_spec | (is_lsst & assoc_spec)) & not_test).to_numpy()
    frame = snap[keep].copy()
    if "n_det" in frame.columns and len(frame):
        frame = (
            frame.sort_values(["object_id", "n_det"])
            .groupby("object_id", as_index=False)
            .tail(1)
        )
    return frame.reset_index(drop=True)


def evaluate_g2(
    snap_scored: pd.DataFrame, train_ids: set[str], cal_ids: set[str],
    *, acknowledge_unevaluable: bool, test_ids: set[str] | None = None,
) -> dict[str, Any]:
    """G2: on the LSST spec-Ia union, median p_snia >= 0.15 AND max > 0.2, n >= 10.

    ``snap_scored`` must already carry the DEPLOYED (post-blend) ``p_snia``.
    Returns a status dict; ``status`` ∈ {PASS, FAIL, G2_UNEVALUABLE}.
    """
    frame = _lsst_spec_ia_frame(snap_scored, train_ids, cal_ids, test_ids)
    n = int(len(frame))
    n_all_nan = 0
    if "lc_all_nan" in frame.columns:
        n_all_nan = int(pd.to_numeric(frame["lc_all_nan"], errors="coerce").fillna(0).sum())
    if n < 10:
        return {
            "guard": "G2", "status": "G2_UNEVALUABLE",
            "n_rows": n, "n_lc_all_nan": n_all_nan,
            "reason": f"only {n} eligible LSST spec-Ia union rows (require n>=10)",
            "acknowledged": bool(acknowledge_unevaluable),
        }
    p = pd.to_numeric(frame["p_snia"], errors="coerce").to_numpy(float)
    p = p[np.isfinite(p)]
    med = float(np.median(p)) if len(p) else float("nan")
    mx = float(np.max(p)) if len(p) else float("nan")
    ok = (med >= 0.15) and (mx > 0.2)
    return {
        "guard": "G2", "status": "PASS" if ok else "FAIL",
        "n_rows": n, "n_lc_all_nan": n_all_nan,
        "median_p_snia": med, "max_p_snia": mx,
        "threshold": {"median_min": 0.15, "max_min": 0.2},
    }


# ── guard G3 (from the BlendSpec) ────────────────────────────────────────────

def evaluate_g3(spec: anchor_blend.BlendSpec) -> dict[str, Any]:
    """G3: per fitted cell cal log-loss(blend) <= cal log-loss(anchor) (α=0 in the
    grid guarantees it), plus the post-fit per-survey pooled verification the
    BlendSpec records.  Terminates at α=0 ⇒ blend==anchor, so a genuine FAIL is
    only possible if the recorded ledger is internally inconsistent."""
    g3 = dict(spec.g3 or {})
    per_cell = g3.get("per_cell", {}) or {}
    per_survey = g3.get("per_survey_verify", {}) or {}
    cell_fail = [ck for ck, v in per_cell.items() if not v.get("pass", True)]
    # after the fallback ladder every survey terminates at cell/collapsed_survey/
    # collapsed_anchor — all of which satisfy blend<=anchor by construction.
    survey_verdicts = {sv: v.get("verdict") for sv, v in per_survey.items()}
    status = "PASS" if not cell_fail else "FAIL"
    return {
        "guard": "G3", "status": status,
        "n_cells": len(per_cell), "cells_failing_pre_fallback": cell_fail,
        "per_survey_verdicts": survey_verdicts,
    }


# ── cal frame helpers ────────────────────────────────────────────────────────

def _cal_conformal_rows(snap: pd.DataFrame, cal_ids: set[str]) -> pd.DataFrame:
    """One row per (object, n_det_bucket) at the bucket's max n_det on the
    calibration tier — the v8 conformal cal convention (object-level within each
    Mondrian bucket)."""
    cal = snap[
        snap["object_id"].astype(str).isin(cal_ids)
        & snap["target_class"].notna()
        & snap["label_quality"].astype(str).isin(CAL_QUALITIES)
    ].copy()
    if len(cal) == 0:
        return cal
    cal["n_det_bucket"] = ndet_bucket(cal["n_det"].to_numpy())
    cal = (
        cal.sort_values(["object_id", "n_det"])
        .groupby(["object_id", "n_det_bucket"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return cal


def _empirical_coverage(
    conformal, proba: np.ndarray, y: np.ndarray, strata: pd.DataFrame
) -> float | None:
    """Deployed-conformal empirical coverage on the cal set (the hard tripwire
    kept in eval; recorded here as an early read)."""
    try:
        sets = np.asarray(conformal.predict_sets(proba, strata), dtype=bool)
    except Exception:
        return None
    valid = (y >= 0) & (y < proba.shape[1])
    if valid.sum() == 0:
        return None
    covered = sets[np.arange(len(y))[valid], y[valid]]
    return float(covered.mean())


def _write_report(report: dict[str, Any], metrics_out: Path, t0: float) -> None:
    report["wall_clock_s"] = round(time.time() - t0, 1)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w") as fh:
        json.dump(_jsonify(report), fh, indent=2)


def drop_expert_columns(df: pd.DataFrame, experts: list[str]) -> pd.DataFrame:
    """Blank every column family of ``experts`` (incl. q__/q_prior__/
    trust_source__) on every row of a snapshot — the v13 ``--drop-expert``
    pre-Stage-A step.  Returns ``df`` unchanged when ``experts`` is empty."""
    if not experts:
        return df
    from debass_meta.models.multiclass_followup import blank_expert_blocks
    from debass_meta.projectors.base import sanitize_expert_key

    out = df.copy()
    blank_expert_blocks(out, np.ones(len(out), dtype=bool),
                        [sanitize_expert_key(k) for k in experts],
                        recompute_cross_traj=True, include_q_prior=True)
    return out


def drop_expert_helpfulness(helpfulness: pd.DataFrame, experts: list[str]) -> pd.DataFrame:
    """Helpfulness rows without ``experts`` (no trust head is trained for them)."""
    if not experts or "expert_key" not in helpfulness.columns:
        return helpfulness
    return helpfulness[~helpfulness["expert_key"].astype(str).isin(experts)].copy()


def _g8_dry_run_table(g8: dict[str, Any], min_corr: float = 0.1) -> str:
    lines = ["| survey | expert | corr final | corr production | avail SN | avail other | avail SN prod | avail other prod |",
             "|---|---|---|---|---|---|---|---|"]
    for survey, entry in g8.get("per_survey", {}).items():
        rows = sorted(entry.get("per_expert", {}).items(), key=lambda kv: -kv[1]["final"])
        for col, e in rows:
            if e["final"] < min_corr and (e["production"] or 0.0) < min_corr:
                continue
            f = lambda v: "—" if v is None else f"{v:.2f}"
            lines.append(f"| {survey} | {col[len('avail__'):]} | {f(e['final'])} | {f(e['production'])} | "
                         f"{f(e['avail_sn'])} | {f(e['avail_other'])} | {f(e['avail_sn_production'])} | "
                         f"{f(e['avail_other_production'])} |")
    return "\n".join(lines)


def _q_prior_audit(snap: pd.DataFrame, trust_dir: Path) -> dict[str, Any]:
    """Which ``q_prior__`` columns head 1 could consume vs which experts have a
    trained trust head (the scorer emits q_prior__ only for those; the rest
    would be NaN at serving — a train/serve skew the v13 Stage-A
    ``q_prior_experts='trained'`` closes).  Head-1 feature discovery
    (``_numeric_feature_cols``) admits every numeric non-constant column of the
    train snapshot, so a column absent at scoring CAN enter head 1 — this
    ledger makes the gap visible."""
    q_prior = sorted(c for c in snap.columns if c.startswith("q_prior__"))
    trained: list[str] = []
    try:
        meta_path = trust_dir / "pooled" / "metadata.json"
        if meta_path.exists():
            experts = json.loads(meta_path.read_text()).get("experts", [])
            trained = sorted(str(e) for e in (experts.keys() if isinstance(experts, dict) else experts))
        elif trust_dir.exists():
            trained = sorted(d.name for d in trust_dir.iterdir() if d.is_dir() and d.name != "pooled")
    except Exception:
        trained = []
    out: dict[str, Any] = {"n_q_prior_in_snapshot": len(q_prior), "n_trained_experts": len(trained)}
    if trained:
        from debass_meta.projectors.base import sanitize_expert_key

        trained_san = {sanitize_expert_key(e) for e in trained} | set(trained)
        out["not_emitted_at_scoring"] = [
            c for c in q_prior if c[len("q_prior__"):] not in trained_san]
    return out


# ── main orchestration ───────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train fusion_v11 (Stage A → hierarchical heads → anchored "
                    "blend → post-blend conformal → guards/gates)")
    p.add_argument("--snapshots", default="data/gold/object_epoch_snapshots_fusion_v11.parquet")
    p.add_argument("--helpfulness", default="data/gold/expert_helpfulness_fusion_v11.parquet")
    p.add_argument("--split", default="data/gold/split_fusion_v11.json")
    p.add_argument("--truth", default="data/truth/object_truth_v11.parquet",
                   help="v11 truth table (B0); Head-2 refuses to fit when absent")
    p.add_argument("--lsst-locked", default="data/gold/lsst_live_locked_test.json",
                   help="Locked LSST-live benchmark manifest (G6-at-train)")
    p.add_argument("--trust-dir", default="models/trust_fusion_v11")
    p.add_argument("--followup-dir", default="models/followup_fusion_v11")
    p.add_argument("--blend-dir", default="models/anchor_blend_v11")
    p.add_argument("--conformal-dir", default="models/conformal_fusion_v11")
    p.add_argument("--output-snapshots",
                   default="data/gold/object_epoch_snapshots_fusion_v11_trust.parquet",
                   help="Trust-augmented snapshot parquet (input to score/eval)")
    p.add_argument("--metrics-out", default="reports/metrics/fusion_v11_train.json")
    p.add_argument("--build-report", default="reports/metrics/fusion_v11_build.json",
                   help="Build report to echo the SCC all-negative census from (optional)")
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.1, help="Conformal miscoverage level")
    p.add_argument("--smoke", action="store_true",
                   help="<=200-object end-to-end run on local data")
    p.add_argument("--skip-stage-a", action="store_true",
                   help="Reuse q/q_prior columns from an existing --output-snapshots parquet")
    p.add_argument("--acknowledge-g2-unevaluable", action="store_true",
                   help="Continue past a G2_UNEVALUABLE verdict (stamps the report headline)")
    p.add_argument("--acknowledge-g7-not-evaluable", action="store_true",
                   help="Continue when G7 could not be enforced because gold lacks "
                        "the tns_type/bts_type provenance columns (stamps the report)")
    p.add_argument("--acknowledge-split-predates-manifest", action="store_true",
                   help="Continue when a locked manifest exists but the split-in-force "
                        "was built without --lsst-live-locked (G6 counterpart "
                        "quarantine never ran)")
    p.add_argument("--acknowledge-no-locked-manifest", action="store_true",
                   help="Continue when NO locked-test manifest exists at all "
                        "(benchmark quarantine + G6 cannot run — a ZTF-only or "
                        "smoke run). Without this flag a missing manifest is a "
                        "hard FAIL, not a silent no-op: on a fresh clone the "
                        "manifest lives under gitignored data/ and is generated "
                        "by the build job, so its absence signals a broken chain.")
    p.add_argument("--fdr-gamma", type=float, default=None,
                   help="Fit + record FDR thresholds on the deployed cal utilities")
    p.add_argument("--fdr-n-det-max", type=int, default=None)
    # head hyperparameters (gated defaults preregistered in HierarchicalFollowup)
    p.add_argument("--survey-cal-min", type=int, default=40)
    p.add_argument("--weak-weight", type=float, default=0.1)
    p.add_argument("--context-weight", type=float, default=0.15)
    p.add_argument("--tns-untyped-weight", type=float, default=0.6)
    p.add_argument("--bts-weight", type=float, default=1.0)
    p.add_argument("--head1-per-survey", action="store_true",
                   help="Request the per-survey head-1 realization (cal-gated; default POOLED)")
    p.add_argument("--head2-lsst-min", type=int, default=30)
    # ── fusion v13 (defaults reproduce v11/v12) ──────────────────────────
    v13 = p.add_argument_group("fusion v13")
    v13.add_argument("--head1-exclude-quality", action="append", default=[],
                     metavar="[SURVEY:]QUALITY",
                     help="B1: drop this label tier from head-1 training, the head-1 "
                          "calibrators, the per-survey gate and the α honesty filter. "
                          "'lsst:weak' = LSST weak rows only (ZTF weak rows are ZTF's "
                          "only 'other' labels); 'weak' = every survey. Repeatable.")
    v13.add_argument("--no-lsst-equalization", action="store_true",
                     help="B2: switch off the head-1 LSST equalization that masked the "
                          "whole ALeRCE family (incl. local alerce_lc) on every LSST "
                          "weak+context row (class-pure once weak rows are excluded)")
    v13.add_argument("--head1-context-mask", choices=("rows", "survey"), default="rows",
                     help="B2: 'rows' (v11/v12) masks lasair/sherlock + babamul only on "
                          "context-labelled rows; 'survey' masks them on every head-1 row "
                          "of a survey that has context labels (not class-pure)")
    v13.add_argument("--head1-survey-mask", action="append", default=[],
                     metavar="SURVEY:EXPERT",
                     help="B2: mask EXPERT on every head-1 row of SURVEY (for genuine "
                          "availability gaps in the gold that G8 names). Repeatable.")
    v13.add_argument("--g8-max-corr", type=float, default=None,
                     help="Guard G8: fail when any expert's weighted |corr(avail, is_SN)| "
                          "on the final head-1 LSST/ZTF train frame exceeds this "
                          "(plan default 0.2; unset = guard off)")
    v13.add_argument("--acknowledge-g8", action="store_true",
                     help="Record a G8 violation as OVERRIDDEN and continue")
    v13.add_argument("--head1-dropout", action="store_true",
                     help="B3: availability-dropout copies for head 1 (and head 2 on its "
                          "own rows); object totals unchanged")
    v13.add_argument("--dropout-aug-weight", type=float, default=0.1,
                     help="B3: weight share each dropout copy takes from its source row")
    v13.add_argument("--dropout-random-frac", type=float, default=0.25,
                     help="B3: fraction of rows (all surveys) getting one random-subset-drop copy")
    v13.add_argument("--dropout-regimes", default="no_broker,no_local,no_expert",
                     help="B3: structured regimes applied to LSST rows (comma list)")
    v13.add_argument("--dropout-keep-one-frac", type=float, default=0.0,
                     help="B3: share of random copies that keep exactly one expert")
    v13.add_argument("--cross-fit-folds", type=int, default=0,
                     help="B4: GroupKFold folds for OOF head predictions on train "
                          "(0 = in-sample, v11/v12). Calibrators fit on OOF-train ∪ cal; "
                          "α and G2 evaluated on OOF-train ∪ cal")
    v13.add_argument("--head1-cal-weights", choices=("train", "object"), default="train",
                     help="v13b: weights of the cross-fitted head-1 calibrators. 'train' = head-1 "
                          "training weights (v13); 'object' = without the label-quality factor, "
                          "so calibrated P(SN) follows the object mix (context rows count fully)")
    v13.add_argument("--stage-a-weak-policy",
                     choices=("all", "is_sn_only", "lsst_is_sn_only", "none"), default=None,
                     help="Passed to train_pooled_trust(weak_policy=...) when supported")
    v13.add_argument("--drop-expert", action="append", default=[], metavar="KEY",
                     help="Drop this expert from the whole stack: its helpfulness rows are "
                          "removed and its snapshot columns blanked before Stage A (no trust "
                          "head, no q__/q_prior__), both heads blank its block at fit and at "
                          "serve, and the BlendSpec excludes it from the anchor. Repeatable.")
    v13.add_argument("--g8-dry-run", action="store_true",
                     help="Build the final head-1 frames with every v13 setting (Stage A "
                          "skipped; raw --snapshots gold), evaluate G8 on LSST and ZTF, write "
                          "the report and exit before any model fitting")
    v13.add_argument("--stage-a-q-prior-experts", choices=("all", "trained"), default=None,
                     help="Passed to train_pooled_trust(q_prior_experts=...) when supported")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    t0 = time.time()
    report: dict[str, Any] = {
        "pipeline": "fusion_v11",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": {k: str(v) for k, v in vars(args).items()},
        "guards": {},
        "gates": [],
        "eval_only_guards": {
            "G1": "ZTF locked-test macro OvR AUC vs v10 CI — EVAL STAGE ONLY",
            "G4": "LSST-live benchmark SN-vs-other AUC vs anchor — EVAL STAGE ONLY",
        },
    }

    # ── split (locked; asserted disjoint) ───────────────────────────────────
    train_ids, cal_ids, test_ids, _split_raw = load_split_manifest(Path(args.split))
    print(f"Split: train={len(train_ids):,} cal={len(cal_ids):,} test={len(test_ids):,}")
    report["split"] = {"source": args.split, "n_train": len(train_ids),
                       "n_cal": len(cal_ids), "n_test": len(test_ids)}

    # ── guard G6 (at train): locked benchmark ∩ (train ∪ cal) = ∅ ────────────
    locked_path = Path(args.lsst_locked)
    locked_ids = load_locked_test_ids(locked_path)
    if not locked_ids:
        # A MISSING manifest means the benchmark quarantine (build) and G6 (train)
        # never ran. The manifest lives under gitignored data/ and does NOT ship
        # via git — the build job regenerates it (cohort-clean) — so its absence
        # here signals a broken chain, NOT a benign no-op. Fail loud unless the
        # operator explicitly opts into a ZTF-only / smoke run.
        acked = bool(args.acknowledge_no_locked_manifest)
        report["guards"]["G6"] = {
            "guard": "G6", "status": "SKIPPED" if acked else "FAIL",
            "reason": f"no locked-test manifest at {locked_path}",
            "acknowledged": acked,
        }
        if not acked:
            raise SystemExit(
                f"G6 SETUP ERROR: no locked-test manifest at {locked_path}. The "
                "LSST-live benchmark quarantine never ran (the manifest is "
                "gitignored and generated by jobs/run_fusion_v11_build.sh's "
                "cohort-clean step — scripts/build_truth_lsst_live.py "
                "--mode cohort-clean). Regenerate it, or pass "
                "--acknowledge-no-locked-manifest for a ZTF-only run.")
        print(f"  G6: no locked-test manifest at {locked_path} — SKIPPED (acknowledged)")
    else:
        # A manifest EXISTS on disk.  If the split-in-force was built WITHOUT it
        # (no --lsst-live-locked at build), the id-level check below still passes
        # trivially (locked LSST ids usually have no local gold row) while the
        # association-counterpart quarantine / SSL / seq exclusions that key off
        # quarantined_ids never ran.  That is a silent G6 no-op — refuse it.
        armed = bool(_split_raw.get("lsst_live_locked_armed"))
        if not armed and not args.acknowledge_split_predates_manifest:
            raise SystemExit(
                f"G6 SETUP ERROR: locked manifest {locked_path} exists but the "
                f"split {args.split} was built WITHOUT --lsst-live-locked "
                "(lsst_live_locked_armed is false/absent). The build's "
                "counterpart quarantine never ran — rebuild the split with "
                "--lsst-live-locked, or pass "
                "--acknowledge-split-predates-manifest to override.")
        overlap = locked_ids & (train_ids | cal_ids)
        report["guards"]["G6"] = {
            "guard": "G6", "status": "PASS" if not overlap else "FAIL",
            "n_locked_test": len(locked_ids), "n_overlap": len(overlap),
            "split_armed_at_build": armed,
            "overlap_sample": sorted(overlap)[:10],
        }
        assert not overlap, (
            f"G6 violation: {len(overlap)} locked-benchmark ids in train∪cal "
            f"(e.g. {sorted(overlap)[:5]})")
        print(f"  G6 PASS: locked-test ({len(locked_ids)}) disjoint from train∪cal")

    drop_experts = [str(k) for k in args.drop_expert]
    report["drop_experts"] = drop_experts

    # ── v13 G8 dry run: final head-1 frames only, no Stage A, no fitting ─────
    if args.g8_dry_run:
        from debass_meta.models.hierarchical_followup import (
            AvailabilityDropout, HierarchicalFollowup,
        )

        dry_src = Path(args.output_snapshots) if args.skip_stage_a else Path(args.snapshots)
        snapshots = pd.read_parquet(dry_src)
        print(f"G8 dry run: loaded {len(snapshots):,} snapshot rows from {dry_src}")
        if args.smoke:
            snapshots, _ = _smoke_subset(snapshots, None, train_ids, cal_ids, test_ids,
                                         seed=args.seed)
        snapshots = drop_expert_columns(snapshots, drop_experts)
        dropout = None
        if args.head1_dropout:
            dropout = AvailabilityDropout(
                regimes=tuple(r.strip() for r in str(args.dropout_regimes).split(",") if r.strip()),
                regime_weight=float(args.dropout_aug_weight),
                random_frac=float(args.dropout_random_frac),
                keep_one_frac=float(args.dropout_keep_one_frac), seed=int(args.seed))
        survey_masks_dr: dict[str, tuple[str, ...]] = {}
        for spec_ in args.head1_survey_mask:
            sv_, _, ex_ = str(spec_).partition(":")
            survey_masks_dr[sv_.lower()] = tuple(list(survey_masks_dr.get(sv_.lower(), ())) + [ex_])
        head = HierarchicalFollowup(
            weak_weight=args.weak_weight, context_weight=args.context_weight,
            tns_untyped_weight=args.tns_untyped_weight, bts_weight=args.bts_weight,
            head1_per_survey=bool(args.head1_per_survey), seed=args.seed,
            head1_exclude_qualities=tuple(str(q) for q in args.head1_exclude_quality),
            equalize_lsst_provenance=not args.no_lsst_equalization,
            context_mask_scope=args.head1_context_mask, head1_survey_masks=survey_masks_dr,
            dropout=dropout, drop_experts=tuple(drop_experts),
            g8_max_corr=args.g8_max_corr if args.g8_max_corr is not None else 0.2,
        )
        g8 = head.g8_dry_run(snapshots, train_ids, cal_ids)
        report["guards"]["G8"] = g8
        report["v13"] = g8["v13"]
        report["g8_dry_run_table"] = _g8_dry_run_table(g8)
        report["guard_statuses"] = {g: v.get("status") for g, v in report["guards"].items()}
        _write_report(report, Path(args.metrics_out), t0)
        print(f"G8 dry run: {g8['status']} (threshold {g8['threshold']}, "
              f"max |corr|={g8['max_abs_corr']}, {len(g8['violations'])} violations, "
              f"head-1 rows={g8['n_head1_rows']})")
        print(report["g8_dry_run_table"])
        print(f"Wrote dry-run report → {args.metrics_out}")
        return 0

    # ── truth-file ordering guard (B0): Head-2 requires object_truth_v11 ─────
    truth_path = Path(args.truth)
    if not truth_path.exists():
        raise SystemExit(
            f"--truth {truth_path} is absent — Head-2 refuses to fit before P1's "
            "rederive lands (B0 ordering). Run scripts/rederive_spec_truth.py first.")
    report["truth"] = {"path": str(truth_path)}

    # ── Stage A: pooled trust (reuse or skip) ────────────────────────────────
    output_snapshots = Path(args.output_snapshots)
    if args.skip_stage_a:
        if not output_snapshots.exists():
            raise SystemExit(f"--skip-stage-a but {output_snapshots} does not exist")
        snap_trust = pd.read_parquet(output_snapshots)
        print(f"Stage A skipped — reusing q columns from {output_snapshots} "
              f"({len(snap_trust):,} rows)")
        if args.smoke:
            snap_trust, _ = _smoke_subset(snap_trust, None, train_ids, cal_ids,
                                          test_ids, seed=args.seed)
        snap_trust = drop_expert_columns(snap_trust, drop_experts)
        report["stage_a"] = {"skipped": True, "reused_snapshot": str(output_snapshots)}
    else:
        import inspect

        from debass_meta.models.pooled_trust import train_pooled_trust

        snapshots = pd.read_parquet(args.snapshots)
        helpfulness = pd.read_parquet(args.helpfulness)
        print(f"Loaded {len(snapshots):,} snapshot rows "
              f"({snapshots['object_id'].nunique():,} objects) + "
              f"{len(helpfulness):,} helpfulness rows")
        leak = [c for c in helpfulness.columns if c.startswith(("q__", "trust_source__"))]
        assert not leak, f"Stage-A leakage: q__/trust_source__ in helpfulness: {leak}"
        if args.smoke:
            snapshots, helpfulness = _smoke_subset(
                snapshots, helpfulness, train_ids, cal_ids, test_ids, seed=args.seed)
        if drop_experts:
            n_h = len(helpfulness)
            helpfulness = drop_expert_helpfulness(helpfulness, drop_experts)
            snapshots = drop_expert_columns(snapshots, drop_experts)
            print(f"  --drop-expert {drop_experts}: removed {n_h - len(helpfulness):,} "
                  "helpfulness rows and blanked the snapshot columns before Stage A")
        print("Stage A: pooled trust (GroupKFold OOF on train; refit for cal/test)…")
        kwargs: dict[str, Any] = {"n_jobs": args.n_jobs, "seed": args.seed}
        stage_a_passthrough: dict[str, Any] = {}
        try:
            params = inspect.signature(train_pooled_trust).parameters
            if "grid_small" in params:
                kwargs["grid_small"] = bool(args.smoke)
            # v13 passthroughs — only when the (concurrently developed) Stage-A
            # signature exposes them; recorded either way.
            for flag, kw in (("stage_a_weak_policy", "weak_policy"),
                             ("stage_a_q_prior_experts", "q_prior_experts")):
                val = getattr(args, flag)
                if val is None:
                    continue
                if kw in params:
                    kwargs[kw] = val
                    stage_a_passthrough[kw] = val
                else:
                    stage_a_passthrough[kw] = f"NOT PASSED (train_pooled_trust has no '{kw}')"
                    print(f"  [warn] --{flag.replace('_', '-')}={val} requested but "
                          f"train_pooled_trust has no '{kw}' kwarg — ignored")
        except (TypeError, ValueError):
            pass
        result = train_pooled_trust(
            helpfulness, snapshots, train_ids, cal_ids, test_ids,
            str(args.trust_dir), **kwargs)
        snap_trust = result.snapshots
        output_snapshots.parent.mkdir(parents=True, exist_ok=True)
        snap_trust.to_parquet(output_snapshots, index=False)
        print(f"Wrote trust-augmented snapshot → {output_snapshots}")
        report["stage_a"] = {"skipped": False, "artifact_dir": result.artifact_dir,
                             "metrics": result.metrics,
                             "v13_passthrough": stage_a_passthrough}

    n_q = sum(1 for c in snap_trust.columns if c.startswith("q__"))
    print(f"  snapshot has {n_q} q__ columns")
    report["q_prior_columns"] = _q_prior_audit(snap_trust, Path(args.trust_dir))
    if report["q_prior_columns"].get("not_emitted_at_scoring"):
        print(f"  [warn] {len(report['q_prior_columns']['not_emitted_at_scoring'])} q_prior__ "
              "columns in the train snapshot have no trained trust head — the scorer "
              "will not emit them (NaN at serving): "
              f"{report['q_prior_columns']['not_emitted_at_scoring'][:6]}…")

    # ── P3: hierarchical heads (G7 inside fit) + per-head calibration ────────
    from debass_meta.models.hierarchical_followup import (
        AvailabilityDropout, G8Error, HierarchicalFollowup,
    )

    # v13 settings (recorded verbatim in the report)
    exclude_q = tuple(str(q) for q in args.head1_exclude_quality)
    survey_masks: dict[str, tuple[str, ...]] = {}
    for spec_ in args.head1_survey_mask:
        sv_, _, ex_ = str(spec_).partition(":")
        if not ex_:
            raise SystemExit(f"--head1-survey-mask expects SURVEY:EXPERT, got {spec_!r}")
        survey_masks[sv_.lower()] = tuple(list(survey_masks.get(sv_.lower(), ())) + [ex_])
    dropout = None
    if args.head1_dropout:
        dropout = AvailabilityDropout(
            regimes=tuple(r.strip() for r in str(args.dropout_regimes).split(",") if r.strip()),
            regime_weight=float(args.dropout_aug_weight),
            random_frac=float(args.dropout_random_frac),
            keep_one_frac=float(args.dropout_keep_one_frac),
            seed=int(args.seed),
        )
    report["v13"] = {
        "head1_exclude_qualities": list(exclude_q),
        "equalize_lsst_provenance": not args.no_lsst_equalization,
        "context_mask_scope": args.head1_context_mask,
        "head1_survey_masks": {k: list(v) for k, v in survey_masks.items()},
        "dropout": dropout.to_dict() if dropout else None,
        "cross_fit_folds": int(args.cross_fit_folds),
        "g8_max_corr": args.g8_max_corr,
        "g8_acknowledged": bool(args.acknowledge_g8),
        "stage_a_weak_policy": args.stage_a_weak_policy,
        "stage_a_q_prior_experts": args.stage_a_q_prior_experts,
        "drop_experts": drop_experts,
    }

    print("P3: fitting HierarchicalFollowup (heads + per-survey calibration)…")
    head = HierarchicalFollowup(
        survey_cal_min=args.survey_cal_min,
        weak_weight=args.weak_weight, context_weight=args.context_weight,
        tns_untyped_weight=args.tns_untyped_weight, bts_weight=args.bts_weight,
        head1_per_survey=bool(args.head1_per_survey),
        head2_lsst_min=args.head2_lsst_min,
        grid_small=bool(args.smoke), n_jobs=args.n_jobs, seed=args.seed,
        head1_exclude_qualities=exclude_q,
        equalize_lsst_provenance=not args.no_lsst_equalization,
        context_mask_scope=args.head1_context_mask,
        head1_survey_masks=survey_masks,
        dropout=dropout,
        cross_fit_folds=int(args.cross_fit_folds),
        g8_max_corr=args.g8_max_corr,
        g8_override=bool(args.acknowledge_g8),
        drop_experts=tuple(drop_experts),
        head1_cal_weights=str(args.head1_cal_weights),
    )
    try:
        head.fit(snap_trust, train_ids, cal_ids, test_ids)
    except G8Error as exc:
        report["guards"]["G8"] = {"guard": "G8", "status": "FAIL",
                                  "threshold": args.g8_max_corr, "message": str(exc)}
        report["guard_statuses"] = {g: v.get("status") for g, v in report["guards"].items()}
        _write_report(report, Path(args.metrics_out), t0)
        raise SystemExit(f"{exc}\nPass --acknowledge-g8 to record the violation and continue.")
    head.save(str(args.followup_dir))
    head_report = dict(head.report_)
    if args.g8_max_corr is not None:
        g8 = dict(head_report.get("g8", {}))
        report["guards"]["G8"] = {
            "guard": "G8", "status": g8.get("status", "FAIL"),
            "threshold": g8.get("threshold"), "max_abs_corr": g8.get("max_abs_corr"),
            "violations": g8.get("violations", []),
            "per_survey": g8.get("per_survey", {}),
            "acknowledged": bool(args.acknowledge_g8),
        }
        print(f"  G8: {g8.get('status')} (max |corr|={g8.get('max_abs_corr')}, "
              f"threshold={g8.get('threshold')}, violations={len(g8.get('violations', []))})")
    for key in ("head1_dropout", "head2_dropout", "cross_fit_head1", "cross_fit_head2",
                "head1_calibration_crossfit", "head2_calibration_crossfit",
                "head1_excluded_train_rows"):
        if key in head_report:
            report[key] = head_report[key]
    # G7 is ENFORCED inside fit() by a hard assert (untyped-provenance == 0):
    # reaching here means it did not violate.  But when the head-2 frame lacks
    # the provenance columns entirely the assert is gated OFF and fit records
    # ``status='not_evaluable_no_provenance_cols'`` — the exact failure mode
    # G7 exists to catch (gold built on pre-B0 truth).  Read the real status
    # from the fit report and FAIL LOUD if the guard could not be evaluated,
    # rather than stamping an unconditional PASS.
    g7_detail = dict(head_report.get("g7", {}))
    g7_fit_status = str(g7_detail.get("status", "missing"))
    if g7_fit_status == "enforced":
        g7_status = "PASS"
    else:
        g7_status = "FAIL"
    report["guards"]["G7"] = {
        "guard": "G7", "status": g7_status,
        "fit_status": g7_fit_status,
        "head2_n_train_rows": head_report.get("head2_n_train_rows"),
        "n_untyped_provenance": g7_detail.get("n_untyped_provenance"),
        "detail": "Head-2 admits only concrete-subtype (tns_type/bts_type) rows; "
                  "G7 requires the provenance columns to be present in gold (B0).",
    }
    if g7_status != "PASS" and not args.acknowledge_g7_not_evaluable:
        raise SystemExit(
            f"G7 {g7_status}: head-2 provenance guard reported "
            f"'{g7_fit_status}'. Gold must carry tns_type/bts_type "
            "(rebuild with the v11 builder) so untyped-provenance rows can be "
            "rejected. Pass --acknowledge-g7-not-evaluable to continue anyway.")
    report["gates"] = list(head_report.get("gate_verdicts", []))
    report["availability_audit"] = head_report.get("availability_audit", {})
    report["head1_provenance_masking"] = head_report.get("head1_provenance_masking", {})
    report["head2"] = {
        "n_train_rows": head_report.get("head2_n_train_rows"),
        "train_surveys": head_report.get("head2_train_surveys"),
        "lsst_enabled": head.head2_lsst_enabled,
        "base_rate": head.base_rate, "base_rate_global": head.base_rate_global,
    }
    print(f"  head1 mode={head.head1_mode}; head2 surveys={head.head2_surveys}; "
          f"head2_lsst_enabled={head.head2_lsst_enabled}")

    # ── P4: anchor + α on CAL (G3 recorded) ──────────────────────────────────
    labelled = snap_trust[snap_trust["target_class"].isin(CLASSES)].copy()
    labelled["object_id"] = labelled["object_id"].astype(str)
    cal_df = labelled[labelled["object_id"].isin(cal_ids)].copy()
    if len(cal_df) == 0:
        raise SystemExit("No calibration rows with a usable target_class — cannot fit α")
    p_cal_model = np.asarray(head.predict_proba(cal_df), dtype=float)
    for i, name in enumerate(_MODEL_COLS):
        cal_df[name] = p_cal_model[:, i]

    blend_dir = Path(args.blend_dir)
    alpha_kwargs: dict[str, Any] = {}
    if exclude_q:
        alpha_kwargs["exclude_qualities"] = exclude_q
    if drop_experts:
        alpha_kwargs["drop_experts"] = tuple(drop_experts)
    if head.oof_frame_ is not None:
        # B4 (v13): α on OOF-train ∪ cal, both in the availability-dropout
        # mixture, weighted by the object-normalized sample weights.  The train
        # rows carry head-1's provenance-masked features (the masked context
        # family is anchor-excluded anyway).
        train_mix = head.oof_frame_.copy()
        p_tr = np.asarray(head.predict_proba_oof(train_mix), dtype=float)
        for i, name in enumerate(_MODEL_COLS):
            train_mix[name] = p_tr[:, i]
        cal_mix, w_cal_mix = head.calibration_mixture(cal_df.drop(columns=list(_MODEL_COLS)))
        cal_mix["sample_weight"] = w_cal_mix
        p_cm = np.asarray(head.predict_proba(cal_mix), dtype=float)
        for i, name in enumerate(_MODEL_COLS):
            cal_mix[name] = p_cm[:, i]
        alpha_df = pd.concat([train_mix, cal_mix], ignore_index=True, sort=False)
        alpha_kwargs["weight_col"] = "sample_weight"
        print(f"P4: fitting anchored-blend α table on OOF-train ∪ cal "
              f"({len(train_mix):,} + {len(cal_mix):,} rows, weighted; honesty-filtered)…")
        report["alpha_fit_frame"] = {
            "mode": "oof_train_mixture+cal_mixture", "n_train_rows": int(len(train_mix)),
            "n_cal_rows": int(len(cal_mix)), **head.oof_coverage(train_mix)}
    else:
        alpha_df = cal_df
        print("P4: fitting anchored-blend α table on cal (honesty-filtered)…")
        report["alpha_fit_frame"] = {"mode": "cal", "n_cal_rows": int(len(cal_df))}
    spec = anchor_blend.fit_alpha(alpha_df, out_dir=blend_dir, **alpha_kwargs)
    report["guards"]["G3"] = evaluate_g3(spec)
    # α fallback ledger + per-cell n after honesty filtering
    cal_blended = anchor_blend.apply(
        anchor_blend.compute_anchor(
            cal_df, base_rate_by_survey=spec.base_rates,
            default_base_rate=spec.default_base_rate),
        spec)
    report["blend"] = {
        "blend_dir": str(blend_dir),
        "base_rates": spec.base_rates,
        "alpha_cells": spec.alpha_cells,
        "alpha_survey": spec.alpha_survey,
        "alpha_global": spec.alpha_global,
        "cal_alpha_fallback_ledger":
            cal_blended["alpha_fallback_level"].value_counts().to_dict(),
        "per_cell_n_after_honesty": {ck: v.get("n") for ck, v in spec.alpha_cells.items()},
    }
    print(f"  wrote BlendSpec → {blend_dir} "
          f"(cells={len(spec.alpha_cells)}, fallback="
          f"{report['blend']['cal_alpha_fallback_ledger']})")

    # ── conformal on POST-BLEND deployed probabilities ───────────────────────
    from debass_meta.models.conformal import MondrianAPS

    conf_cal = _cal_conformal_rows(snap_trust, cal_ids)
    conformal_dir = Path(args.conformal_dir)
    conformal_dir.mkdir(parents=True, exist_ok=True)
    conformal_path = conformal_dir / "mondrian_aps.pkl"
    if len(conf_cal) == 0:
        raise SystemExit("No calibration rows for conformal fit")
    p_conf_model = np.asarray(head.predict_proba(conf_cal), dtype=float)
    for i, name in enumerate(_MODEL_COLS):
        conf_cal[name] = p_conf_model[:, i]
    conf_cal = anchor_blend.apply(
        anchor_blend.compute_anchor(
            conf_cal, base_rate_by_survey=spec.base_rates,
            default_base_rate=spec.default_base_rate),
        spec)
    p_deploy_cal = conf_cal[list(_MODEL_COLS)].to_numpy(dtype=float)
    y_cal = class_index(conf_cal["target_class"])
    strata_cal = pd.DataFrame({
        "survey": conf_cal["survey"].astype(str).str.lower().to_numpy(),
        "n_det": conf_cal["n_det"].astype(int).to_numpy(),
        "n_det_bucket": ndet_bucket(conf_cal["n_det"].to_numpy()),
    })
    conformal = MondrianAPS()
    conformal.fit(p_deploy_cal, y_cal, strata_cal, alpha=args.alpha)
    conformal.save(str(conformal_path))
    emp_cov = _empirical_coverage(conformal, p_deploy_cal, y_cal, strata_cal)
    report["conformal"] = {
        "path": str(conformal_path), "alpha": args.alpha,
        "n_cal_rows": int(len(conf_cal)),
        "empirical_coverage_cal": emp_cov,
        "target_coverage": 1.0 - args.alpha,
        "coverage_tripwire_note":
            "hard eval check on both locked tests; cal read shown here",
    }
    print(f"Wrote conformal → {conformal_path} (cal empirical coverage={emp_cov})")

    # ── guard G2 on the deployed CAL/OOF-train LSST spec-Ia union ────────────
    # Attach deployed p_snia to the full labelled snapshot for the G2 union.
    # v13 cross-fit: train rows use the OUT-OF-FOLD heads (the v12 G2 scored
    # train rows in-sample); cal/test rows the deployed heads.
    scored = labelled.copy()
    if head.oof_frame_ is not None:
        p_full_model = np.asarray(head.predict_proba_oof(scored), dtype=float)
        report["g2_frame"] = {"mode": "oof_train+cal", **head.oof_coverage(scored)}
    else:
        p_full_model = np.asarray(head.predict_proba(scored), dtype=float)
        report["g2_frame"] = {"mode": "in_sample_train+cal"}
    for i, name in enumerate(_MODEL_COLS):
        scored[name] = p_full_model[:, i]
    scored = anchor_blend.apply(
        anchor_blend.compute_anchor(
            scored, base_rate_by_survey=spec.base_rates,
            default_base_rate=spec.default_base_rate),
        spec)
    g2 = evaluate_g2(scored, train_ids, cal_ids,
                     acknowledge_unevaluable=args.acknowledge_g2_unevaluable,
                     test_ids=test_ids)
    report["guards"]["G2"] = g2
    print(f"  G2: {g2['status']} "
          f"(n={g2.get('n_rows')}, median={g2.get('median_p_snia')}, "
          f"max={g2.get('max_p_snia')})")

    # ── FDR ledger (deployed cal utilities; scorer refits at score time) ─────
    if args.fdr_gamma is not None:
        try:
            from debass_meta.models.selection import UTILITY_PRESETS

            from scripts.score_fusion_v8 import GOAL_SCORE_COLS, compute_fdr_thresholds
            for goal, s_col in GOAL_SCORE_COLS.items():
                u = np.asarray(UTILITY_PRESETS[goal], dtype=float)
                scored[s_col] = scored[list(_MODEL_COLS)].to_numpy(float) @ u
            fdr = compute_fdr_thresholds(
                scored, Path(args.split), float(args.fdr_gamma),
                n_det_max=args.fdr_n_det_max)
            report["fdr"] = {"gamma": args.fdr_gamma, "thresholds": fdr}
        except Exception as exc:  # best-effort ledger; the scorer is authoritative
            report["fdr"] = {"gamma": args.fdr_gamma, "error": str(exc)}

    # ── seq_v11 in/out gate (spec §5; §2.4 preregistration) ──────────────────
    has_seq_v11 = any(c.startswith("proj__seq_v11__") for c in snap_trust.columns)
    report["gates"].append({
        "gate": "seq_v11_in_out",
        "decision": "in" if has_seq_v11 else "out",
        "detail": ("proj__seq_v11__ columns present in gold; feeds heads + anchor"
                   if has_seq_v11 else
                   "no proj__seq_v11__ columns — seq arm OUT of the deployed blend "
                   "(standalone benchmark diagnostics at eval time only)"),
    })

    # ── SCC negativity census echo (from the build report, if present) ───────
    build_report_path = Path(args.build_report)
    if build_report_path.exists():
        try:
            br = json.loads(build_report_path.read_text())
            report["negativity_census_echo"] = br.get(
                "all_negative_census", br.get("negativity_census"))
        except Exception as exc:
            report["negativity_census_echo"] = {"error": str(exc)}

    # ── overall guard status + exit code ─────────────────────────────────────
    guard_statuses = {g: v.get("status") for g, v in report["guards"].items()}
    report["guard_statuses"] = guard_statuses
    failed = [g for g, s in guard_statuses.items() if s == "FAIL"]
    g2_unevaluable = guard_statuses.get("G2") == "G2_UNEVALUABLE"

    report["paths"] = {
        "snapshot_trust": str(output_snapshots), "trust_dir": str(args.trust_dir),
        "followup_dir": str(args.followup_dir), "blend_dir": str(blend_dir),
        "conformal": str(conformal_path),
    }
    metrics_out = Path(args.metrics_out)
    _write_report(report, metrics_out, t0)
    print(f"Wrote train report → {metrics_out} ({report['wall_clock_s']} s)")
    print(f"Guard statuses: {guard_statuses}")

    if failed:
        print(f"GUARD FAILURE: {failed} — nonzero exit")
        return 1
    if g2_unevaluable and not args.acknowledge_g2_unevaluable:
        print("G2_UNEVALUABLE (n<10) and not acknowledged — nonzero exit "
              "(pass --acknowledge-g2-unevaluable to continue)")
        return 1
    if g2_unevaluable:
        print("G2_UNEVALUABLE acknowledged — HEADLINE STAMPED: G2 could not be evaluated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
