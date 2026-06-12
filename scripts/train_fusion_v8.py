#!/usr/bin/env python3
"""fusion_v8 training orchestrator — Stage A pooled trust → Stage B multiclass → conformal.

Pipeline (design §5.1, spec-pinned interfaces):
  1. Load gold snapshot + long-format helpfulness + locked split manifest.
  2. Stage A: ``train_pooled_trust`` (GroupKFold OOF on train, refit for cal/test,
     hierarchical per-expert calibration) → write q__/q_prior__/trust_source__ into the
     snapshot → persist the trust-augmented snapshot parquet.
  3. Component gates (decided on CAL ONLY, design §8.1): forward-additive chain
     ext_features → traj_features → pooled_trust_q → expert_dropout_aug.  A failing
     component is dropped before the final model is fit.  Test is never consulted for
     gate decisions (anti-peeking; the headline is pre-registered, spec correction #4b).
  4. Stage B: ``train_multiclass_followup`` on the gated column set.
  5. Optional bagging-seed variants (guard #4b: seed spread on the headline < 0.02;
     the variants are SAVED here and SCORED only by eval_fusion_v8.py so all
     test-touching happens in one place).
  6. Mondrian split-conformal APS fit on cal → models/conformal_fusion_v8/.

No-leakage argument: the split manifest is loaded verbatim and asserted disjoint;
cal is used exclusively for gates / Dirichlet calibration / conformal quantiles
(never model fitting — that is enforced inside the pinned G3/G4 modules); gate
decisions read only train+cal rows.

Outputs:
  models/trust_fusion_v8/                      (Stage A, written by train_pooled_trust)
  models/followup_fusion_v8/                   (Stage B artifact + gate/seed variants)
  models/conformal_fusion_v8/mondrian_aps.pkl
  data/gold/object_epoch_snapshots_fusion_v8_trust.parquet
  reports/metrics/fusion_v8_train.json         (metrics + component-gate ledger)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

try:
    from debass_meta.models.multiclass_followup import CLASSES
except Exception:  # pragma: no cover — G4 module built in parallel
    CLASSES = ("snia", "nonIa_snlike", "other")

SPEC_QUALITIES = ("spectroscopic",)
CAL_QUALITIES = ("spectroscopic", "tns_untyped")
NDET_BUCKETS = ((1, 4, "1-4"), (5, 9, "5-9"), (10, 20, "10-20"))


# ── small shared helpers ─────────────────────────────────────────────────────

def _jsonify(obj: Any) -> Any:
    """Recursively coerce numpy/pandas scalars so json.dump never chokes."""
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_jsonify(v) for v in obj.tolist()]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def load_split_manifest(path: Path) -> tuple[set[str], set[str], set[str], dict]:
    """Load data/gold/split_fusion_v8.json (G5 output).

    Tolerant to key spellings {train_ids,cal_ids,test_ids} / {train,cal,test},
    possibly nested under "split". Asserts the three sets are pairwise disjoint.
    """
    with open(path) as fh:
        raw = json.load(fh)
    node = raw.get("split", raw) if isinstance(raw, dict) else raw
    out: list[set[str]] = []
    for part in ("train", "cal", "test"):
        ids = node.get(f"{part}_ids", node.get(part))
        if ids is None:
            raise KeyError(f"Split manifest {path} missing '{part}_ids' (or '{part}')")
        out.append({str(x) for x in ids})
    train_ids, cal_ids, test_ids = out
    assert not (train_ids & cal_ids), "split integrity: train/cal overlap"
    assert not (train_ids & test_ids), "split integrity: train/test overlap"
    assert not (cal_ids & test_ids), "split integrity: cal/test overlap"
    return train_ids, cal_ids, test_ids, raw


def ndet_bucket(n_det: np.ndarray) -> np.ndarray:
    """Map n_det to the Mondrian bucket labels {1-4, 5-9, 10-20} (design §4.2)."""
    n = np.asarray(n_det, dtype=float)
    out = np.full(n.shape, "10-20", dtype=object)
    out[(n >= 1) & (n <= 4)] = "1-4"
    out[(n >= 5) & (n <= 9)] = "5-9"
    return out


def class_index(target_class: pd.Series) -> np.ndarray:
    """Map target_class strings → CLASSES indices (-1 where unknown)."""
    mapping = {c: i for i, c in enumerate(CLASSES)}
    return target_class.astype(str).map(mapping).fillna(-1).astype(int).to_numpy()


def _smoke_subset(
    snapshots: pd.DataFrame,
    helpfulness: pd.DataFrame | None,
    train_ids: set[str],
    cal_ids: set[str],
    test_ids: set[str],
    *,
    max_objects: int = 200,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Deterministic ≤max_objects subset preserving split membership ratios."""
    rng = np.random.default_rng(seed)
    present = set(snapshots["object_id"].astype(str).unique())
    quotas = (
        (sorted(present & train_ids), int(max_objects * 0.6)),
        (sorted(present & cal_ids), int(max_objects * 0.2)),
        (sorted(present & test_ids), int(max_objects * 0.2)),
    )
    keep: set[str] = set()
    for ids, quota in quotas:
        if not ids:
            continue
        take = min(quota, len(ids))
        keep.update(rng.choice(np.asarray(ids, dtype=object), size=take, replace=False).tolist())
    snap = snapshots[snapshots["object_id"].astype(str).isin(keep)].copy()
    help_df = None
    if helpfulness is not None:
        help_df = helpfulness[helpfulness["object_id"].astype(str).isin(keep)].copy()
    print(f"  --smoke: subset to {len(keep)} objects / {len(snap):,} snapshot rows")
    return snap, help_df


# ── gate machinery (design §8.1, spec correction #4b) ────────────────────────

_GATE_RULE_TEXT = (
    "kept iff >=1 metric improves with CI95 lower bound > 0 AND no metric "
    "significantly regresses (CI95 upper bound < 0) — cal only. A saturated "
    "metric (delta exactly 0, e.g. purity@50 already 1.0) cannot block a "
    "CI-significant improvement on the other metric."
)


def _gate_keep_rule(
    auc: tuple[float | None, float | None, float | None],
    pur: tuple[float | None, float | None, float | None],
) -> bool:
    """Paired-bootstrap gate: significant improvement somewhere, significant
    regression nowhere.  See _GATE_RULE_TEXT."""
    improves = any(lo is not None and lo > 0 for _, lo, _ in (auc, pur))
    regresses = any(hi is not None and hi < 0 for _, _, hi in (auc, pur))
    return improves and not regresses

def _macro_ovr_auc(y_idx: np.ndarray, proba: np.ndarray) -> float | None:
    """Macro OvR AUC over the classes computable in this sample."""
    from sklearn.metrics import roc_auc_score

    aucs: list[float] = []
    for ci in range(proba.shape[1]):
        y = (y_idx == ci).astype(int)
        p = proba[:, ci]
        mask = np.isfinite(p)
        if mask.sum() < 10 or len(np.unique(y[mask])) < 2:
            continue
        try:
            aucs.append(float(roc_auc_score(y[mask], p[mask])))
        except Exception:
            continue
    return float(np.mean(aucs)) if aucs else None


def _purity_at_k(y_is_target: np.ndarray, score: np.ndarray, k: int = 50) -> float | None:
    mask = np.isfinite(score)
    if mask.sum() < 10:
        return None
    order = np.argsort(-score[mask], kind="stable")
    top = order[: min(k, mask.sum())]
    return float(np.asarray(y_is_target, dtype=float)[mask][top].mean())


def _paired_delta_ci(
    fn: Callable[[np.ndarray], float | None],
    n_rows: int,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float | None, float | None, float | None]:
    """Paired object bootstrap of a delta metric.

    ``fn(idx)`` evaluates metric_with(idx) − metric_without(idx) on resampled
    row indices (rows here are object-level: one row per object per slice, so
    a row bootstrap IS an object bootstrap).
    """
    point = fn(np.arange(n_rows))
    if point is None:
        return None, None, None
    deltas: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_rows, size=n_rows)
        d = fn(idx)
        if d is not None:
            deltas.append(d)
    if len(deltas) < max(20, n_boot // 4):
        return point, None, None
    return point, float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def _gate_eval_frames(
    snapshots: pd.DataFrame, cal_ids: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cal-only evaluation frames (object-level, spec-only):

      auc frame   — one row per cal object at exactly n_det == 5
      purity frame — latest-epoch row per cal object (for purity@50, u_Ia)
    """
    cal = snapshots[
        snapshots["object_id"].astype(str).isin(cal_ids)
        & snapshots["target_class"].notna()
        & snapshots.get("label_quality", pd.Series(index=snapshots.index, dtype=object))
        .astype(str)
        .isin(SPEC_QUALITIES)
    ].copy()
    auc_frame = cal[cal["n_det"] == 5].copy()
    purity_frame = (
        cal.sort_values(["object_id", "n_det"]).groupby("object_id", as_index=False).tail(1)
    )
    return auc_frame.reset_index(drop=True), purity_frame.reset_index(drop=True)


def _predict_variant(artifact: Any, df: pd.DataFrame) -> np.ndarray:
    """(N,3) calibrated probabilities from a MulticlassFollowupArtifact."""
    return np.asarray(artifact.predict_proba(df), dtype=float)


def run_component_gates(
    snapshots: pd.DataFrame,
    train_ids: set[str],
    cal_ids: set[str],
    test_ids: set[str],
    *,
    gate_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[str], float, list[dict]]:
    """Forward-additive component gates on cal (design §8.1).

    Returns (columns_to_drop_for_final_model, final_aug_frac, gate_ledger).
    """
    from debass_meta.models.multiclass_followup import (
        MulticlassFollowupArtifact,
        train_multiclass_followup,
    )

    rng = np.random.default_rng(args.seed)
    ledger: list[dict] = []

    # Component → column sets ------------------------------------------------
    try:
        from debass_meta.features.lightcurve_ext import EXT_FEATURE_NAMES

        ext_cols = [c for c in snapshots.columns if c in set(EXT_FEATURE_NAMES)]
    except Exception as exc:  # G1 module missing → component skipped
        ext_cols = []
        ledger.append({
            "component": "ext_features", "decision": "skipped",
            "reason": f"lightcurve_ext not importable ({exc})",
        })
    else:
        if not ext_cols:
            ledger.append({"component": "ext_features", "decision": "skipped",
                           "reason": "no EXT_FEATURE_NAMES columns in snapshot"})
    traj_cols = [c for c in snapshots.columns if c.startswith(("traj__", "traj_x__"))]
    q_cols = [c for c in snapshots.columns if c.startswith(("q__", "q_prior__"))]

    components: list[tuple[str, list[str]]] = []
    if ext_cols:
        components.append(("ext_features", ext_cols))
    if traj_cols:
        components.append(("traj_features", traj_cols))
    else:
        ledger.append({"component": "traj_features", "decision": "skipped",
                       "reason": "no traj__ columns in snapshot"})
    if q_cols:
        components.append(("pooled_trust_q", q_cols))
    else:
        ledger.append({"component": "pooled_trust_q", "decision": "skipped",
                       "reason": "no q__ columns in snapshot"})

    auc_frame, purity_frame = _gate_eval_frames(snapshots, cal_ids)
    if len(auc_frame) < 30 or len(purity_frame) < 60:
        ledger.append({
            "component": "_all", "decision": "kept",
            "reason": (f"cal gate frames too small (auc n={len(auc_frame)}, "
                       f"purity n={len(purity_frame)}) — all components kept by default"),
        })
        return [], args.aug_frac, ledger

    y_auc = class_index(auc_frame["target_class"])
    y_pur = (purity_frame["target_class"].astype(str) == CLASSES[0]).to_numpy()

    def _fit_and_eval(name: str, drop_cols: list[str], aug_frac: float) -> dict:
        out_dir = gate_dir / name
        sub = snapshots.drop(columns=[c for c in drop_cols if c in snapshots.columns])
        train_multiclass_followup(
            sub, train_ids, cal_ids, test_ids, str(out_dir),
            weak_weight=args.weak_weight, context_weight=args.context_weight,
            tns_untyped_weight=args.tns_untyped_weight, bts_weight=args.bts_weight,
            aug_frac=aug_frac, aug_weight=args.aug_weight,
            grid_small=True, n_jobs=args.n_jobs, seed=args.seed,
        )
        artifact = MulticlassFollowupArtifact.load(str(out_dir))
        p_auc = _predict_variant(artifact, auc_frame.drop(columns=[c for c in drop_cols if c in auc_frame.columns]))
        p_pur = _predict_variant(artifact, purity_frame.drop(columns=[c for c in drop_cols if c in purity_frame.columns]))
        return {
            "name": name,
            "proba_auc": p_auc,
            "p_snia_purity": p_pur[:, 0],
            "macro_auc": _macro_ovr_auc(y_auc, p_auc),
            "purity50": _purity_at_k(y_pur, p_pur[:, 0], k=50),
        }

    all_gateable = [c for _, cols in components for c in cols]
    current_drop = list(all_gateable)
    current = _fit_and_eval("base", current_drop, aug_frac=0.0)
    print(f"  gate base: macro_auc@5={current['macro_auc']}, purity@50={current['purity50']}")

    for name, cols in components:
        candidate_drop = [c for c in current_drop if c not in set(cols)]
        cand = _fit_and_eval(f"plus_{name}", candidate_drop, aug_frac=0.0)

        def _delta_auc(idx: np.ndarray) -> float | None:
            a = _macro_ovr_auc(y_auc[idx], cand["proba_auc"][idx])
            b = _macro_ovr_auc(y_auc[idx], current["proba_auc"][idx])
            return None if a is None or b is None else a - b

        def _delta_pur(idx: np.ndarray) -> float | None:
            a = _purity_at_k(y_pur[idx], cand["p_snia_purity"][idx], 50)
            b = _purity_at_k(y_pur[idx], current["p_snia_purity"][idx], 50)
            return None if a is None or b is None else a - b

        d_auc, auc_lo, auc_hi = _paired_delta_ci(_delta_auc, len(auc_frame), args.gate_boot, rng)
        d_pur, pur_lo, pur_hi = _paired_delta_ci(_delta_pur, len(purity_frame), args.gate_boot, rng)

        keep = _gate_keep_rule((d_auc, auc_lo, auc_hi), (d_pur, pur_lo, pur_hi))
        entry = {
            "component": name,
            "decision": "kept" if keep else "dropped",
            "cal_macro_auc_with": cand["macro_auc"],
            "cal_macro_auc_without": current["macro_auc"],
            "delta_macro_auc": d_auc, "delta_macro_auc_ci95": [auc_lo, auc_hi],
            "cal_purity50_with": cand["purity50"],
            "cal_purity50_without": current["purity50"],
            "delta_purity50": d_pur, "delta_purity50_ci95": [pur_lo, pur_hi],
            "rule": _GATE_RULE_TEXT,
        }
        ledger.append(entry)
        print(f"  gate {name}: {entry['decision']} (d_auc={d_auc}, d_pur={d_pur})")
        if keep:
            current = cand
            current_drop = candidate_drop

    # Expert-dropout augmentation gate (spec correction #4a) -------------------
    cand = _fit_and_eval("plus_aug", current_drop, aug_frac=args.aug_frac)

    def _delta_auc_aug(idx: np.ndarray) -> float | None:
        a = _macro_ovr_auc(y_auc[idx], cand["proba_auc"][idx])
        b = _macro_ovr_auc(y_auc[idx], current["proba_auc"][idx])
        return None if a is None or b is None else a - b

    def _delta_pur_aug(idx: np.ndarray) -> float | None:
        a = _purity_at_k(y_pur[idx], cand["p_snia_purity"][idx], 50)
        b = _purity_at_k(y_pur[idx], current["p_snia_purity"][idx], 50)
        return None if a is None or b is None else a - b

    d_auc, auc_lo, auc_hi = _paired_delta_ci(_delta_auc_aug, len(auc_frame), args.gate_boot, rng)
    d_pur, pur_lo, pur_hi = _paired_delta_ci(_delta_pur_aug, len(purity_frame), args.gate_boot, rng)
    keep_aug = _gate_keep_rule((d_auc, auc_lo, auc_hi), (d_pur, pur_lo, pur_hi))
    ledger.append({
        "component": "expert_dropout_aug",
        "decision": "kept" if keep_aug else "dropped",
        "cal_macro_auc_with": cand["macro_auc"], "cal_macro_auc_without": current["macro_auc"],
        "delta_macro_auc": d_auc, "delta_macro_auc_ci95": [auc_lo, auc_hi],
        "cal_purity50_with": cand["purity50"], "cal_purity50_without": current["purity50"],
        "delta_purity50": d_pur, "delta_purity50_ci95": [pur_lo, pur_hi],
        "rule": _GATE_RULE_TEXT,
    })
    final_aug = args.aug_frac if keep_aug else 0.0
    print(f"  gate expert_dropout_aug: {'kept' if keep_aug else 'dropped'}")
    return current_drop, final_aug, ledger


# ── main orchestration ───────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train fusion_v8 (Stage A → Stage B → conformal)")
    parser.add_argument("--snapshots", default="data/gold/object_epoch_snapshots_fusion_v8.parquet")
    parser.add_argument("--helpfulness", default="data/gold/expert_helpfulness_fusion_v8.parquet")
    parser.add_argument("--split", default="data/gold/split_fusion_v8.json")
    parser.add_argument("--trust-dir", default="models/trust_fusion_v8")
    parser.add_argument("--followup-dir", default="models/followup_fusion_v8")
    parser.add_argument("--conformal-dir", default="models/conformal_fusion_v8")
    parser.add_argument("--output-snapshots",
                        default="data/gold/object_epoch_snapshots_fusion_v8_trust.parquet",
                        help="Trust-augmented snapshot parquet (input to score/eval)")
    parser.add_argument("--metrics-out", default="reports/metrics/fusion_v8_train.json")
    parser.add_argument("--n-jobs", type=int, default=8,
                        help="LightGBM threads (match -pe omp slots)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true",
                        help="<=200-object run: grid_small, gates+seed-spread skipped by default")
    parser.add_argument("--skip-stage-a", action="store_true",
                        help="Reuse q/q_prior columns from an existing --output-snapshots parquet")
    parser.add_argument("--skip-gates", action="store_true",
                        help="Skip component gates; include every component")
    parser.add_argument("--run-gates", action="store_true",
                        help="Force gates on even with --smoke")
    parser.add_argument("--skip-seed-spread", action="store_true",
                        help="Skip the 5-seed bagging-spread variants (guard #4b)")
    parser.add_argument("--gate-boot", type=int, default=500,
                        help="Bootstrap resamples for gate delta CIs (cal-only)")
    parser.add_argument("--alpha", type=float, default=0.1, help="Conformal miscoverage level")
    parser.add_argument("--aug-frac", type=float, default=0.25)
    parser.add_argument("--aug-weight", type=float, default=0.3)
    parser.add_argument("--weak-weight", type=float, default=0.1)
    parser.add_argument("--context-weight", type=float, default=0.15)
    parser.add_argument("--tns-untyped-weight", type=float, default=0.6)
    parser.add_argument("--bts-weight", type=float, default=1.0)
    args = parser.parse_args()

    t0 = time.time()
    report: dict[str, Any] = {
        "pipeline": "fusion_v8",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": {k: str(v) for k, v in vars(args).items()},
    }

    # ── split (locked; asserted disjoint) ───────────────────────────────────
    train_ids, cal_ids, test_ids, _split_raw = load_split_manifest(Path(args.split))
    print(f"Split: train={len(train_ids):,} cal={len(cal_ids):,} test={len(test_ids):,}")
    report["split"] = {"source": args.split, "n_train": len(train_ids),
                       "n_cal": len(cal_ids), "n_test": len(test_ids)}

    # ── Stage A: pooled trust ────────────────────────────────────────────────
    output_snapshots = Path(args.output_snapshots)
    if args.skip_stage_a:
        if not output_snapshots.exists():
            raise SystemExit(f"--skip-stage-a but {output_snapshots} does not exist")
        snap_trust = pd.read_parquet(output_snapshots)
        print(f"Loaded {len(snap_trust):,} trust-snapshot rows "
              f"({snap_trust['object_id'].nunique():,} objects) from {output_snapshots}")
        if args.smoke:
            snap_trust, _ = _smoke_subset(snap_trust, None, train_ids, cal_ids, test_ids,
                                          seed=args.seed)
        stage_a_metrics: dict = {}
        prior = Path(args.metrics_out)
        if prior.exists():
            try:
                stage_a_metrics = json.load(open(prior)).get("stage_a", {}).get("metrics", {})
            except Exception:
                pass
        report["stage_a"] = {"skipped": True, "reused_snapshot": str(output_snapshots),
                             "metrics": stage_a_metrics}
        print(f"Stage A skipped — reusing q columns from {output_snapshots}")
    else:
        import inspect

        from debass_meta.models.pooled_trust import train_pooled_trust

        snapshots = pd.read_parquet(args.snapshots)
        print(f"Loaded {len(snapshots):,} snapshot rows "
              f"({snapshots['object_id'].nunique():,} objects) from {args.snapshots}")
        helpfulness = pd.read_parquet(args.helpfulness)
        print(f"Loaded {len(helpfulness):,} helpfulness rows from {args.helpfulness}")
        leak = [c for c in helpfulness.columns if c.startswith(("q__", "trust_source__"))]
        assert not leak, f"Stage-A leakage: q__/trust_source__ columns in helpfulness: {leak}"
        if args.smoke:
            snapshots, helpfulness = _smoke_subset(
                snapshots, helpfulness, train_ids, cal_ids, test_ids, seed=args.seed)

        print("Stage A: pooled trust (GroupKFold OOF on train; refit for cal/test)…")
        kwargs: dict[str, Any] = {"n_jobs": args.n_jobs, "seed": args.seed}
        try:  # extra kwarg beyond the pinned signature — pass only if supported
            if "grid_small" in inspect.signature(train_pooled_trust).parameters:
                kwargs["grid_small"] = bool(args.smoke)
        except (TypeError, ValueError):
            pass
        result = train_pooled_trust(
            helpfulness, snapshots, train_ids, cal_ids, test_ids,
            str(args.trust_dir), **kwargs,
        )
        snap_trust = result.snapshots
        output_snapshots.parent.mkdir(parents=True, exist_ok=True)
        snap_trust.to_parquet(output_snapshots, index=False)
        print(f"Wrote trust-augmented snapshot → {output_snapshots} ({len(snap_trust):,} rows)")
        report["stage_a"] = {"skipped": False, "artifact_dir": result.artifact_dir,
                             "metrics": result.metrics}

    n_q = sum(1 for c in snap_trust.columns if c.startswith("q__"))
    print(f"  snapshot has {n_q} q__ columns")

    # ── component gates (cal-only decisions) ─────────────────────────────────
    followup_dir = Path(args.followup_dir)
    run_gates = args.run_gates or (not args.skip_gates and not args.smoke)
    if run_gates:
        print("Component gates (decided on cal, design §8.1)…")
        drop_cols, final_aug, ledger = run_component_gates(
            snap_trust, train_ids, cal_ids, test_ids,
            gate_dir=followup_dir / "_gates", args=args,
        )
    else:
        drop_cols, final_aug = [], args.aug_frac
        ledger = [{"component": "_all", "decision": "kept",
                   "reason": "gates skipped (--skip-gates or --smoke); all components included"}]
    report["gates"] = ledger
    report["gated_dropped_columns"] = drop_cols
    report["aug_frac_final"] = final_aug

    # ── Stage B: multiclass follow-up ────────────────────────────────────────
    from debass_meta.models.multiclass_followup import (
        MulticlassFollowupArtifact,
        train_multiclass_followup,
    )

    snap_for_b = snap_trust.drop(columns=[c for c in drop_cols if c in snap_trust.columns])
    print(f"Stage B: multiclass follow-up ({len(snap_for_b.columns)} cols after gating, "
          f"aug_frac={final_aug})…")
    stage_b_metrics = train_multiclass_followup(
        snap_for_b, train_ids, cal_ids, test_ids, str(followup_dir),
        weak_weight=args.weak_weight, context_weight=args.context_weight,
        tns_untyped_weight=args.tns_untyped_weight, bts_weight=args.bts_weight,
        aug_frac=final_aug, aug_weight=args.aug_weight,
        grid_small=args.smoke, n_jobs=args.n_jobs, seed=args.seed,
    )
    report["stage_b"] = {"model_dir": str(followup_dir), "metrics": stage_b_metrics}

    artifact = MulticlassFollowupArtifact.load(str(followup_dir))

    # ── Dirichlet calibration gate entry (informational; raw vs calibrated on cal)
    try:
        auc_frame, _ = _gate_eval_frames(snap_for_b, cal_ids)
        if len(auc_frame) >= 30:
            from sklearn.metrics import log_loss

            y5 = class_index(auc_frame["target_class"])
            p_raw = np.asarray(artifact.predict_proba_raw(auc_frame), dtype=float)
            p_cal = np.asarray(artifact.predict_proba(auc_frame), dtype=float)
            eps = 1e-6

            def _ll(p: np.ndarray) -> float:
                q = np.clip(p, eps, 1.0)
                q = q / q.sum(axis=1, keepdims=True)
                return float(log_loss(y5, q, labels=list(range(len(CLASSES)))))

            report["gates"].append({
                "component": "dirichlet_calibration",
                "decision": "kept",
                "reason": "fallback ladder internal to MulticlassFollowupArtifact",
                "cal_macro_auc_calibrated": _macro_ovr_auc(y5, p_cal),
                "cal_macro_auc_raw": _macro_ovr_auc(y5, p_raw),
                "cal_logloss_calibrated": _ll(p_cal),
                "cal_logloss_raw": _ll(p_raw),
            })
    except Exception as exc:
        report["gates"].append({"component": "dirichlet_calibration",
                                "decision": "kept", "reason": f"cal check failed: {exc}"})

    # ── seed-spread variants (guard #4b; scored on test ONLY by eval) ───────
    seed_variants: dict[str, Any] | None = None
    if not args.skip_seed_spread and not args.smoke:
        seeds = [args.seed + k for k in range(1, 5)]
        dirs: list[str] = []
        print(f"Seed-spread variants (grid_small) for seeds {seeds}…")
        for s in seeds:
            vdir = followup_dir / "_seed_variants" / f"seed{s}"
            train_multiclass_followup(
                snap_for_b, train_ids, cal_ids, test_ids, str(vdir),
                weak_weight=args.weak_weight, context_weight=args.context_weight,
                tns_untyped_weight=args.tns_untyped_weight, bts_weight=args.bts_weight,
                aug_frac=final_aug, aug_weight=args.aug_weight,
                grid_small=True, n_jobs=args.n_jobs, seed=s,
            )
            dirs.append(str(vdir))
        seed_variants = {"seeds": [args.seed] + seeds,
                         "main_dir": str(followup_dir), "variant_dirs": dirs,
                         "dropped_columns": drop_cols}
    report["seed_variants"] = seed_variants

    # ── conformal: Mondrian split-conformal APS on cal ───────────────────────
    from debass_meta.models.conformal import MondrianAPS

    cal_rows = snap_trust[
        snap_trust["object_id"].astype(str).isin(cal_ids)
        & snap_trust["target_class"].notna()
        & snap_trust["label_quality"].astype(str).isin(CAL_QUALITIES)
    ].copy()
    # One row per (object, bucket) at the bucket's max n_det — keeps the cal
    # nonconformity sample object-level within each Mondrian bucket.
    cal_rows["n_det_bucket"] = ndet_bucket(cal_rows["n_det"].to_numpy())
    cal_rows = (
        cal_rows.sort_values(["object_id", "n_det"])
        .groupby(["object_id", "n_det_bucket"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    proba_cal = np.asarray(artifact.predict_proba(cal_rows), dtype=float)
    y_cal = class_index(cal_rows["target_class"])
    strata_cal = pd.DataFrame({
        "survey": cal_rows["survey"].astype(str).str.lower().to_numpy(),
        "n_det": cal_rows["n_det"].astype(int).to_numpy(),
        "n_det_bucket": cal_rows["n_det_bucket"].to_numpy(),
    })
    conformal = MondrianAPS()
    conformal.fit(proba_cal, y_cal, strata_cal, alpha=args.alpha)
    conformal_dir = Path(args.conformal_dir)
    conformal_dir.mkdir(parents=True, exist_ok=True)
    conformal_path = conformal_dir / "mondrian_aps.pkl"
    conformal.save(str(conformal_path))
    bucket_counts = (
        strata_cal.assign(cls=[CLASSES[i] if 0 <= i < len(CLASSES) else "?" for i in y_cal])
        .groupby(["cls", "survey", "n_det_bucket"]).size()
    )
    report["conformal"] = {
        "path": str(conformal_path), "alpha": args.alpha,
        "n_cal_rows": int(len(cal_rows)),
        "bucket_counts": {" / ".join(map(str, k)): int(v) for k, v in bucket_counts.items()},
    }
    print(f"Wrote conformal artifact → {conformal_path} ({len(cal_rows):,} cal rows)")

    # ── persist train report ─────────────────────────────────────────────────
    report["paths"] = {
        "snapshots": args.snapshots, "helpfulness": args.helpfulness,
        "snapshot_trust": str(output_snapshots), "trust_dir": str(args.trust_dir),
        "followup_dir": str(followup_dir), "conformal": str(conformal_path),
    }
    report["wall_clock_s"] = round(time.time() - t0, 1)
    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w") as fh:
        json.dump(_jsonify(report), fh, indent=2)
    print(f"Wrote train report → {metrics_out} ({report['wall_clock_s']} s)")


if __name__ == "__main__":
    main()
