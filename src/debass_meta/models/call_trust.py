"""Call trust: one meaning of trust for every expert (fusion v13g, opt-in).

The Stage-A trust heads of the SN-filter experts (``expert_trust.SN_FILTER_EXPERTS``,
target ``is_sn``) estimate P(object is a supernova); the others (target
``is_topclass_correct``) estimate P(the expert's top class is correct).  So
``q__<expert>`` does not mean the same thing across experts.  The follow-up
product is a calibrated P(SN); trust is a diagnostic, and this module gives it
ONE definition for every expert::

    call_trust = P(this expert's SN-vs-not call is correct)
               = q_sn          if the expert's projected p_sn >= 0.5 (it calls SN)
                 1 - q_sn      otherwise
    p_sn       = p_snia + p_nonIa_snlike  (the expert's projection)

``q_sn`` is a pooled ``is_sn`` head fitted next to the main Stage-A model
(``train_pooled_trust(call_trust=True)`` / ``--stage-a-call-trust``) on the same
rows, the same held-out folds and the same features, with the same hierarchical
per-expert calibration (no dedicated-head fallback).  Experts already in the
SN-filter set reuse their existing head (their ``q__`` IS ``q_sn``).  Nothing here
touches head-1 / head-2 inputs: the ``q_sn__`` / ``call_trust__`` / ``sn_call__``
columns are blocked from the Stage-B feature discovery
(``multiclass_followup._BLOCKED_PREFIXES``) and the existing ``q__`` / ``q_prior__``
columns and every P output are unchanged.

Artifact: ``<trust_dir>/pooled/call_trust/`` (``pooled/`` is skipped by every
trust-dir walker): ``model.pkl``, ``global_calibrator.pkl``, ``calibrators/<san>.pkl``,
``metadata.json``.  The features, expert levels, per-expert row counts and exactness
codes are the main pooled model's (``pooled/metadata.json``), so scoring rebuilds
the training features exactly (``PooledTrustView`` with the call-trust bundle).

Train rows carry OUT-OF-FOLD ``q_sn`` (every train object is scored by the fold
model that held it out, also for rows the head was not fitted on); cal/test rows
the refit-on-train model, exactly as the scorer computes them.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from debass_meta.projectors import sanitize_expert_key

from .calibrate import IsotonicCalibrator
from .expert_trust import _binary_metrics, _predict_binary_classifier
from .pooled_trust import (
    POOLED_SUBDIR,
    PooledTrustView,
    _fit_expert_calibrator,
    _fit_pooled_classifier,
    _oof_pooled_fits,
    _prepare_pooled_matrix,
    stage_a_weights,
)

CALL_TRUST_SUBDIR = "call_trust"
CALL_TRUST_KIND = "call_trust_is_sn_fusion_v13g"
SN_CALL_THRESHOLD = 0.5
# Columns the scorer / training snapshot gain; blocked from Stage-B features.
CALL_TRUST_PREFIXES = ("q_sn__", "call_trust__", "sn_call__")


@dataclass
class CallTrustFit:
    """Fitted call-trust heads (in memory, from :func:`fit_call_trust`)."""

    bundle: Any
    fold_bundles: list[Any]
    fold_of_object: dict[str, int]
    global_calibrator: Any
    experts: dict[str, dict[str, Any]]            # expert_key -> calibrator, calibrator_kind, metrics
    reused_sn_filter_experts: list[str]
    skipped: dict[str, str]
    params: dict[str, Any]
    n_estimators: int
    summary: dict[str, Any] = field(default_factory=dict)


def fit_call_trust(
    long_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    expert_levels: list[str],
    params: dict[str, Any],
    n_estimators: int,
    main_fold_of_object: dict[str, int],
    head_experts,
    sn_filter,
    seed: int,
    n_jobs: int,
) -> CallTrustFit:
    """Fit the pooled ``is_sn`` head for the non-SN-filter experts with a trust head.

    ``long_df`` is the main Stage-A long table (``assemble_stage_a_long(...,
    with_sn_target=True)`` after the ``y`` filter, with ``_split`` and
    ``log_expert_train_rows``): the same rows as the main model, target ``y_sn``.
    ``main_fold_of_object`` is the main model's held-out fold per train object
    (the same folds); ``head_experts`` the experts that have a trust head.
    """
    head_experts = {str(k) for k in head_experts}
    sn_filter = {str(k) for k in sn_filter}
    reused = sorted(head_experts & sn_filter)
    want = sorted(head_experts - sn_filter)
    skipped: dict[str, str] = {}
    keep = np.zeros(len(long_df), dtype=bool)
    experts_col = long_df["expert_key"].to_numpy()
    y_sn_all = pd.to_numeric(long_df["y_sn"], errors="coerce").to_numpy(dtype=float)
    for key in want:
        rows = (experts_col == key) & np.isfinite(y_sn_all)
        if not rows.any():
            skipped[key] = "no is_sn labels on its Stage-A rows"
            continue
        keep |= rows
    sub = long_df[keep].reset_index(drop=True)
    y = sub["y_sn"].to_numpy(dtype=int)
    split = sub["_split"].to_numpy()
    tr, ca, te = split == "train", split == "cal", split == "test"
    if int(tr.sum()) == 0:
        raise ValueError("fit_call_trust: no training rows with an is_sn label")
    X = _prepare_pooled_matrix(sub, feature_cols, expert_levels)
    w_tr = stage_a_weights(sub[tr])
    groups = sub.loc[tr, "object_id"].astype(str).to_numpy()
    oof, fold_bundles, fold_of_object = _oof_pooled_fits(
        X[tr], y[tr], w_tr, groups, params=params, n_estimators=n_estimators,
        seed=seed, n_jobs=n_jobs, fixed_fold_of_group=main_fold_of_object or None)
    bundle = _fit_pooled_classifier(
        X[tr], y[tr], w_tr, params=params, n_estimators=n_estimators, seed=seed, n_jobs=n_jobs)
    q_raw = np.asarray(_predict_binary_classifier(bundle, X), dtype=float)

    global_cal = None
    q_cal, y_cal = q_raw[ca], y[ca].astype(float)
    if len(y_cal) >= 10 and len(np.unique(y_cal)) == 2:
        global_cal = IsotonicCalibrator().fit(q_cal, y_cal.astype(int))

    per_expert: dict[str, dict[str, Any]] = {}
    sub_experts = sub["expert_key"].to_numpy()
    for key in want:
        if key in skipped:
            continue
        e = sub_experts == key
        calibrator, kind, notes = _fit_expert_calibrator(
            q_raw[e & ca], y[e & ca].astype(float), q_raw[e & te], y[e & te].astype(float), global_cal)
        for note in notes:
            print(f"    call_trust {key}: {note}")
        raw_test = _binary_metrics(y[e & te].astype(float), q_raw[e & te])
        cal_vals = (np.asarray(calibrator.transform(q_raw[e & te]), dtype=float)
                    if calibrator is not None and (e & te).any() else q_raw[e & te])
        cal_test = _binary_metrics(y[e & te].astype(float), cal_vals)
        cal_set = _binary_metrics(y[e & ca].astype(float), q_raw[e & ca])
        per_expert[key] = {
            "calibrator": calibrator,
            "calibrator_kind": kind,
            "metrics": {
                "raw_auc": raw_test.get("roc_auc"), "cal_auc": cal_test.get("roc_auc"),
                "brier": cal_test.get("brier"), "ece": cal_test.get("ece"),
                "cal_set_raw_auc": cal_set.get("roc_auc"),
                "n_train_rows": int((e & tr).sum()), "n_cal_rows": int((e & ca).sum()),
                "n_test": int((e & te).sum()), "positive_rate_train": (
                    float(y[e & tr].mean()) if (e & tr).any() else None),
            },
        }
    summary = {
        "n_train_rows": int(tr.sum()), "n_cal_rows": int(ca.sum()), "n_test_rows": int(te.sum()),
        "n_folds": len(fold_bundles), "same_folds_as_pooled": bool(main_fold_of_object),
        "has_global_calibrator": global_cal is not None,
        "experts_fitted": sorted(per_expert), "reused_sn_filter_experts": reused,
        "skipped": dict(skipped),
        "per_expert": {k: v["metrics"] | {"calibrator_kind": v["calibrator_kind"]}
                       for k, v in per_expert.items()},
    }
    return CallTrustFit(
        bundle=bundle, fold_bundles=fold_bundles, fold_of_object=dict(fold_of_object),
        global_calibrator=global_cal, experts=per_expert, reused_sn_filter_experts=reused,
        skipped=skipped, params=dict(params), n_estimators=int(n_estimators), summary=summary)


def save_call_trust(fit: CallTrustFit, pooled_dir: Path, *, seed: int) -> Path:
    """Write ``<pooled_dir>/call_trust/`` (bundle, calibrators, metadata)."""
    out = Path(pooled_dir) / CALL_TRUST_SUBDIR
    (out / "calibrators").mkdir(parents=True, exist_ok=True)
    with open(out / "model.pkl", "wb") as fh:
        pickle.dump(fit.bundle, fh)
    if fit.global_calibrator is not None:
        with open(out / "global_calibrator.pkl", "wb") as fh:
            pickle.dump(fit.global_calibrator, fh)
    experts: dict[str, Any] = {}
    for key, state in fit.experts.items():
        san = sanitize_expert_key(key)
        if state["calibrator_kind"] in {"isotonic", "platt"} and state["calibrator"] is not None:
            with open(out / "calibrators" / f"{san}.pkl", "wb") as fh:
                pickle.dump(state["calibrator"], fh)
        experts[key] = {"calibrator_kind": state["calibrator_kind"], "metrics": state["metrics"]}
    meta = {
        "kind": CALL_TRUST_KIND,
        "target": "is_sn",
        "definition": "call_trust = q_sn if p_sn >= 0.5 else 1 - q_sn; p_sn = p_snia + p_nonIa_snlike",
        "sn_call_threshold": SN_CALL_THRESHOLD,
        "experts": experts,
        "reused_sn_filter_experts": list(fit.reused_sn_filter_experts),
        "skipped": dict(fit.skipped),
        "params": fit.params,
        "n_estimators": int(fit.n_estimators),
        "seed": int(seed),
        "n_folds": len(fit.fold_bundles),
        "same_folds_as_pooled": bool(fit.summary.get("same_folds_as_pooled")),
        "has_global_calibrator": fit.global_calibrator is not None,
        "features": "pooled metadata (pooled/metadata.json)",
    }
    with open(out / "metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    return out


def load_call_trust_assets(trust_dir: str | Path) -> dict[str, Any] | None:
    """The saved call-trust heads of a trust dir, or ``None`` when the artifact
    was trained without them (every v8-v13f artifact)."""
    ct_dir = Path(trust_dir) / POOLED_SUBDIR / CALL_TRUST_SUBDIR
    if not (ct_dir / "metadata.json").exists():
        return None
    with open(ct_dir / "metadata.json") as fh:
        meta = json.load(fh)
    with open(ct_dir / "model.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    global_cal = None
    if (ct_dir / "global_calibrator.pkl").exists():
        with open(ct_dir / "global_calibrator.pkl", "rb") as fh:
            global_cal = pickle.load(fh)
    cals: dict[str, Any] = {}
    for key, entry in meta.get("experts", {}).items():
        san = sanitize_expert_key(key)
        path = ct_dir / "calibrators" / f"{san}.pkl"
        if entry.get("calibrator_kind") in {"isotonic", "platt"} and path.exists():
            with open(path, "rb") as fh:
                cals[key] = pickle.load(fh)
        elif entry.get("calibrator_kind") == "global":
            cals[key] = global_cal
    return {"metadata": meta, "bundle": bundle, "global_calibrator": global_cal, "calibrators": cals}


def call_trust_view(expert_key: str, assets: dict[str, Any], pooled_metadata: dict[str, Any]) -> PooledTrustView:
    """``PooledTrustView`` of the ``is_sn`` head of ``expert_key``: ``predict_trust``
    is ``q_sn`` (same features as the main pooled model)."""
    kind = assets["metadata"]["experts"][expert_key]["calibrator_kind"]
    return PooledTrustView(
        expert_key=expert_key, calibrator=assets["calibrators"].get(expert_key),
        calibrator_kind=kind, fallback_used=False, pooled_bundle=assets["bundle"],
        pooled_metadata=pooled_metadata, global_calibrator=assets["global_calibrator"], has_head=True)


def _avail(df: pd.DataFrame, san: str) -> np.ndarray:
    col = f"avail__{san}"
    if col not in df.columns:
        return np.zeros(len(df), dtype=bool)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0


def emit_q_sn_columns(
    snapshots: pd.DataFrame,
    fit: CallTrustFit,
    pooled_metadata: dict[str, Any],
    train_ids: set[str],
) -> pd.DataFrame:
    """``q_sn__<san>`` on the training snapshot: out-of-fold on train objects,
    refit-on-train (== the scorer) elsewhere, NaN where the expert is absent.
    SN-filter experts reuse their ``q__`` (already P(SN)).  Needs the main
    emission's ``q__`` columns in ``snapshots``."""
    assets = {
        "metadata": {"experts": {k: {"calibrator_kind": v["calibrator_kind"]} for k, v in fit.experts.items()}},
        "bundle": fit.bundle, "global_calibrator": fit.global_calibrator,
        "calibrators": {k: v["calibrator"] for k, v in fit.experts.items()},
    }
    snap_train = snapshots["object_id"].astype(str).isin(train_ids).to_numpy()
    fold_index = snapshots["object_id"].astype(str).map(fit.fold_of_object).to_numpy(dtype=float)
    new_cols: dict[str, Any] = {}
    for key in fit.experts:
        san = sanitize_expert_key(key)
        view = call_trust_view(key, assets, pooled_metadata)
        X = view._pooled_features(snapshots, prior_mode=False)
        raw = np.asarray(_predict_binary_classifier(fit.bundle, X), dtype=float).copy()
        for fold, fold_bundle in enumerate(fit.fold_bundles):
            m = (fold_index == fold) & snap_train
            if m.any():
                raw[m] = _predict_binary_classifier(fold_bundle, X[m])
        q = (np.asarray(view.calibrator.transform(raw), dtype=float)
             if view.calibrator is not None else raw).copy()
        q[~_avail(snapshots, san)] = np.nan
        new_cols[f"q_sn__{san}"] = q
    for key in fit.reused_sn_filter_experts:
        san = sanitize_expert_key(key)
        if f"q__{san}" in snapshots.columns:
            new_cols[f"q_sn__{san}"] = pd.to_numeric(snapshots[f"q__{san}"], errors="coerce").to_numpy(float)
    collision = [c for c in new_cols if c in snapshots.columns]
    base = snapshots.drop(columns=collision) if collision else snapshots
    return pd.concat([base.copy(), pd.DataFrame(new_cols, index=snapshots.index)], axis=1)


def expert_p_sn(df: pd.DataFrame, san: str) -> np.ndarray:
    """The expert's projected ``p_snia + p_nonIa_snlike`` (NaN when it has neither)."""
    def col(name: str) -> pd.Series:
        c = f"proj__{san}__{name}"
        return pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index)

    a, b = col("p_snia"), col("p_nonIa_snlike")
    return np.where(a.isna() & b.isna(), np.nan, a.fillna(0.0) + b.fillna(0.0))


def call_trust_from_q_sn(q_sn: np.ndarray, p_sn: np.ndarray, available: np.ndarray):
    """``(sn_call, call_trust)``: 1/0 call and the probability that it is
    correct; NaN where the expert is unavailable or has no q_sn / p_sn."""
    q_sn = np.asarray(q_sn, dtype=float)
    p_sn = np.asarray(p_sn, dtype=float)
    ok = np.asarray(available, dtype=bool) & np.isfinite(q_sn) & np.isfinite(p_sn)
    call = np.where(p_sn >= SN_CALL_THRESHOLD, 1.0, 0.0)
    trust = np.where(call == 1.0, q_sn, 1.0 - q_sn)
    return np.where(ok, call, np.nan), np.where(ok, trust, np.nan)


def attach_call_trust(df: pd.DataFrame, trust_dir: str | Path) -> pd.DataFrame:
    """Scorer step: ``q_sn__`` / ``sn_call__`` / ``call_trust__`` for every expert
    with a trust head, when the trust dir carries call-trust heads (otherwise ``df``
    is returned untouched).  Call after ``attach_trust_columns`` (needs ``q__`` for
    the SN-filter experts).  A ``q_sn__`` column already in ``df`` (the training
    snapshot, out-of-fold on train rows) is kept."""
    assets = load_call_trust_assets(trust_dir)
    if assets is None:
        return df
    pooled = PooledTrustView.load_pooled_assets(Path(trust_dir) / POOLED_SUBDIR)
    pooled_meta = dict(pooled.get("metadata") or {})
    df = df.copy()
    meta = assets["metadata"]
    new_cols: dict[str, Any] = {}
    keys = list(meta["experts"]) + list(meta.get("reused_sn_filter_experts", []))
    for key in keys:
        san = sanitize_expert_key(key)
        avail = _avail(df, san)
        qc = f"q_sn__{san}"
        if qc in df.columns and not df[qc].isna().all():
            q_sn = pd.to_numeric(df[qc], errors="coerce").to_numpy(dtype=float)
        elif key in meta.get("reused_sn_filter_experts", []):
            q_sn = (pd.to_numeric(df[f"q__{san}"], errors="coerce").to_numpy(dtype=float)
                    if f"q__{san}" in df.columns else np.full(len(df), np.nan))
        else:
            q_sn = np.full(len(df), np.nan)
            if avail.any():
                view = call_trust_view(key, assets, pooled_meta)
                q_sn[avail] = np.asarray(view.predict_trust(df), dtype=float)[avail]
        q_sn = np.where(avail, q_sn, np.nan)
        call, trust = call_trust_from_q_sn(q_sn, expert_p_sn(df, san), avail)
        new_cols[qc] = q_sn
        new_cols[f"sn_call__{san}"] = call
        new_cols[f"call_trust__{san}"] = trust
    collision = [c for c in new_cols if c in df.columns]
    base = df.drop(columns=collision) if collision else df
    return pd.concat([base, pd.DataFrame(new_cols, index=df.index)], axis=1)
