"""Pooled Stage-A trust model for fusion_v8.

One LightGBM binary classifier over long-format (object, epoch, expert) rows
replaces the per-expert trust heads of v6e2 so data-poor experts borrow split
structure ("if flip_rate>0.4 at n_det<5, distrust everyone's p_snia") from
data-rich ones.  Per-expert *semantics* are inherited verbatim from
``expert_trust.py``:

- target: claim-conditional correctness — ``is_topclass_correct`` for ternary
  experts, ``is_sn`` for ``SN_FILTER_EXPERTS`` (stacked into one column ``y``);
- honesty filters: drop ``latest_object_unsafe`` rows, drop
  ``broker_consensus`` label rows, drop ``alerce_self_label`` rows for ALeRCE
  experts (expert_trust.py:301-351);
- OOF protocol: GroupKFold(5) on object_id over the train split produces
  out-of-fold q for train rows; a refit-on-train model scores cal/test rows
  (expert_trust.py:372-416);
- hierarchical per-expert calibration isotonic → Platt → global isotonic with
  the v6e2 guards (cal_n >= 200, pos >= 20, neg >= 20, test AUC drop <= 0.05;
  expert_trust.py:427-475).

No-leakage argument
-------------------
Stage-A features are functions of (i) the truncated lightcurve at this n_det
(no-leakage proof inherited from ``features/lightcurve*.py``), (ii) the
expert's own as-of-alert_jd projection / trajectory, and (iii) static expert
metadata.  The target compares those as-of predictions to external truth, so
no feature encodes the label.  q values written back into the snapshot are OOF
for train objects (no row is scored by a model that saw its own object) and
refit-on-train for cal/test objects (disjoint by the locked split).
``q__``/``trust_source__`` columns are asserted absent from the feature set
(anti-stacking-leak), and split integrity is asserted loudly.

Per-expert safety net (spec correction #5): any data-rich expert
(>= ``DATA_RICH_MIN_ROWS`` helpfulness rows) whose pooled test AUC regresses
by more than ``FALLBACK_AUC_TOL`` versus a freshly trained v6e2-style
dedicated head falls back to that dedicated head inside ``PooledTrustView``.

q_prior readout (spec correction #4c): the pooled model scored with all
``own_*`` slots NaN'd answers "expected helpfulness if we fetched this
broker" and is emitted for ALL registered experts.

fusion v13 (docs/fusion_v13_plan.md) Stage-A corrections
--------------------------------------------------------
1. The SN-filter set used at training (``is_sn_filter`` is a Stage-A
   feature AND selects the trust target) is persisted in the pooled metadata
   (``sn_filter_experts``); inference reproduces it, falling back to
   ``expert_trust.LEGACY_SN_FILTER_EXPERTS`` for v8-v12 artifacts so their
   scores stay bit-identical.
2. ``q_prior__`` on train rows is OUT-OF-FOLD (``q_prior_oof=True``): the
   GroupKFold fold models fitted for ``q__`` also score the prior-mode
   matrix of every train object they held out.  Cal/test rows keep the
   refit-on-train readout, exactly as before.
3. ``q__``/``trust_source__`` are emitted only for experts with a trained
   head (``emit_headless_q=False``) — the scorer's ``attach_trust_columns``
   iterates trust dirs, so headless experts (context-only Babamul / Lasair
   Sherlock) never had a ``q__`` column at score time.
4. ``weak_policy`` gates ``label_quality == "weak"`` rows per expert:
   ``"all"`` (legacy default), ``"is_sn_only"`` (kept only for is_sn-target
   experts outside the ALeRCE family), ``"none"``.  Per-expert weak-row
   counts are recorded in the metrics/metadata.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from debass_meta.projectors import ALL_EXPERT_KEYS, EXPERT_REGISTRY, sanitize_expert_key

from .calibrate import IsotonicCalibrator
from .expert_trust import (
    LEGACY_SN_FILTER_EXPERTS,
    SN_FILTER_EXPERTS,
    _binary_metrics,
    _expert_feature_cols,
    _predict_binary_classifier,
    _prepare_numeric_frame,
    is_alerce_family,
    trust_target_col,
)

# --------------------------------------------------------------------------
# Column-space constants (pinned by the fusion_v8 spec)
# --------------------------------------------------------------------------

KEY_COLS = ["object_id", "n_det", "alert_jd"]

# Generic own-prediction slots: the expert's proj__<san>__* columns renamed
# into a SHARED column space so split structure transfers across experts.
OWN_PRED_SLOTS: dict[str, str] = {
    "own_p_snia": "p_snia",
    "own_p_nonIa": "p_nonIa_snlike",
    "own_p_other": "p_other",
    "own_top1": "top1_prob",
    "own_margin": "margin",
    "own_entropy": "entropy",
}

TRAJ_STAT_NAMES = ["n", "last", "mean", "std", "min", "max", "slope", "delta", "volatility", "ewm"]
OWN_TRAJ_SLOTS = [f"own_traj_{stat}" for stat in TRAJ_STAT_NAMES]

META_NUMERIC_COLS = [
    "is_sn_filter",
    "survey_match",
    "temporal_exactness_code",
    "log_expert_train_rows",
]
EXPERT_ID_COL = "expert_id"

TEMPORAL_EXACTNESS_CODE: dict[str, float] = {
    "exact_alert": 2.0,
    "rerun_exact": 1.5,
    "static_safe": 1.0,
    "latest_object_unsafe": 0.0,
}

# Stage-A label-quality weights (frozen decision; mirrors Stage-B w_qual).
LABEL_QUALITY_WEIGHTS: dict[str, float] = {
    "spectroscopic": 1.0,
    "tns_untyped": 0.6,
    "context": 0.15,
    "weak": 0.1,
}
DEFAULT_LABEL_QUALITY_WEIGHT = 0.1

# Spec correction #5: per-expert fallback gate thresholds (module-level so
# tests can monkeypatch them).
DATA_RICH_MIN_ROWS = 2000
FALLBACK_AUC_TOL = 0.02

POOLED_SUBDIR = "pooled"
GLOBAL_CALIBRATOR_FILENAME = "global_calibrator.pkl"

# fusion v13: weak-label admission policies for Stage A (see module doc §4).
WEAK_LABEL_QUALITY = "weak"
WEAK_POLICIES = ("all", "is_sn_only", "lsst_is_sn_only", "none")
# Which experts get a q_prior__ column: every registered expert (legacy,
# pinned by the v8 interface) or only those with a trained head (what the
# scorer's attach_trust_columns can reproduce).
Q_PRIOR_EXPERT_MODES = ("all", "trained")
# q_prior feature protocol persisted in the pooled metadata.  Version 2
# (fusion v13): the scorer builds prior-mode features exactly as training
# does (own pred + own traj slots NaN'd, modal exactness code) and emits
# q_prior__ for the persisted ``q_prior_experts`` set.  Artifacts without the
# key (v8-v12) are scored by the legacy path (proj__ columns masked only,
# trust dirs only) so their outputs stay bit-identical.
PRIOR_MODE_VERSION = 2


def _resolve_sn_filter_experts(sn_filter_experts) -> frozenset[str]:
    """Training-time SN-filter set: explicit override or the module default."""
    if sn_filter_experts is None:
        return frozenset(SN_FILTER_EXPERTS)
    return frozenset(str(key) for key in sn_filter_experts)


def weak_rows_allowed(
    weak_policy: str, expert_key: str, target_col: str, *, is_lsst: bool = True
) -> bool:
    """Whether ``label_quality == "weak"`` rows may train this expert's head.

    ``"all"``: legacy — always.  ``"none"``: never.  ``"is_sn_only"``: only
    when the head is trained against ``is_sn`` (a weak "SN" label is exactly
    right for an SN-vs-other target but grades every correct Ia call wrong
    under ``is_topclass_correct``) AND the expert is outside the ALeRCE family
    (the weak LSST labels are ALeRCE-stamp-derived — circular).
    ``"lsst_is_sn_only"``: the ``is_sn_only`` rule for LSST rows
    (``is_lsst=True``); ZTF rows (``is_lsst=False``) behave as under ``"all"``
    so the ZTF Stage-A rows match v12 (ZTF weak rows come from the ALeRCE LC
    classifier / labels.csv and DO carry Ia labels).
    """
    if weak_policy not in WEAK_POLICIES:
        raise ValueError(f"weak_policy must be one of {WEAK_POLICIES}, got {weak_policy!r}")
    if weak_policy == "all":
        return True
    if weak_policy == "none":
        return False
    if weak_policy == "lsst_is_sn_only" and not is_lsst:
        return True
    return target_col == "is_sn" and not is_alerce_family(expert_key)


def _row_is_lsst(frame: pd.DataFrame) -> np.ndarray:
    """Per-row LSST flag: ``survey_is_lsst`` (gold LC feature) where present,
    else a ``survey`` column, else the id shape (LSST diaObjectIds are pure
    digits; ZTF ids are ``ZTF..``)."""
    n = len(frame)
    out = np.full(n, np.nan)
    if "survey_is_lsst" in frame.columns:
        out = pd.to_numeric(frame["survey_is_lsst"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(out).any() and "survey" in frame.columns:
        sv = frame["survey"].astype(str).str.lower()
        known = sv.isin(["lsst", "ztf"]).to_numpy()
        fill = np.where(sv.to_numpy() == "lsst", 1.0, 0.0)
        out = np.where(np.isnan(out) & known, fill, out)
    if np.isnan(out).any():
        by_id = frame["object_id"].astype(str).str.fullmatch(r"\d+").to_numpy().astype(float)
        out = np.where(np.isnan(out), by_id, out)
    return out > 0.5

# Columns never allowed into the Stage-A feature matrix (label-derived,
# identifiers, absolute timestamps, free-text).
_BLOCKED_GENERIC = {
    "object_id",
    "expert_key",
    "expert_id",
    "available",
    "temporal_exactness",
    "mapped_pred_class",
    "target_class",
    "target_follow_proxy",
    "label_source",
    "label_quality",
    "prediction_type",
    "reason",
    "is_topclass_correct",
    "is_sn",
    "is_helpful_for_follow_proxy",
    "mapped_p_true_class",
    "alert_jd",
    "y",
    "target_col",
    "_split",
}

# Per-expert column families excluded from the *generic* LC-context block.
# The expert's own proj/traj columns re-enter via the own_* slots; q__ and
# trust_source__ are stacking leaks; traj_x__ is excluded per the design §1.4
# Stage-A block table (101 LC context + 6 own-pred + 10 own-traj + metadata).
_PER_EXPERT_PREFIXES = (
    "proj__",
    "avail__",
    "exact__",
    "event_count__",
    "source_event_time_jd__",
    "temporal_exactness__",
    "mapped_pred_class__",
    "prediction_type__",
    "reason__",
    "q__",
    "q_prior__",
    "trust_source__",
    "traj__",
    "traj_x__",
)


# --------------------------------------------------------------------------
# Result / view dataclasses (pinned interface)
# --------------------------------------------------------------------------


@dataclass
class PooledTrustResult:
    """Pinned return type of :func:`train_pooled_trust`.

    snapshots : input snapshots + ``q__<san>`` (float, NaN where the expert is
        absent) + ``q_prior__<san>`` (all registered experts) +
        ``trust_source__<san>`` (oof / train_model / unavailable).
    metrics : per-expert dict with at least
        {raw_auc, cal_auc, brier, ece, n_test, calibrator_kind, fallback_used}.
    artifact_dir : root of the ``models/trust_fusion_v8``-style layout.
    """

    snapshots: pd.DataFrame
    metrics: dict
    artifact_dir: str


# --------------------------------------------------------------------------
# Calibrators
# --------------------------------------------------------------------------


class PlattCalibrator:
    """2-parameter logistic calibration on logit(q_raw).

    Stable where isotonic collapses (the documented fink_lsst/early_snia
    incident): two parameters are learnable from ~30 cal points, and the map
    is monotone so it cannot change ranking metrics.
    """

    name = "platt"

    def __init__(self) -> None:
        self._lr = None

    @staticmethod
    def _logit(y_prob: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1.0 - 1e-6)
        return np.log(p / (1.0 - p))

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        from sklearn.linear_model import LogisticRegression

        x = self._logit(y_prob).reshape(-1, 1)
        y = np.asarray(y_true, dtype=int)
        self._lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=42)
        self._lr.fit(x, y)
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        if self._lr is None:
            return np.asarray(y_prob, dtype=float)
        x = self._logit(y_prob).reshape(-1, 1)
        class_index = list(self._lr.classes_).index(1)
        return self._lr.predict_proba(x)[:, class_index].astype(float)


# --------------------------------------------------------------------------
# Long-format assembly
# --------------------------------------------------------------------------


def discover_generic_context_cols(df: pd.DataFrame) -> list[str]:
    """Numeric object-level context columns shared by every expert row.

    Excludes per-expert column families, label-derived columns, identifiers
    and absolute timestamps.  No information-content pruning here — that is
    done on the train split only (see :func:`train_pooled_trust`).
    """
    cols: list[str] = []
    own_slots = set(OWN_PRED_SLOTS) | set(OWN_TRAJ_SLOTS) | set(META_NUMERIC_COLS)
    for column in df.columns:
        if column in _BLOCKED_GENERIC or column in own_slots:
            continue
        if column.startswith(_PER_EXPERT_PREFIXES):
            continue
        if not is_numeric_dtype(df[column]):
            continue
        cols.append(str(column))
    return sorted(set(cols))


def _expert_survey_scope(expert_key: str) -> str:
    return EXPERT_REGISTRY.get(expert_key, ("any", ""))[0]


def _survey_match_values(frame: pd.DataFrame, expert_key: str) -> np.ndarray:
    """1.0 when the expert's survey scope matches the object's survey.

    'any'-scope experts always match.  NaN when the object survey is unknown.
    """
    scope = _expert_survey_scope(expert_key)
    n = len(frame)
    if scope == "any":
        return np.ones(n, dtype=float)
    if "survey_is_lsst" not in frame.columns:
        return np.full(n, np.nan)
    is_lsst = pd.to_numeric(frame["survey_is_lsst"], errors="coerce").to_numpy(dtype=float)
    want = 1.0 if scope == "lsst" else 0.0
    out = np.where(np.isnan(is_lsst), np.nan, (np.round(is_lsst) == want).astype(float))
    return out


def _assemble_features_for_expert(
    frame: pd.DataFrame,
    expert_key: str,
    *,
    generic_cols: list[str],
    log_rows: float | None = None,
    default_exact_code: float | None = None,
    prior_mode: bool = False,
    sn_filter_experts=None,
) -> pd.DataFrame:
    """Build the Stage-A numeric feature block for one expert.

    Works on BOTH frame shapes:
    - helpfulness long rows (own proj cols + a flat ``temporal_exactness``);
    - wide gold snapshots (own proj cols + ``temporal_exactness__<san>``).

    ``prior_mode=True`` NaN's every own_* slot (own predictions AND own
    trajectory) — the q_prior counterfactual "we have not fetched this broker
    yet" (spec correction #4c) — and uses the expert's training-time modal
    temporal-exactness code.

    ``sn_filter_experts`` is the SN-filter set that defines the
    ``is_sn_filter`` feature.  It MUST be the set the model was trained with
    (persisted in the pooled metadata); ``None`` means the current module
    default, which is only correct at training time.
    """
    san = sanitize_expert_key(expert_key)
    sn_filter = _resolve_sn_filter_experts(sn_filter_experts)
    out = pd.DataFrame(index=frame.index)

    for column in generic_cols:
        if column in frame.columns:
            out[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            out[column] = np.nan

    # Own prediction slots (shared column space across experts).
    for slot, suffix in OWN_PRED_SLOTS.items():
        source = f"proj__{san}__{suffix}"
        if prior_mode or source not in frame.columns:
            out[slot] = np.nan
        else:
            out[slot] = pd.to_numeric(frame[source], errors="coerce")
    # Scalar fallback for experts that only emit p_snia_scalar (mirrors
    # build_expert_helpfulness.py's fallback for mapped_p_true_class).
    scalar_col = f"proj__{san}__p_snia_scalar"
    if not prior_mode and scalar_col in frame.columns:
        scalar = pd.to_numeric(frame[scalar_col], errors="coerce")
        out["own_p_snia"] = out["own_p_snia"].fillna(scalar)

    # Own trajectory slots (NaN for non-traj experts / pre-traj tables).
    for slot, stat in zip(OWN_TRAJ_SLOTS, TRAJ_STAT_NAMES):
        source = f"traj__{san}__{stat}"
        if prior_mode or source not in frame.columns:
            out[slot] = np.nan
        else:
            out[slot] = pd.to_numeric(frame[source], errors="coerce")

    # Expert metadata.
    out["is_sn_filter"] = 1.0 if expert_key in sn_filter else 0.0
    out["survey_match"] = _survey_match_values(frame, expert_key)

    exact_col = f"temporal_exactness__{san}"
    if prior_mode:
        codes = pd.Series(np.nan, index=frame.index)
    elif exact_col in frame.columns:
        codes = frame[exact_col].map(TEMPORAL_EXACTNESS_CODE)
    elif "temporal_exactness" in frame.columns:
        codes = frame["temporal_exactness"].map(TEMPORAL_EXACTNESS_CODE)
    else:
        codes = pd.Series(np.nan, index=frame.index)
    codes = pd.to_numeric(codes, errors="coerce")
    if default_exact_code is not None:
        codes = codes.fillna(float(default_exact_code))
    out["temporal_exactness_code"] = codes

    out["log_expert_train_rows"] = float(log_rows) if log_rows is not None else np.nan
    out["expert_key"] = expert_key
    return out


def assemble_stage_a_long(
    helpfulness: pd.DataFrame,
    *,
    apply_honesty_filters: bool = True,
    weak_policy: str = "all",
    sn_filter_experts=None,
) -> tuple[pd.DataFrame, list[str], dict[str, tuple[pd.DataFrame, str]]]:
    """Assemble the pooled Stage-A long table from a helpfulness table.

    Returns (long_df, generic_cols, expert_frames) where ``long_df`` has the
    key columns, the stacked target ``y`` (NaN preserved so callers/tests can
    inspect base rates), ``target_col``, weighting columns, and the numeric
    feature blocks; ``expert_frames`` maps expert_key -> (filtered original
    helpfulness sub-frame, target_col) for the dedicated-head fallback path.

    Honesty filters are verbatim from expert_trust.py:301-351.

    ``weak_policy`` (fusion v13) gates ``label_quality == "weak"`` rows per
    expert AFTER the target column is resolved (see :func:`weak_rows_allowed`;
    the LSST/ZTF decision is per row for ``lsst_is_sn_only``); the per-expert
    ledger ``{expert: {allowed_lsst, allowed_ztf, allowed, kept, dropped,
    kept_lsst, dropped_lsst}}`` is attached as ``long_df.attrs["weak_ledger"]``.
    ``sn_filter_experts`` fixes the SN-filter set (target selection AND the
    ``is_sn_filter`` feature); ``None`` = module default.
    """
    if weak_policy not in WEAK_POLICIES:
        raise ValueError(f"weak_policy must be one of {WEAK_POLICIES}, got {weak_policy!r}")
    sn_filter = _resolve_sn_filter_experts(sn_filter_experts)
    generic_cols = discover_generic_context_cols(helpfulness)
    long_parts: list[pd.DataFrame] = []
    expert_frames: dict[str, tuple[pd.DataFrame, str]] = {}
    weak_ledger: dict[str, dict[str, Any]] = {}

    for expert_key in sorted(str(key) for key in helpfulness["expert_key"].dropna().unique()):
        sub = helpfulness[helpfulness["expert_key"] == expert_key].copy()
        if apply_honesty_filters:
            if "temporal_exactness" in sub.columns:
                sub = sub[sub["temporal_exactness"] != "latest_object_unsafe"]
            # Anti-circularity: never grade a broker against its own labels.
            if "label_source" in sub.columns:
                sub = sub[sub["label_source"] != "broker_consensus"]
                if is_alerce_family(expert_key):
                    sub = sub[sub["label_source"] != "alerce_self_label"]
        if len(sub) == 0:
            continue

        target_col = trust_target_col(expert_key, sn_filter)
        if target_col not in sub.columns or sub[target_col].dropna().empty:
            # Legacy fallback (verbatim): helpfulness table pre-dates is_sn.
            if target_col != "is_topclass_correct":
                if (
                    "is_topclass_correct" not in sub.columns
                    or sub["is_topclass_correct"].dropna().empty
                ):
                    continue
                target_col = "is_topclass_correct"
            else:
                continue

        # fusion v13: weak-label admission (after the target is known); the
        # decision is per survey so "lsst_is_sn_only" can leave ZTF rows alone.
        allowed_lsst = weak_rows_allowed(weak_policy, expert_key, target_col, is_lsst=True)
        allowed_ztf = weak_rows_allowed(weak_policy, expert_key, target_col, is_lsst=False)
        if "label_quality" in sub.columns:
            is_weak = (sub["label_quality"].astype(str) == WEAK_LABEL_QUALITY).to_numpy()
        else:
            is_weak = np.zeros(len(sub), dtype=bool)
        is_lsst_row = _row_is_lsst(sub)
        drop = is_weak & (
            (is_lsst_row & (not allowed_lsst)) | (~is_lsst_row & (not allowed_ztf))
        )
        kept = is_weak & ~drop
        weak_ledger[expert_key] = {
            "allowed_lsst": bool(allowed_lsst),
            "allowed_ztf": bool(allowed_ztf),
            "allowed": bool(allowed_lsst and allowed_ztf),
            "target_col": target_col,
            "kept": int(kept.sum()),
            "dropped": int(drop.sum()),
            "kept_lsst": int((kept & is_lsst_row).sum()),
            "dropped_lsst": int((drop & is_lsst_row).sum()),
        }
        if drop.any():
            sub = sub[~drop]
        if len(sub) == 0:
            continue

        feats = _assemble_features_for_expert(
            sub, expert_key, generic_cols=generic_cols, sn_filter_experts=sn_filter
        )
        feats["object_id"] = sub["object_id"].astype(str)
        feats["n_det"] = pd.to_numeric(sub["n_det"], errors="coerce")
        feats["alert_jd"] = pd.to_numeric(sub["alert_jd"], errors="coerce")
        feats["y"] = pd.to_numeric(sub[target_col], errors="coerce")
        feats["target_col"] = target_col
        feats["label_quality"] = sub["label_quality"] if "label_quality" in sub.columns else None
        long_parts.append(feats)
        expert_frames[expert_key] = (sub, target_col)

    if not long_parts:
        raise ValueError("assemble_stage_a_long: no expert rows survived assembly")
    long_df = pd.concat(long_parts, ignore_index=True, sort=False)
    long_df.attrs["weak_ledger"] = weak_ledger
    long_df.attrs["weak_policy"] = weak_policy
    return long_df, generic_cols, expert_frames


# --------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------


def stage_a_weights(df: pd.DataFrame) -> np.ndarray:
    """w_i = w_qual(i) * w_expert(e_i) * 1 / n_rows(obj_i, e_i).

    w_expert(e) = (M_total / (E * M_e)) ** 0.5 — sqrt-tempered expert
    balancing so a high-coverage expert cannot drown a 180-row one while the
    small expert still borrows the shared split structure.
    """
    if "label_quality" in df.columns:
        w_qual = (
            df["label_quality"]
            .map(LABEL_QUALITY_WEIGHTS)
            .fillna(DEFAULT_LABEL_QUALITY_WEIGHT)
            .to_numpy(dtype=float)
        )
    else:
        w_qual = np.ones(len(df), dtype=float)

    counts = df["expert_key"].value_counts()
    m_total = float(len(df))
    n_experts = float(len(counts))
    w_expert = (
        df["expert_key"].map(lambda e: np.sqrt(m_total / (n_experts * float(counts[e]))))
        .to_numpy(dtype=float)
    )

    pair_counts = df.groupby(["object_id", "expert_key"])["expert_key"].transform("count")
    w_pair = 1.0 / pair_counts.to_numpy(dtype=float)
    return w_qual * w_expert * w_pair


# --------------------------------------------------------------------------
# Pooled LightGBM fit / predict
# --------------------------------------------------------------------------


def _pooled_base_kwargs(seed: int, n_jobs: int) -> dict[str, Any]:
    return dict(
        objective="binary",
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        feature_fraction=0.7,
        bagging_fraction=0.8,
        bagging_freq=1,
        reg_alpha=0.1,
        reg_lambda=0.1,
        max_cat_threshold=32,
        cat_smooth=20,
        random_state=seed,
        deterministic=True,
        force_row_wise=True,
        n_jobs=n_jobs,
        verbose=-1,
    )


def _pooled_param_grid(grid_small: bool) -> list[dict[str, Any]]:
    if grid_small:
        return [
            {"num_leaves": 31, "min_child_samples": 20, "learning_rate": 0.05, "reg_lambda": 0.1}
        ]
    return [
        {
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "learning_rate": learning_rate,
            "reg_lambda": reg_lambda,
        }
        for num_leaves in (15, 31, 63)
        for min_child_samples in (20, 50, 100)
        for learning_rate in (0.03, 0.05)
        for reg_lambda in (0.1, 1.0)
    ]


def _prepare_pooled_matrix(
    df: pd.DataFrame, feature_cols: list[str], expert_levels: list[str]
) -> pd.DataFrame:
    """Numeric features + the expert_id categorical, fixed column order.

    Categories are pinned to ``expert_levels`` so codes are identical between
    training and any later prediction frame.
    """
    data: dict[str, Any] = {}
    for column in feature_cols:
        if column in df.columns:
            data[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            data[column] = np.full(len(df), np.nan)
    X = pd.DataFrame(data, index=df.index)
    X[EXPERT_ID_COL] = pd.Categorical(
        df["expert_key"].astype(str), categories=list(expert_levels)
    )
    return X


def _fit_pooled_classifier(
    X: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    *,
    params: dict[str, Any],
    n_estimators: int,
    seed: int,
    n_jobs: int,
):
    from lightgbm import LGBMClassifier

    unique = np.unique(y)
    if len(unique) < 2:
        return {"kind": "constant", "prob": float(unique[0]) if len(unique) else 0.0}
    kwargs = {**_pooled_base_kwargs(seed, n_jobs), **params}
    model = LGBMClassifier(n_estimators=int(n_estimators), **kwargs)
    model.fit(X, y, sample_weight=w, categorical_feature=[EXPERT_ID_COL])
    return {"kind": "sklearn", "model": model}


def _weighted_logloss(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1.0 - 1e-7)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    return float(-(w * (y * np.log(p) + (1.0 - y) * np.log(1.0 - p))).sum() / w.sum())


def _select_pooled_params(
    X: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    n_jobs: int,
    grid_small: bool,
) -> tuple[dict[str, Any], int]:
    """Small-grid selection on an inner grouped fold carved from TRAIN.

    Cal is never touched by model fitting — it is reserved for calibration
    (v6e2 Phase-0.2 discipline).  Returns (best params, n_estimators from
    early stopping).

    Speedup escape hatch: DEBASS_POOLED_TRUST_PARAMS (JSON with keys
    'params' and 'n_estimators') skips the grid entirely.  ONLY safe when the
    value is the deterministic winner of a previous identical run (same data,
    seed, grid) — e.g. re-running after a crash later in the pipeline.
    """
    import os

    override = os.environ.get("DEBASS_POOLED_TRUST_PARAMS")
    if override:
        payload = json.loads(override)
        params = dict(payload["params"])
        n_estimators = int(payload["n_estimators"])
        print(
            f"  pooled_trust: GRID SKIPPED — params pinned via "
            f"DEBASS_POOLED_TRUST_PARAMS: {params}, n_estimators={n_estimators}",
            flush=True,
        )
        return params, n_estimators

    configs = _pooled_param_grid(grid_small)
    default = (dict(configs[0]), 300)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 5 or len(np.unique(y)) < 2:
        return default

    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    fit_idx, val_idx = next(splitter.split(X, y, groups=groups))
    if len(np.unique(y[fit_idx])) < 2 or len(np.unique(y[val_idx])) < 2:
        return default

    best: tuple[float, dict[str, Any], int] | None = None
    for config in configs:
        kwargs = {**_pooled_base_kwargs(seed, n_jobs), **config}
        model = LGBMClassifier(n_estimators=2000, **kwargs)
        model.fit(
            X.iloc[fit_idx],
            y[fit_idx],
            sample_weight=w[fit_idx],
            eval_set=[(X.iloc[val_idx], y[val_idx])],
            eval_sample_weight=[w[val_idx]],
            eval_metric="binary_logloss",
            categorical_feature=[EXPERT_ID_COL],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        class_index = list(model.classes_).index(1)
        p_val = model.predict_proba(X.iloc[val_idx])[:, class_index]
        loss = _weighted_logloss(y[val_idx], p_val, w[val_idx])
        best_iter = int(getattr(model, "best_iteration_", None) or model.n_estimators)
        if best is None or loss < best[0] - 1e-12:
            best = (loss, dict(config), max(best_iter, 30))
    if best is None:
        return default
    return best[1], best[2]


def _oof_pooled_fits(
    X: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    groups: np.ndarray,
    *,
    params: dict[str, Any],
    n_estimators: int,
    seed: int,
    n_jobs: int,
    n_splits: int = 5,
) -> tuple[np.ndarray, list[Any], dict[str, int]]:
    """GroupKFold(5) OOF on train rows — anti-stacking-leak, same as v6e2.

    Returns ``(oof, fold_bundles, fold_of_group)``: the OOF predictions, the
    fitted fold models (fold k was fit WITHOUT the objects it predicted) and
    the held-out fold index of every train object.  fusion v13 reuses the
    fold models to score the prior-mode matrix of train objects so
    ``q_prior__`` is out-of-fold too (module doc §2).  When fewer than two
    object groups exist there are no folds: the in-sample fit is returned as
    ``oof`` and both fold structures are empty.
    """
    from debass_meta.models.folds import StableGroupKFold as GroupKFold  # CPU-independent folds (v13f)

    oof = np.full(len(y), np.nan, dtype=float)
    fold_bundles: list[Any] = []
    fold_of_group: dict[str, int] = {}
    unique_groups = np.unique(groups)
    splits = min(int(n_splits), len(unique_groups))
    if splits >= 2:
        group_kfold = GroupKFold(n_splits=splits)
        for fold, (fit_idx, pred_idx) in enumerate(group_kfold.split(X, y, groups=groups)):
            bundle = _fit_pooled_classifier(
                X.iloc[fit_idx], y[fit_idx], w[fit_idx],
                params=params, n_estimators=n_estimators, seed=seed, n_jobs=n_jobs,
            )
            oof[pred_idx] = _predict_binary_classifier(bundle, X.iloc[pred_idx])
            fold_bundles.append(bundle)
            for group in np.unique(groups[pred_idx]):
                fold_of_group[str(group)] = fold
    else:
        bundle = _fit_pooled_classifier(
            X, y, w, params=params, n_estimators=n_estimators, seed=seed, n_jobs=n_jobs
        )
        oof[:] = _predict_binary_classifier(bundle, X)
    return oof, fold_bundles, fold_of_group


def _oof_pooled_predictions(
    X: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    groups: np.ndarray,
    *,
    params: dict[str, Any],
    n_estimators: int,
    seed: int,
    n_jobs: int,
    n_splits: int = 5,
) -> np.ndarray:
    """OOF predictions only (thin wrapper kept for the v8 call surface)."""
    return _oof_pooled_fits(
        X, y, w, groups,
        params=params, n_estimators=n_estimators, seed=seed, n_jobs=n_jobs,
        n_splits=n_splits,
    )[0]


# --------------------------------------------------------------------------
# Hierarchical per-expert calibration (guards verbatim from expert_trust.py)
# --------------------------------------------------------------------------


def _fit_expert_calibrator(
    q_cal_raw: np.ndarray,
    y_cal: np.ndarray,
    q_test_raw: np.ndarray,
    y_test: np.ndarray,
    global_calibrator: IsotonicCalibrator | None,
) -> tuple[Any, str, list[str]]:
    """isotonic -> Platt -> global hierarchy.  Returns (calibrator, kind, notes).

    Tier-1 guards verbatim from expert_trust.py:427-475 (cal_n >= 200,
    pos >= 20, neg >= 20, post-fit test AUC drop <= 0.05).  Platt needs
    cal_n >= 30 with both classes; it is monotone, so no AUC guard is needed.
    """
    notes: list[str] = []
    q_cal_raw = np.asarray(q_cal_raw, dtype=float)
    y_cal = np.asarray(y_cal, dtype=float)
    q_test_raw = np.asarray(q_test_raw, dtype=float)
    y_test = np.asarray(y_test, dtype=float)

    valid = ~np.isnan(q_cal_raw) & np.isin(y_cal, [0, 1])
    n_valid = int(valid.sum())
    n_pos = int(y_cal[valid].sum()) if n_valid > 0 else 0
    n_neg = n_valid - n_pos

    iso_skip_reason: str | None = None
    if n_valid < 200:
        iso_skip_reason = f"cal_n<200 (have {n_valid})"
    elif n_pos < 20:
        iso_skip_reason = f"cal_pos<20 (have {n_pos})"
    elif n_neg < 20:
        iso_skip_reason = f"cal_neg<20 (have {n_neg})"
    elif len(np.unique(y_cal[valid])) != 2:
        iso_skip_reason = "single_class_in_cal"
    else:
        candidate = IsotonicCalibrator()
        candidate.fit(q_cal_raw[valid], y_cal[valid].astype(int))
        if len(y_test) > 20:
            guard_valid = ~np.isnan(q_test_raw) & np.isin(y_test, [0, 1])
            if guard_valid.sum() > 20 and len(np.unique(y_test[guard_valid])) == 2:
                from sklearn.metrics import roc_auc_score

                try:
                    auc_raw = roc_auc_score(y_test[guard_valid], q_test_raw[guard_valid])
                    auc_cal = roc_auc_score(
                        y_test[guard_valid],
                        np.asarray(candidate.transform(q_test_raw[guard_valid]), dtype=float),
                    )
                    if (auc_raw - auc_cal) > 0.05:
                        iso_skip_reason = f"auc_drop>0.05 (raw={auc_raw:.3f}, cal={auc_cal:.3f})"
                    else:
                        return candidate, "isotonic", notes
                except Exception:
                    # Keep it if the guard fails for non-AUC reasons (verbatim).
                    return candidate, "isotonic", notes
            else:
                return candidate, "isotonic", notes
        else:
            return candidate, "isotonic", notes
    notes.append(f"isotonic skipped ({iso_skip_reason})")

    if n_valid >= 30 and len(np.unique(y_cal[valid])) == 2:
        try:
            platt = PlattCalibrator()
            platt.fit(q_cal_raw[valid], y_cal[valid].astype(int))
            return platt, "platt", notes
        except Exception as exc:  # pragma: no cover - defensive
            notes.append(f"platt failed ({exc})")
    else:
        notes.append(f"platt skipped (cal_n<30 or single class; have {n_valid})")

    if global_calibrator is not None:
        return global_calibrator, "global", notes
    notes.append("no global calibrator available")
    return None, "none", notes


# --------------------------------------------------------------------------
# Dedicated per-expert fallback head (spec correction #5)
# --------------------------------------------------------------------------


def _fit_dedicated_classifier(X: pd.DataFrame, y: np.ndarray, *, seed: int, n_jobs: int):
    """v6e2-style dedicated head (expert_trust._fit_binary_classifier params)
    with deterministic LightGBM flags added per fusion_v8 coding standards."""
    from lightgbm import LGBMClassifier

    unique = np.unique(y)
    if len(unique) < 2:
        return {"kind": "constant", "prob": float(unique[0]) if len(unique) else 0.0}
    model = LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=5,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        reg_alpha=0.1,
        reg_lambda=0.1,
        is_unbalance=True,
        random_state=seed,
        n_jobs=n_jobs,
        verbose=-1,
        deterministic=True,
        force_row_wise=True,
    )
    model.fit(X, y)
    return {"kind": "sklearn", "model": model}


def _should_fallback(
    dedicated_auc: float | None, pooled_auc: float | None, tol: float | None = None
) -> bool:
    """Per-expert hard gate: dedicated head beating pooled by > tol AUC."""
    if tol is None:
        tol = FALLBACK_AUC_TOL
    if dedicated_auc is None or pooled_auc is None:
        return False
    return (float(dedicated_auc) - float(pooled_auc)) > float(tol)


def _train_dedicated_head(
    expert_rows: pd.DataFrame,
    expert_key: str,
    target_col: str,
    train_ids: set[str],
    cal_ids: set[str],
    test_ids: set[str],
    *,
    seed: int,
    n_jobs: int,
    n_splits: int = 5,
) -> dict[str, Any] | None:
    """Train a v6e2-style per-expert head on the (already honesty-filtered)
    helpfulness sub-frame; same OOF protocol as the pooled model."""
    from debass_meta.models.folds import StableGroupKFold as GroupKFold  # CPU-independent folds (v13f)

    rows = expert_rows[pd.to_numeric(expert_rows[target_col], errors="coerce").notna()].copy()
    if len(rows) == 0:
        return None
    feature_cols = _expert_feature_cols(rows, expert_key)
    leakage_cols = [c for c in feature_cols if c.startswith(("q__", "trust_source__"))]
    assert len(leakage_cols) == 0, f"Stacking leakage detected for {expert_key}: {leakage_cols}"

    object_ids = rows["object_id"].astype(str)
    train_mask = object_ids.isin(train_ids).to_numpy()
    cal_mask = object_ids.isin(cal_ids).to_numpy()
    test_mask = object_ids.isin(test_ids).to_numpy()
    if int(train_mask.sum()) == 0:
        return None

    X_all, fill_values = _prepare_numeric_frame(rows, feature_cols)
    y_all = pd.to_numeric(rows[target_col], errors="coerce").astype(int).to_numpy()

    train_X = X_all[train_mask]
    train_y = y_all[train_mask]
    groups = object_ids[train_mask].to_numpy()
    unique_groups = np.unique(groups)
    splits = min(int(n_splits), len(unique_groups))
    oof = np.full(len(train_y), np.nan, dtype=float)
    if splits >= 2:
        group_kfold = GroupKFold(n_splits=splits)
        for fit_idx, pred_idx in group_kfold.split(train_X, train_y, groups=groups):
            fold_bundle = _fit_dedicated_classifier(
                train_X.iloc[fit_idx], train_y[fit_idx], seed=seed, n_jobs=n_jobs
            )
            oof[pred_idx] = _predict_binary_classifier(fold_bundle, train_X.iloc[pred_idx])

    bundle = _fit_dedicated_classifier(train_X, train_y, seed=seed, n_jobs=n_jobs)
    if splits < 2:
        oof[:] = _predict_binary_classifier(bundle, train_X)

    # Experts whose rows all come from train/cal-only objects (e.g. the LSST
    # weak-label cohort: fink_lsst/pittgoogle experts have ZERO rows in the
    # locked ZTF test set) yield empty masks here — LightGBM refuses empty
    # input, and an absent slice simply means "no comparison possible"
    # (_should_fallback(None, ...) already returns False).
    q_cal = (
        _predict_binary_classifier(bundle, X_all[cal_mask])
        if int(cal_mask.sum()) > 0 else np.zeros(0, dtype=float)
    )
    q_test = (
        _predict_binary_classifier(bundle, X_all[test_mask])
        if int(test_mask.sum()) > 0 else np.zeros(0, dtype=float)
    )
    test_metrics = _binary_metrics(y_all[test_mask], q_test)

    oof_lookup: dict[tuple[str, int, float], float] = {}
    train_rows = rows[train_mask]
    for key, value in zip(_normalized_keys(train_rows), oof):
        oof_lookup[key] = float(value)

    return {
        "bundle": bundle,
        "feature_cols": feature_cols,
        "fill_values": fill_values,
        "oof_lookup": oof_lookup,
        "q_cal": q_cal,
        "y_cal": y_all[cal_mask],
        "q_test": q_test,
        "y_test": y_all[test_mask],
        "test_auc": test_metrics.get("roc_auc"),
    }


# --------------------------------------------------------------------------
# Key normalisation for OOF write-back
# --------------------------------------------------------------------------


def _normalized_keys(frame: pd.DataFrame) -> list[tuple[str, int, float]]:
    """(object_id, n_det, alert_jd) tuples robust to int/float dtype drift
    between the helpfulness table and the gold snapshot."""
    object_ids = frame["object_id"].astype(str).to_numpy()
    n_dets = pd.to_numeric(frame["n_det"], errors="coerce").to_numpy(dtype=float)
    jds = pd.to_numeric(frame["alert_jd"], errors="coerce").to_numpy(dtype=float)
    keys: list[tuple[str, int, float]] = []
    for oid, n_det, jd in zip(object_ids, n_dets, jds):
        n_key = int(round(n_det)) if np.isfinite(n_det) else -1
        jd_key = round(float(jd), 6) if np.isfinite(jd) else float("nan")
        keys.append((oid, n_key, jd_key))
    return keys


# --------------------------------------------------------------------------
# Main training entry point (pinned interface)
# --------------------------------------------------------------------------


def train_pooled_trust(
    helpfulness: pd.DataFrame,
    snapshots: pd.DataFrame,
    train_ids: set,
    cal_ids: set,
    test_ids: set,
    out_dir: str,
    *,
    n_jobs: int = 8,
    seed: int = 42,
    grid_small: bool = False,
    weak_policy: str = "all",
    sn_filter_experts=None,
    q_prior_oof: bool = True,
    emit_headless_q: bool = False,
    q_prior_experts: str = "all",
) -> PooledTrustResult:
    """Train the pooled Stage-A trust model and emit q into the snapshots.

    Parameters mirror the pinned fusion_v8 interface.  ``grid_small=True``
    (additive keyword) collapses the design §5.2 hyperparameter grid to a
    single config — used by unit tests and smoke runs.

    fusion v13 keywords (all additive; module doc):
    - ``weak_policy``: "all" (legacy default) | "is_sn_only" | "none" —
      admission of ``label_quality == "weak"`` rows per expert.
    - ``sn_filter_experts``: SN-filter set for this training run (default:
      ``expert_trust.SN_FILTER_EXPERTS``); persisted in the pooled metadata.
    - ``q_prior_oof``: score train objects' ``q_prior__`` with the GroupKFold
      fold model that held them out (default True; False = legacy in-sample
      refit readout).
    - ``emit_headless_q``: also emit ``q__``/``trust_source__`` for experts
      WITHOUT a trained head (legacy True behaviour; the scorer never
      produces those columns, so the default False matches serving).
    - ``q_prior_experts``: "all" registered experts (legacy default) or only
      "trained" ones.

    Emission rules:
    - ``q__<san>``: calibrated trust, float, NaN where ``avail__<san> == 0``;
    - ``trust_source__<san>``: 'oof' (train rows matched to OOF predictions),
      'train_model' (refit-on-train predictions), 'unavailable';
    - ``q_prior__<san>``: pooled model with all own_* slots NaN'd, emitted for
      ALL registered experts on every row (spec correction #4c); OOF on train
      objects when ``q_prior_oof``.
    """
    if weak_policy not in WEAK_POLICIES:
        raise ValueError(f"weak_policy must be one of {WEAK_POLICIES}, got {weak_policy!r}")
    if q_prior_experts not in Q_PRIOR_EXPERT_MODES:
        raise ValueError(
            f"q_prior_experts must be one of {Q_PRIOR_EXPERT_MODES}, got {q_prior_experts!r}"
        )
    sn_filter = _resolve_sn_filter_experts(sn_filter_experts)
    train_ids = {str(object_id) for object_id in train_ids}
    cal_ids = {str(object_id) for object_id in cal_ids}
    test_ids = {str(object_id) for object_id in test_ids}

    # --- Split integrity validation (loud) ---
    assert len(train_ids & cal_ids) == 0, "train/cal overlap detected"
    assert len(train_ids & test_ids) == 0, "train/test overlap detected"
    assert len(cal_ids & test_ids) == 0, "cal/test overlap detected"
    print(
        f"  pooled_trust: split validated train={len(train_ids):,}, "
        f"cal={len(cal_ids):,}, test={len(test_ids):,}"
    )
    print(
        f"  pooled_trust: weak_policy={weak_policy}, q_prior_oof={q_prior_oof}, "
        f"emit_headless_q={emit_headless_q}, q_prior_experts={q_prior_experts}, "
        f"sn_filter_experts={sorted(sn_filter)}"
    )

    out_dir_path = Path(out_dir)
    pooled_dir = out_dir_path / POOLED_SUBDIR
    pooled_dir.mkdir(parents=True, exist_ok=True)

    # --- Long-format assembly (honesty filters verbatim) ---
    long_df, generic_cols, expert_frames = assemble_stage_a_long(
        helpfulness,
        apply_honesty_filters=True,
        weak_policy=weak_policy,
        sn_filter_experts=sn_filter,
    )
    weak_ledger: dict[str, dict[str, Any]] = dict(long_df.attrs.get("weak_ledger", {}))
    for expert_key, entry in weak_ledger.items():
        if entry["dropped"]:
            print(
                f"    {expert_key}: weak_policy={weak_policy} dropped "
                f"{entry['dropped']:,} weak rows ({entry['dropped_lsst']:,} LSST)"
            )
    long_df = long_df[long_df["y"].notna()].reset_index(drop=True)
    if len(long_df) == 0:
        raise ValueError("train_pooled_trust: no labelled Stage-A rows after honesty filters")
    long_df["y"] = long_df["y"].astype(int)
    object_ids = long_df["object_id"].astype(str)
    long_df["_split"] = np.select(
        [object_ids.isin(train_ids), object_ids.isin(cal_ids), object_ids.isin(test_ids)],
        ["train", "cal", "test"],
        default="none",
    )

    train_mask = (long_df["_split"] == "train").to_numpy()
    cal_mask = (long_df["_split"] == "cal").to_numpy()
    test_mask = (long_df["_split"] == "test").to_numpy()
    if int(train_mask.sum()) == 0:
        raise ValueError("train_pooled_trust: no training rows in the train split")

    # --- Expert coverage prior + modal temporal exactness (per expert) ---
    train_counts = (
        long_df.loc[train_mask, "expert_key"].value_counts().to_dict()
    )
    expert_train_rows = {str(k): int(v) for k, v in train_counts.items()}
    long_df["log_expert_train_rows"] = (
        long_df["expert_key"].map(lambda e: np.log1p(float(expert_train_rows.get(e, 0))))
    )
    default_exact_codes: dict[str, float | None] = {}
    for expert_key in expert_frames:
        rows_e = long_df[long_df["expert_key"] == expert_key]
        rows_pref = rows_e[rows_e["_split"] == "train"]
        codes = rows_pref["temporal_exactness_code"].dropna()
        if codes.empty:
            codes = rows_e["temporal_exactness_code"].dropna()
        default_exact_codes[expert_key] = float(codes.mode().iloc[0]) if not codes.empty else None

    # --- Feature schema: prune uninformative generic cols on TRAIN only ---
    train_long = long_df[train_mask]
    generic_final: list[str] = []
    for column in generic_cols:
        series = train_long[column]
        if series.isna().all():
            continue
        if series.nunique(dropna=True) <= 1:
            continue
        generic_final.append(column)
    feature_cols = generic_final + list(OWN_PRED_SLOTS) + OWN_TRAJ_SLOTS + META_NUMERIC_COLS
    leakage_cols = [c for c in feature_cols if c.startswith(("q__", "q_prior__", "trust_source__"))]
    assert len(leakage_cols) == 0, f"Stage-A stacking leakage detected: {leakage_cols}"

    expert_levels = sorted(set(ALL_EXPERT_KEYS) | set(expert_frames))
    X_all = _prepare_pooled_matrix(long_df, feature_cols, expert_levels)
    y_all = long_df["y"].to_numpy(dtype=int)
    weights_train = stage_a_weights(long_df[train_mask])
    groups_train = long_df.loc[train_mask, "object_id"].astype(str).to_numpy()
    print(
        f"  pooled_trust: {int(train_mask.sum()):,} train rows over "
        f"{len(expert_frames)} experts, {len(feature_cols)} numeric features + expert_id"
    )

    # --- Hyperparameter selection on an inner grouped fold of train ---
    params, n_estimators = _select_pooled_params(
        X_all[train_mask], y_all[train_mask], weights_train, groups_train,
        seed=seed, n_jobs=n_jobs, grid_small=grid_small,
    )
    print(f"  pooled_trust: params={params}, n_estimators={n_estimators}")

    # --- OOF on train; refit-on-train for cal/test (v6e2 protocol) ---
    # The fold models are kept so q_prior__ on train objects can be scored
    # out-of-fold too (fusion v13 §2); the OOF q__ values are unchanged.
    oof_train, prior_fold_bundles, prior_fold_of_object = _oof_pooled_fits(
        X_all[train_mask], y_all[train_mask], weights_train, groups_train,
        params=params, n_estimators=n_estimators, seed=seed, n_jobs=n_jobs,
    )
    if not q_prior_oof:
        prior_fold_bundles, prior_fold_of_object = [], {}
    pooled_bundle = _fit_pooled_classifier(
        X_all[train_mask], y_all[train_mask], weights_train,
        params=params, n_estimators=n_estimators, seed=seed, n_jobs=n_jobs,
    )
    q_raw_refit = _predict_binary_classifier(pooled_bundle, X_all)

    pooled_oof_lookup: dict[str, dict[tuple[str, int, float], float]] = {}
    train_rows_df = long_df[train_mask]
    train_keys = _normalized_keys(train_rows_df)
    for expert_key, key, value in zip(
        train_rows_df["expert_key"].to_numpy(), train_keys, oof_train
    ):
        pooled_oof_lookup.setdefault(str(expert_key), {})[key] = float(value)

    # --- Global (tier-3) calibrator on all pooled cal rows ---
    global_calibrator: IsotonicCalibrator | None = None
    q_cal_all = q_raw_refit[cal_mask]
    y_cal_all = y_all[cal_mask].astype(float)
    cal_valid = ~np.isnan(q_cal_all) & np.isin(y_cal_all, [0, 1])
    if int(cal_valid.sum()) >= 10 and len(np.unique(y_cal_all[cal_valid])) == 2:
        global_calibrator = IsotonicCalibrator()
        global_calibrator.fit(q_cal_all[cal_valid], y_cal_all[cal_valid].astype(int))
        with open(pooled_dir / GLOBAL_CALIBRATOR_FILENAME, "wb") as fh:
            pickle.dump(global_calibrator, fh)

    # --- Per-expert calibration + fallback gate + metrics ---
    metrics: dict[str, Any] = {}
    expert_state: dict[str, dict[str, Any]] = {}
    expert_col = long_df["expert_key"].to_numpy()
    for expert_key, (sub, target_col) in expert_frames.items():
        e_mask = expert_col == expert_key
        e_cal = e_mask & cal_mask
        e_test = e_mask & test_mask
        q_cal_e = q_raw_refit[e_cal]
        y_cal_e = y_all[e_cal].astype(float)
        q_test_e = q_raw_refit[e_test]
        y_test_e = y_all[e_test].astype(float)

        pooled_test_metrics = _binary_metrics(y_test_e, q_test_e)
        pooled_auc = pooled_test_metrics.get("roc_auc")

        calibrator, calibrator_kind, notes = _fit_expert_calibrator(
            q_cal_e, y_cal_e, q_test_e, y_test_e, global_calibrator
        )
        for note in notes:
            print(f"    {expert_key}: {note}")

        # Spec correction #5: dedicated-head safety net for data-rich experts.
        n_rows_e = int(e_mask.sum())
        fallback_used = False
        dedicated_auc: float | None = None
        dedicated: dict[str, Any] | None = None
        if n_rows_e >= DATA_RICH_MIN_ROWS:
            dedicated = _train_dedicated_head(
                sub, expert_key, target_col, train_ids, cal_ids, test_ids,
                seed=seed, n_jobs=n_jobs,
            )
            if dedicated is not None:
                dedicated_auc = dedicated["test_auc"]
                if _should_fallback(dedicated_auc, pooled_auc):
                    fallback_used = True
                    print(
                        f"    {expert_key}: FALLBACK to dedicated head "
                        f"(dedicated={dedicated_auc}, pooled={pooled_auc})"
                    )
                    calibrator, calibrator_kind, fb_notes = _fit_expert_calibrator(
                        dedicated["q_cal"], dedicated["y_cal"].astype(float),
                        dedicated["q_test"], dedicated["y_test"].astype(float),
                        global_calibrator,
                    )
                    for note in fb_notes:
                        print(f"    {expert_key} (fallback): {note}")

        if fallback_used and dedicated is not None:
            deployed_raw_test = np.asarray(dedicated["q_test"], dtype=float)
            deployed_y_test = np.asarray(dedicated["y_test"], dtype=float)
        else:
            deployed_raw_test = q_test_e
            deployed_y_test = y_test_e
        raw_metrics = _binary_metrics(deployed_y_test, deployed_raw_test)
        if calibrator is not None and len(deployed_raw_test) > 0:
            calibrated_test = np.asarray(calibrator.transform(deployed_raw_test), dtype=float)
        else:
            calibrated_test = deployed_raw_test
        cal_metrics = _binary_metrics(deployed_y_test, calibrated_test)
        # Cal-SET raw AUC of the deployed head (LSST has no locked test rows
        # in the v12 split, so this is the only held-out trust read there).
        if fallback_used and dedicated is not None:
            cal_set_metrics = _binary_metrics(
                np.asarray(dedicated["y_cal"], dtype=float),
                np.asarray(dedicated["q_cal"], dtype=float),
            )
        else:
            cal_set_metrics = _binary_metrics(y_cal_e, q_cal_e)

        weak_entry = weak_ledger.get(expert_key, {})
        metrics[expert_key] = {
            "raw_auc": raw_metrics.get("roc_auc"),
            "cal_auc": cal_metrics.get("roc_auc"),
            "brier": cal_metrics.get("brier"),
            "ece": cal_metrics.get("ece"),
            "n_test": int(e_test.sum()),
            "calibrator_kind": calibrator_kind,
            "fallback_used": bool(fallback_used),
            "target_col": target_col,
            "n_train_rows": int((e_mask & train_mask).sum()),
            "n_cal_rows": int(e_cal.sum()),
            "cal_set_raw_auc": cal_set_metrics.get("roc_auc"),
            "pooled_raw_auc": pooled_auc,
            "dedicated_auc": dedicated_auc,
            "is_sn_filter": bool(expert_key in sn_filter),
            "weak_policy": weak_policy,
            "n_weak_rows_kept": int(weak_entry.get("kept", 0)),
            "n_weak_rows_dropped": int(weak_entry.get("dropped", 0)),
            "n_weak_rows_kept_lsst": int(weak_entry.get("kept_lsst", 0)),
            "n_weak_rows_dropped_lsst": int(weak_entry.get("dropped_lsst", 0)),
        }
        expert_state[expert_key] = {
            "calibrator": calibrator,
            "calibrator_kind": calibrator_kind,
            "fallback_used": fallback_used,
            "dedicated": dedicated if fallback_used else None,
            "target_col": target_col,
        }

    overall_test_metrics = _binary_metrics(
        y_all[test_mask].astype(float), q_raw_refit[test_mask]
    )
    metrics["_pooled"] = {
        "params": params,
        "n_estimators": int(n_estimators),
        "n_features": len(feature_cols),
        "n_train_rows": int(train_mask.sum()),
        "n_cal_rows": int(cal_mask.sum()),
        "n_test_rows": int(test_mask.sum()),
        "raw_auc_all_experts": overall_test_metrics.get("roc_auc"),
        "has_global_calibrator": global_calibrator is not None,
        # fusion v13 provenance
        "weak_policy": weak_policy,
        "weak_ledger": weak_ledger,
        "sn_filter_experts": sorted(sn_filter),
        "q_prior_oof": bool(q_prior_oof),
        "q_prior_n_folds": len(prior_fold_bundles),
        "emit_headless_q": bool(emit_headless_q),
        "q_prior_experts": q_prior_experts,
    }

    # --- Persist artifacts: pooled/ + per-expert dirs ---
    with open(pooled_dir / "model.pkl", "wb") as fh:
        pickle.dump(pooled_bundle, fh)
    pooled_metadata = {
        "kind": "pooled_trust_fusion_v8",
        "feature_cols": feature_cols,
        "categorical_col": EXPERT_ID_COL,
        "expert_levels": expert_levels,
        "generic_cols": generic_final,
        "own_pred_slots": list(OWN_PRED_SLOTS),
        "own_traj_slots": OWN_TRAJ_SLOTS,
        "meta_cols": META_NUMERIC_COLS,
        "expert_train_rows": expert_train_rows,
        "default_exact_codes": default_exact_codes,
        "params": params,
        "n_estimators": int(n_estimators),
        "seed": int(seed),
        "experts": sorted(expert_frames),
        "train_ids": sorted(train_ids),
        "cal_ids": sorted(cal_ids),
        "test_ids": sorted(test_ids),
        # fusion v13: the SN-filter set defines the is_sn_filter feature —
        # inference MUST reproduce it (PooledTrustView falls back to
        # LEGACY_SN_FILTER_EXPERTS when this key is absent, i.e. v8-v12).
        "sn_filter_experts": sorted(sn_filter),
        "weak_policy": weak_policy,
        "weak_ledger": weak_ledger,
        "q_prior_oof": bool(q_prior_oof),
        "emit_headless_q": bool(emit_headless_q),
        # Scoring contract (score_fusion_v8.attach_trust_columns): with
        # prior_mode_version >= 2 the scorer reproduces the training-time
        # prior-mode features and emits q_prior__ for exactly this set.
        "q_prior_experts": q_prior_experts,
        "prior_mode_version": PRIOR_MODE_VERSION,
    }
    with open(pooled_dir / "metadata.json", "w") as fh:
        json.dump(pooled_metadata, fh, indent=2)

    for expert_key, state in expert_state.items():
        san = sanitize_expert_key(expert_key)
        expert_dir = out_dir_path / san
        expert_dir.mkdir(parents=True, exist_ok=True)
        calibrator = state["calibrator"]
        if state["calibrator_kind"] in {"isotonic", "platt"} and calibrator is not None:
            with open(expert_dir / "calibrator.pkl", "wb") as fh:
                pickle.dump(calibrator, fh)
        if state["fallback_used"] and state["dedicated"] is not None:
            with open(expert_dir / "model.pkl", "wb") as fh:
                pickle.dump(state["dedicated"]["bundle"], fh)
        expert_metadata = {
            "expert_key": expert_key,
            "target_col": state["target_col"],
            "calibrator_kind": state["calibrator_kind"],
            "fallback_used": bool(state["fallback_used"]),
            "pooled_subdir": f"../{POOLED_SUBDIR}",
            "feature_cols": (
                state["dedicated"]["feature_cols"] if state["fallback_used"] else None
            ),
            "metrics": metrics[expert_key],
        }
        with open(expert_dir / "metadata.json", "w") as fh:
            json.dump(expert_metadata, fh, indent=2)

    # --- Emission into the snapshot frame ---
    emission_ledger: dict[str, Any] = {}
    output_snapshots = _emit_into_snapshots(
        snapshots,
        pooled_bundle=pooled_bundle,
        pooled_oof_lookup=pooled_oof_lookup,
        expert_state=expert_state,
        global_calibrator=global_calibrator,
        feature_cols=feature_cols,
        generic_cols=generic_final,
        expert_levels=expert_levels,
        expert_train_rows=expert_train_rows,
        default_exact_codes=default_exact_codes,
        train_ids=train_ids,
        sn_filter_experts=sn_filter,
        prior_fold_bundles=prior_fold_bundles,
        prior_fold_of_object=prior_fold_of_object,
        emit_headless_q=emit_headless_q,
        q_prior_experts=q_prior_experts,
        ledger=emission_ledger,
    )
    metrics["_pooled"]["emission"] = emission_ledger
    print(
        f"  pooled_trust: emitted q__ for {len(emission_ledger.get('q_experts', []))} experts, "
        f"q_prior__ for {len(emission_ledger.get('q_prior_experts', []))}; "
        f"q_prior OOF on {emission_ledger.get('n_train_rows_prior_oof', 0):,} of "
        f"{emission_ledger.get('n_train_rows', 0):,} train snapshot rows"
    )

    return PooledTrustResult(
        snapshots=output_snapshots,
        metrics=metrics,
        artifact_dir=str(out_dir_path),
    )


def _emit_into_snapshots(
    snapshots: pd.DataFrame,
    *,
    pooled_bundle: Any,
    pooled_oof_lookup: dict[str, dict[tuple[str, int, float], float]],
    expert_state: dict[str, dict[str, Any]],
    global_calibrator: IsotonicCalibrator | None,
    feature_cols: list[str],
    generic_cols: list[str],
    expert_levels: list[str],
    expert_train_rows: dict[str, int],
    default_exact_codes: dict[str, float | None],
    train_ids: set[str],
    sn_filter_experts=None,
    prior_fold_bundles: list[Any] | None = None,
    prior_fold_of_object: dict[str, int] | None = None,
    emit_headless_q: bool = False,
    q_prior_experts: str = "all",
    ledger: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """q__/trust_source__ for experts present in the snapshot AND with a
    trained head (``emit_headless_q=True`` restores the legacy "every expert
    with an avail__ column" rule), q_prior__ for ALL registered experts
    (``q_prior_experts="trained"`` restricts it likewise).  NaN-absent rule:
    absent expert -> q NaN.

    q_prior on train objects: when ``prior_fold_bundles`` /
    ``prior_fold_of_object`` are given, every train object's prior-mode row is
    scored by the GroupKFold fold model that held that object out (fusion
    v13 §2); train objects that contributed no labelled Stage-A row (hence
    were never fitted on) and all cal/test rows use the refit-on-train model.
    ``ledger`` (optional dict) receives emission counts.
    """
    sn_filter = _resolve_sn_filter_experts(sn_filter_experts)
    prior_fold_bundles = list(prior_fold_bundles or [])
    prior_fold_of_object = dict(prior_fold_of_object or {})
    snap_keys = _normalized_keys(snapshots)
    snap_object_ids = snapshots["object_id"].astype(str)
    snap_train = snap_object_ids.isin(train_ids).to_numpy()
    new_cols: dict[str, Any] = {}

    # Held-out fold of every train object (NaN -> no fold model held it out).
    prior_fold_masks: list[np.ndarray] = []
    if prior_fold_bundles and prior_fold_of_object:
        fold_index = snap_object_ids.map(prior_fold_of_object).to_numpy(dtype=float)
        for fold in range(len(prior_fold_bundles)):
            prior_fold_masks.append((fold_index == fold) & snap_train)
    n_prior_oof_rows = int(sum(int(m.sum()) for m in prior_fold_masks))
    q_experts_emitted: list[str] = []
    q_prior_experts_emitted: list[str] = []

    for expert_key in expert_levels:
        san = sanitize_expert_key(expert_key)
        state = expert_state.get(expert_key)
        calibrator = state["calibrator"] if state else None
        fallback_used = bool(state and state["fallback_used"])
        log_rows = np.log1p(float(expert_train_rows.get(expert_key, 0)))
        default_code = default_exact_codes.get(expert_key)
        has_head = state is not None

        # ---- q__ + trust_source__ (avail column present AND a trained head,
        #      matching what score_fusion_v8.attach_trust_columns produces) ----
        avail_col = f"avail__{san}"
        if avail_col in snapshots.columns and (has_head or emit_headless_q):
            q_experts_emitted.append(expert_key)
            avail = (
                pd.to_numeric(snapshots[avail_col], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=float)
                > 0
            )
            if fallback_used and state is not None and state["dedicated"] is not None:
                dedicated = state["dedicated"]
                X_ded, _ = _prepare_numeric_frame(
                    snapshots, dedicated["feature_cols"], fill_values=dedicated["fill_values"]
                )
                raw = _predict_binary_classifier(dedicated["bundle"], X_ded)
                oof_lookup = dedicated["oof_lookup"]
            else:
                # default_exact_code as PooledTrustView.predict_trust_raw does,
                # so cal/test q__ written here equal the scorer's (parity).
                feats = _assemble_features_for_expert(
                    snapshots, expert_key, generic_cols=generic_cols, log_rows=log_rows,
                    default_exact_code=default_code, sn_filter_experts=sn_filter,
                )
                X_e = _prepare_pooled_matrix(feats, feature_cols, expert_levels)
                raw = _predict_binary_classifier(pooled_bundle, X_e)
                oof_lookup = pooled_oof_lookup.get(expert_key, {})

            raw = np.asarray(raw, dtype=float)
            matched = np.zeros(len(snapshots), dtype=bool)
            if oof_lookup:
                for i, key in enumerate(snap_keys):
                    oof_value = oof_lookup.get(key)
                    if oof_value is not None and np.isfinite(oof_value):
                        raw[i] = oof_value
                        matched[i] = True
            if calibrator is not None:
                q_values = np.asarray(calibrator.transform(raw), dtype=float)
            else:
                q_values = raw
            q_values = q_values.copy()
            q_values[~avail] = np.nan
            trust_source = np.where(
                ~avail, "unavailable", np.where(matched & snap_train, "oof", "train_model")
            )
            new_cols[f"q__{san}"] = q_values
            new_cols[f"trust_source__{san}"] = trust_source

        # ---- q_prior__ for ALL registered experts (correction #4c) ----
        if q_prior_experts == "trained" and not has_head:
            continue
        q_prior_experts_emitted.append(expert_key)
        feats_prior = _assemble_features_for_expert(
            snapshots,
            expert_key,
            generic_cols=generic_cols,
            log_rows=log_rows,
            default_exact_code=default_code,
            prior_mode=True,
            sn_filter_experts=sn_filter,
        )
        X_prior = _prepare_pooled_matrix(feats_prior, feature_cols, expert_levels)
        raw_prior = np.asarray(
            _predict_binary_classifier(pooled_bundle, X_prior), dtype=float
        ).copy()
        # fusion v13 §2: train objects are scored by the fold model that held
        # them out — the refit model saw their labels (via EVERY expert's row
        # for that object), so its readout is in-sample on train rows.
        for fold_mask, fold_bundle in zip(prior_fold_masks, prior_fold_bundles):
            if fold_mask.any():
                raw_prior[fold_mask] = _predict_binary_classifier(
                    fold_bundle, X_prior[fold_mask]
                )
        # q_prior is a POOLED-model readout by definition: when the expert
        # fell back to a dedicated head, its calibrator was fit on dedicated
        # outputs, so use the global calibrator (or raw) instead.
        if not fallback_used and state is not None and calibrator is not None:
            prior_values = np.asarray(calibrator.transform(raw_prior), dtype=float)
        elif global_calibrator is not None:
            prior_values = np.asarray(global_calibrator.transform(raw_prior), dtype=float)
        else:
            prior_values = raw_prior
        new_cols[f"q_prior__{san}"] = prior_values

    if ledger is not None:
        ledger.update(
            {
                "q_experts": q_experts_emitted,
                "q_prior_experts": q_prior_experts_emitted,
                "headless_experts_with_avail": sorted(
                    key for key in expert_levels
                    if key not in expert_state
                    and f"avail__{sanitize_expert_key(key)}" in snapshots.columns
                ),
                "n_train_rows": int(snap_train.sum()),
                "n_train_rows_prior_oof": n_prior_oof_rows,
                "q_prior_oof_folds": len(prior_fold_bundles),
            }
        )

    collision = [c for c in new_cols if c in snapshots.columns]
    base = snapshots.drop(columns=collision) if collision else snapshots
    return pd.concat(
        [base.copy(), pd.DataFrame(new_cols, index=snapshots.index)], axis=1
    )


# --------------------------------------------------------------------------
# Per-expert facade (pinned interface)
# --------------------------------------------------------------------------


class PooledTrustView:
    """Per-expert ``ExpertTrustArtifact``-compatible facade over the pooled
    model: ``PooledTrustView.load("models/trust_fusion_v8/<san>")`` then
    ``view.predict_trust(snapshot_rows)`` exactly like
    ``ExpertTrustArtifact.load(...).predict_trust(...)`` today.

    When the spec-correction-#5 fallback fired for this expert, the view
    transparently serves the dedicated per-expert LightGBM head instead.
    """

    def __init__(
        self,
        *,
        expert_key: str,
        calibrator: Any,
        calibrator_kind: str,
        fallback_used: bool,
        pooled_bundle: Any = None,
        pooled_metadata: dict[str, Any] | None = None,
        dedicated_bundle: Any = None,
        dedicated_feature_cols: list[str] | None = None,
        global_calibrator: Any = None,
        has_head: bool = True,
    ) -> None:
        self.expert_key = expert_key
        self.calibrator = calibrator
        self.calibrator_kind = calibrator_kind
        self.fallback_used = bool(fallback_used)
        self._pooled_bundle = pooled_bundle
        self._pooled_metadata = pooled_metadata or {}
        self._dedicated_bundle = dedicated_bundle
        self._dedicated_feature_cols = dedicated_feature_cols or []
        self._global_calibrator = global_calibrator
        # False for a headless view (registered expert without a per-expert
        # dir): only q_prior__ is defined for it.
        self.has_head = bool(has_head)
        # The SN-filter set that defined the is_sn_filter feature at training
        # time.  Artifacts trained before fusion v13 did not persist it: they
        # were all trained with the legacy 4-expert set, never the module's
        # current one (which v13 extended with the ALeRCE stamp family).
        self.sn_filter_experts = self.sn_filter_experts_from_metadata(self._pooled_metadata)

    @staticmethod
    def sn_filter_experts_from_metadata(pooled_metadata: dict[str, Any] | None) -> frozenset[str]:
        """Persisted ``sn_filter_experts`` or the legacy v8-v12 set."""
        persisted = (pooled_metadata or {}).get("sn_filter_experts")
        if persisted is None:
            return frozenset(LEGACY_SN_FILTER_EXPERTS)
        return frozenset(str(key) for key in persisted)

    @staticmethod
    def load_pooled_assets(pooled_dir: str | Path) -> dict[str, Any]:
        """``{bundle, metadata, global_calibrator}`` from a ``pooled/`` dir
        (each None/{} when absent).  Load once and share across views."""
        pooled_path = Path(pooled_dir)
        assets: dict[str, Any] = {"bundle": None, "metadata": {}, "global_calibrator": None}
        if (pooled_path / "model.pkl").exists():
            with open(pooled_path / "model.pkl", "rb") as fh:
                assets["bundle"] = pickle.load(fh)
            with open(pooled_path / "metadata.json") as fh:
                assets["metadata"] = json.load(fh)
        global_path = pooled_path / GLOBAL_CALIBRATOR_FILENAME
        if global_path.exists():
            with open(global_path, "rb") as fh:
                assets["global_calibrator"] = pickle.load(fh)
        return assets

    @classmethod
    def load(cls, expert_dir: str, pooled_assets: dict[str, Any] | None = None) -> "PooledTrustView":
        expert_path = Path(expert_dir)
        with open(expert_path / "metadata.json") as fh:
            metadata = json.load(fh)
        expert_key = str(metadata["expert_key"])
        calibrator_kind = str(metadata.get("calibrator_kind", "none"))
        fallback_used = bool(metadata.get("fallback_used", False))
        pooled_path = (expert_path / metadata.get("pooled_subdir", f"../{POOLED_SUBDIR}")).resolve()

        assets = pooled_assets if pooled_assets is not None else cls.load_pooled_assets(pooled_path)
        pooled_bundle = assets.get("bundle")
        pooled_metadata: dict[str, Any] = dict(assets.get("metadata") or {})
        global_calibrator = assets.get("global_calibrator")

        calibrator = None
        if calibrator_kind in {"isotonic", "platt"}:
            calibrator_path = expert_path / "calibrator.pkl"
            if calibrator_path.exists():
                with open(calibrator_path, "rb") as fh:
                    calibrator = pickle.load(fh)
        elif calibrator_kind == "global":
            calibrator = global_calibrator

        dedicated_bundle = None
        dedicated_feature_cols: list[str] | None = None
        if fallback_used:
            with open(expert_path / "model.pkl", "rb") as fh:
                dedicated_bundle = pickle.load(fh)
            dedicated_feature_cols = [str(c) for c in (metadata.get("feature_cols") or [])]

        return cls(
            expert_key=expert_key,
            calibrator=calibrator,
            calibrator_kind=calibrator_kind,
            fallback_used=fallback_used,
            pooled_bundle=pooled_bundle,
            pooled_metadata=pooled_metadata,
            dedicated_bundle=dedicated_bundle,
            dedicated_feature_cols=dedicated_feature_cols,
            global_calibrator=global_calibrator,
            has_head=True,
        )

    @classmethod
    def headless(cls, expert_key: str, pooled_assets: dict[str, Any]) -> "PooledTrustView":
        """View for a registered expert WITHOUT a per-expert dir: serves only
        ``predict_prior`` (pooled model in prior mode, global calibrator) —
        what training emits as ``q_prior__`` for such experts."""
        return cls(
            expert_key=expert_key,
            calibrator=None,
            calibrator_kind="none",
            fallback_used=False,
            pooled_bundle=pooled_assets.get("bundle"),
            pooled_metadata=dict(pooled_assets.get("metadata") or {}),
            global_calibrator=pooled_assets.get("global_calibrator"),
            has_head=False,
        )

    @property
    def prior_mode_version(self) -> int:
        """Persisted q_prior protocol (1 = legacy v8-v12 artifact)."""
        return int(self._pooled_metadata.get("prior_mode_version", 1) or 1)

    @property
    def prior_calibrator(self) -> Any:
        """The calibrator training applied to q_prior__: the expert's own
        (pooled-model) calibrator, else the global one, else none.  A
        dedicated-head fallback's calibrator was fit on dedicated outputs, so
        the pooled prior readout uses the global calibrator instead."""
        if self.has_head and not self.fallback_used and self.calibrator is not None:
            return self.calibrator
        return self._global_calibrator

    def _pooled_features(self, df: pd.DataFrame, *, prior_mode: bool) -> pd.DataFrame:
        meta = self._pooled_metadata
        log_rows = np.log1p(float(meta.get("expert_train_rows", {}).get(self.expert_key, 0)))
        default_code = meta.get("default_exact_codes", {}).get(self.expert_key)
        feats = _assemble_features_for_expert(
            df,
            self.expert_key,
            generic_cols=[str(c) for c in meta.get("generic_cols", [])],
            log_rows=log_rows,
            default_exact_code=default_code,
            prior_mode=prior_mode,
            sn_filter_experts=self.sn_filter_experts,
        )
        return _prepare_pooled_matrix(
            feats,
            [str(c) for c in meta["feature_cols"]],
            [str(level) for level in meta["expert_levels"]],
        )

    def predict_prior_raw(self, df: pd.DataFrame) -> np.ndarray:
        """Uncalibrated q_prior: the pooled model on the training-time
        prior-mode features (own pred + own traj slots NaN'd, modal exactness
        code) — identical to ``_emit_into_snapshots`` on non-train rows."""
        if self._pooled_bundle is None:
            raise RuntimeError(
                f"PooledTrustView({self.expert_key}): pooled model not loaded"
            )
        X = self._pooled_features(df, prior_mode=True)
        return np.asarray(_predict_binary_classifier(self._pooled_bundle, X), dtype=float)

    def predict_prior(self, df: pd.DataFrame) -> np.ndarray:
        """Calibrated q_prior with training's calibrator rule (prior_calibrator)."""
        raw = self.predict_prior_raw(df)
        calibrator = self.prior_calibrator
        if calibrator is None:
            return raw
        return np.asarray(calibrator.transform(raw), dtype=float)

    def predict_trust(self, df: pd.DataFrame) -> np.ndarray:
        """Calibrated trust probability for this expert's claim on each row."""
        raw = self.predict_trust_raw(df)
        if self.calibrator is None:
            return raw
        return np.asarray(self.calibrator.transform(raw), dtype=float)

    def predict_trust_raw(self, df: pd.DataFrame) -> np.ndarray:
        """Uncalibrated trust probability (diagnostics / stacking features)."""
        if self.fallback_used and self._dedicated_bundle is not None:
            X, _ = _prepare_numeric_frame(df, self._dedicated_feature_cols)
            return np.asarray(
                _predict_binary_classifier(self._dedicated_bundle, X), dtype=float
            )
        if self._pooled_bundle is None:
            raise RuntimeError(
                f"PooledTrustView({self.expert_key}): pooled model not loaded"
            )
        X = self._pooled_features(df, prior_mode=False)
        return np.asarray(_predict_binary_classifier(self._pooled_bundle, X), dtype=float)
