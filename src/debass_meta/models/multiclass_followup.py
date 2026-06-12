"""Multiclass (ternary) follow-up head for fusion_v8.

Replaces the binary ``target_follow_proxy`` head with a 3-class LightGBM
softmax over ``CLASSES = ("snia", "nonIa_snlike", "other")`` plus a
Dirichlet calibration layer (fallback ladder: Dirichlet -> vector
temperature -> temperature -> none, guarded by cal-split grouped-CV
log-loss).

No-leakage argument
-------------------
* Every feature is computed from the lightcurve truncated at ``n_det``
  detections, or from broker events with ``event_time_jd <= alert_jd``
  (built upstream); this module never recomputes features.
* The model is fit on TRAIN rows only.  The early-stopping fold is carved
  from TRAIN by grouped split (object level) — the CAL split is never
  touched by model fitting; it is reserved for calibration/conformal.
* Provenance masking (spec correction #1) is applied to TRAIN rows before
  fitting: rows whose label derives from an ALeRCE self-label get every
  ``alerce*`` expert block force-NaN'd; rows whose label derives from
  catalog context (sherlock/babamul/host-context/Gaia/MPC/SIMBAD) get the
  ``lasair/sherlock`` and ``babamul`` blocks force-NaN'd.  This is a
  structural anti-circularity guard, not a downweighting.
* Feature discovery blocks every label-derived / truth-catalog column
  (see ``_BLOCKED_COLS`` / ``_BLOCKED_PREFIXES``) and asserts loudly.

Design choice (documented per spec): ``q_prior__<expert>`` columns ARE
legitimate features and are INCLUDED by feature auto-discovery.  They are
produced by the pooled trust model from lightcurve context only (the six
own-prediction slots NaN'd), so they carry no expert output and no label
information; they are available identically at train and score time.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from debass_meta.projectors import ALL_EXPERT_KEYS, sanitize_expert_key

from .calibrate import TemperatureScaler

CLASSES = ("snia", "nonIa_snlike", "other")
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}

# Label-quality -> base weight name (resolved against the train kwargs).
_SPEC_QUALITIES = ("spectroscopic", "tns_untyped")

# ---------------------------------------------------------------------------
# Feature discovery (followup.py blocklist pattern, extended per fusion_v8 spec)
# ---------------------------------------------------------------------------

_BLOCKED_COLS = {
    # identifiers / labels (followup.py blocklist)
    "object_id",
    "target_follow_proxy",
    "target_class",
    "target_label",
    "label_source",
    "label_quality",
    # absolute timestamps — won't generalize to future observations
    "alert_jd",
    # association metadata — not a classification signal
    "lightcurve_association_sep_arcsec",
    # DP1 metadata / truth-catalog columns (spec correction #2 contract)
    "diaObjectId",
    "is_published_sn",
    "is_gaia_known_star",
    "is_known_variable",
    "simbad_main_type",
    "gaia_var_class",
    # bookkeeping columns this module may create
    "is_aug",
    "is_bts_expansion",
    "sample_weight",
    "base_weight",
    "w_qual",
    "w_cls",
    "w_bts",
    "split",
    # scoring-output columns (defensive: never retrain on a scored parquet)
    "p_snia",
    "p_nonia",
    "p_nonIa",
    "p_other",
    "p_follow_proxy",
    "ensemble_p_snia",
    "set_snia",
    "set_nonia",
    "set_other",
    "set_size",
    "priority_score",
    "rank",
    "selected",
    "eligible",
    "demoted",
}

_BLOCKED_PREFIXES = (
    "source_event_time_jd__",
    "trust_source__",
    "published_sn_",
    "dp1_",
    "target_",
    "label_",
)


def _numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    """Auto-discover Stage-B feature columns.

    Mirrors ``followup.py:_numeric_feature_cols`` (blocklist + all-NaN /
    constant pruning) with the fusion_v8 extensions: DP1 truth-catalog
    columns, label columns, and scoring-output columns are blocked.
    ``q__*`` and ``q_prior__*`` columns are deliberately INCLUDED — q is the
    Stage-A trust output (OOF on train rows, so no stacking leak) and
    q_prior is computed from lightcurve context only (see module docstring).
    """
    cols: list[str] = []
    for column in df.columns:
        if column in _BLOCKED_COLS:
            continue
        if column.startswith(_BLOCKED_PREFIXES):
            continue
        if not is_numeric_dtype(df[column]):
            continue
        # Drop columns that are all-NaN or constant — zero signal
        if df[column].isna().all():
            continue
        if df[column].nunique(dropna=True) <= 1:
            continue
        cols.append(column)
    return sorted(cols)


def _assert_no_label_features(feature_cols: Sequence[str]) -> None:
    """Loud failure: no label-derived column may enter Stage-B X."""
    bad = [
        c for c in feature_cols
        if c in _BLOCKED_COLS or c.startswith(_BLOCKED_PREFIXES)
    ]
    assert not bad, f"label-derived columns leaked into Stage-B features: {bad}"


def _prepare_frame(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    frame = df.copy()
    for column in feature_cols:
        if column not in frame.columns:
            frame[column] = np.nan
    numeric = frame[feature_cols].apply(pd.to_numeric, errors="coerce")
    # LightGBM handles NaN natively — no imputation needed.
    return numeric


# ---------------------------------------------------------------------------
# Provenance masking (spec correction #1) — reusable by the gold builder
# ---------------------------------------------------------------------------

# Experts whose feature blocks are circular with ALeRCE-derived labels.
# Precedent: expert_trust.py honesty filter treats `alerce/*` AND the local
# `alerce_lc` rerun as the same broker family.
_ALERCE_FAMILY = tuple(
    key for key in ALL_EXPERT_KEYS
    if key.startswith("alerce/") or key == "alerce_lc"
)
# Experts whose feature blocks are circular with catalog-context labels.
_CONTEXT_FAMILY = ("lasair/sherlock", "babamul")

# label_source substrings that mark a context/catalog-derived label.
_CONTEXT_SOURCE_TOKENS = (
    "sherlock",
    "babamul",
    "host_context",
    "gaia_stellar",
    "gaia_parallax",
    "mpc_exclusion",
    "simbad",
)


def _mask_expert_blocks(
    df: pd.DataFrame, row_mask: pd.Series, expert_keys: Sequence[str]
) -> None:
    """Force-NaN ``proj__``/``q__`` and zero ``avail__``/``exact__`` columns
    of the given experts on the masked rows (in place).

    ``avail__``/``exact__`` are set to 0.0 rather than NaN so masked rows are
    indistinguishable from rows where the expert genuinely never fired
    (the §1.6 missingness contract: avail carries absence, proj/q carry the
    claim).  A NaN avail would create a third state that flags "this row was
    provenance-masked", which itself correlates with the label tier.
    """
    if not row_mask.any():
        return
    for expert_key in expert_keys:
        san = sanitize_expert_key(expert_key)
        nan_cols = [c for c in df.columns if c.startswith(f"proj__{san}__")]
        if f"q__{san}" in df.columns:
            nan_cols.append(f"q__{san}")
        zero_cols = [
            c for c in (f"avail__{san}", f"exact__{san}") if c in df.columns
        ]
        if nan_cols:
            df.loc[row_mask, nan_cols] = np.nan
        if zero_cols:
            df.loc[row_mask, zero_cols] = 0.0


def apply_provenance_masking(
    df: pd.DataFrame,
    *,
    label_source_col: str = "label_source",
    label_quality_col: str = "label_quality",
) -> pd.DataFrame:
    """Structural anti-circularity masking (fusion_v8 spec correction #1).

    Returns a masked COPY of ``df``:

    * rows whose ``label_source`` contains ``"alerce"`` (e.g.
      ``alerce_self_label``, ``alerce_stamp_lsst (SN)``): all ``alerce*``
      expert blocks (``alerce/*`` + local ``alerce_lc``) are masked;
    * rows whose label derives from catalog context — ``label_quality ==
      "context"`` or ``label_source`` matching sherlock / babamul /
      host-context / Gaia / MPC / SIMBAD tokens: the ``lasair/sherlock``
      and ``babamul`` blocks are masked.

    Masking = ``proj__*``/``q__*`` -> NaN, ``avail__*``/``exact__*`` -> 0.0
    (see ``_mask_expert_blocks``).  Labels and weights are never changed.
    Mask counts are recorded in ``out.attrs["provenance_masking"]``.
    """
    out = df.copy()
    n = len(out)
    if label_source_col in out.columns:
        source = out[label_source_col].astype(str).str.lower()
    else:
        source = pd.Series([""] * n, index=out.index, dtype=str)
    if label_quality_col in out.columns:
        quality = out[label_quality_col].astype(str).str.lower()
    else:
        quality = pd.Series([""] * n, index=out.index, dtype=str)

    alerce_mask = source.str.contains("alerce", na=False)
    context_mask = quality.eq("context") | source.str.contains(
        "|".join(_CONTEXT_SOURCE_TOKENS), na=False, regex=True
    )

    _mask_expert_blocks(out, alerce_mask, _ALERCE_FAMILY)
    _mask_expert_blocks(out, context_mask, _CONTEXT_FAMILY)

    out.attrs["provenance_masking"] = {
        "n_rows": int(n),
        "n_alerce_masked": int(alerce_mask.sum()),
        "n_context_masked": int(context_mask.sum()),
    }
    return out


# ---------------------------------------------------------------------------
# Sample weighting:  w = w_qual × 1/n_rows(object) × w_bts   (× w_cls later)
# ---------------------------------------------------------------------------

def compute_base_weights(
    df: pd.DataFrame,
    *,
    weak_weight: float = 0.1,
    context_weight: float = 0.15,
    tns_untyped_weight: float = 0.6,
    bts_weight: float = 1.0,
) -> np.ndarray:
    """Per-row base weight ``w_qual × 1/n_rows(object) × w_bts``.

    * ``w_qual``: spectroscopic 1.0 / tns_untyped / context / weak per kwargs
      (missing or unknown quality is treated conservatively as ``weak``).
    * ``1/n_rows(object)``: each object contributes total weight w_qual×w_bts
      across its epoch rows — with uniform quality every object's rows sum to
      exactly the same total, killing the late-easy-epoch pooling bias.
    * ``w_bts``: applied to BTS-expansion rows, identified by an
      ``is_bts_expansion`` column when present, else by ``label_source``
      containing ``"bts"``.

    Class weighting is intentionally NOT applied here (see
    ``_class_weight_vector``); tests check object totals pre-class-weighting.
    """
    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=float)

    qual_map = {
        "spectroscopic": 1.0,
        "tns_untyped": float(tns_untyped_weight),
        "context": float(context_weight),
        "weak": float(weak_weight),
    }
    if "label_quality" in df.columns:
        quality = df["label_quality"].astype(str).str.lower()
        w_qual = quality.map(qual_map).fillna(float(weak_weight)).to_numpy(float)
    else:
        w_qual = np.ones(n, dtype=float)

    counts = df.groupby(df["object_id"].astype(str))["object_id"].transform("size")
    w_rows = 1.0 / counts.to_numpy(float)

    w_bts = np.ones(n, dtype=float)
    if bts_weight != 1.0:
        if "is_bts_expansion" in df.columns:
            bts_mask = (
                df["is_bts_expansion"].fillna(0).astype(float).to_numpy() > 0.5
            )
        elif "label_source" in df.columns:
            bts_mask = (
                df["label_source"].astype(str).str.lower()
                .str.contains("bts", na=False).to_numpy()
            )
        else:
            bts_mask = np.zeros(n, dtype=bool)
        w_bts = np.where(bts_mask, float(bts_weight), 1.0)

    return w_qual * w_rows * w_bts


def _class_weight_vector(y: np.ndarray, base_weights: np.ndarray) -> np.ndarray:
    """Inverse effective-class-frequency weights on the weighted counts:
    ``w_cls(c) = W_total / (3 · W_c)`` — the 3-class replacement for
    ``is_unbalance``."""
    out = np.ones(len(CLASSES), dtype=float)
    total = float(base_weights.sum())
    for c in range(len(CLASSES)):
        w_c = float(base_weights[y == c].sum())
        out[c] = total / (len(CLASSES) * w_c) if w_c > 0 else 0.0
    return out


# ---------------------------------------------------------------------------
# Expert-dropout row augmentation (spec correction #4a)
# ---------------------------------------------------------------------------

def expert_dropout_augment(
    df: pd.DataFrame,
    base_weights: np.ndarray,
    *,
    aug_frac: float = 0.25,
    aug_weight: float = 0.3,
    keep_one_frac: float = 0.5,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Duplicate ``aug_frac`` of rows with random expert blocks NaN'd out.

    Trains the DP1 1-expert (and 0-expert) deployment regime.  For each
    augmented copy either (a) keep-exactly-one mode: exactly one available
    expert survives, all others are dropped, or (b) subset mode: a random
    non-empty subset of available experts is dropped.  Dropping an expert =
    ``proj__<san>__*`` / ``q__<san>`` -> NaN, ``avail__<san>`` /
    ``exact__<san>`` -> 0.0 (the §1.6 absent-expert encoding).  Labels and
    every non-expert column are copied verbatim; augmented copies carry
    weight ``aug_weight × base_weight`` and ``is_aug = 1.0``.

    Only rows with >= 1 available expert are eligible (a row with no expert
    cannot be perturbed).  Returns ``(aug_df, aug_weights, info)``;
    ``info["source_positions"]`` holds the iloc positions of the source rows.
    """
    rng = np.random.default_rng(seed)
    base_weights = np.asarray(base_weights, dtype=float)
    assert len(base_weights) == len(df), "base_weights must align with df rows"

    sans = sorted(
        c[len("avail__"):] for c in df.columns if c.startswith("avail__")
    )
    empty_info = {
        "n_source_rows": int(len(df)),
        "n_eligible": 0,
        "n_aug": 0,
        "n_keep_one": 0,
        "source_positions": np.zeros(0, dtype=int),
    }
    if not sans or aug_frac <= 0.0 or len(df) == 0:
        return df.iloc[0:0].copy(), np.zeros(0, dtype=float), empty_info

    avail = (
        df[[f"avail__{s}" for s in sans]]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(float)
        > 0.5
    )
    n_avail = avail.sum(axis=1)
    eligible = np.flatnonzero(n_avail >= 1)
    empty_info["n_eligible"] = int(len(eligible))
    n_aug = int(round(float(aug_frac) * len(eligible)))
    if n_aug == 0:
        return df.iloc[0:0].copy(), np.zeros(0, dtype=float), empty_info

    pick = np.sort(rng.choice(eligible, size=n_aug, replace=False))
    avail_pick = avail[pick]
    n_avail_pick = avail_pick.sum(axis=1)

    # keep-exactly-one mode needs >= 2 available experts to change the row.
    keep_one = (rng.random(n_aug) < float(keep_one_frac)) & (n_avail_pick >= 2)
    drop = np.zeros_like(avail_pick, dtype=bool)
    for i in range(n_aug):
        avail_idx = np.flatnonzero(avail_pick[i])
        if keep_one[i]:
            keep_j = rng.choice(avail_idx)
            drop[i, avail_idx] = True
            drop[i, keep_j] = False
        else:
            subset = avail_idx[rng.random(len(avail_idx)) < 0.5]
            if len(subset) == 0:
                subset = np.array([rng.choice(avail_idx)])
            drop[i, subset] = True

    aug = df.iloc[pick].copy().reset_index(drop=True)
    for j, san in enumerate(sans):
        mask = drop[:, j]
        if not mask.any():
            continue
        nan_cols = [c for c in aug.columns if c.startswith(f"proj__{san}__")]
        if f"q__{san}" in aug.columns:
            nan_cols.append(f"q__{san}")
        if nan_cols:
            aug.loc[mask, nan_cols] = np.nan
        zero_cols = [
            c for c in (f"avail__{san}", f"exact__{san}") if c in aug.columns
        ]
        if zero_cols:
            aug.loc[mask, zero_cols] = 0.0

    aug["is_aug"] = 1.0
    aug_weights = base_weights[pick] * float(aug_weight)
    info = {
        "n_source_rows": int(len(df)),
        "n_eligible": int(len(eligible)),
        "n_aug": int(n_aug),
        "n_keep_one": int(keep_one.sum()),
        "source_positions": pick,
    }
    return aug, aug_weights, info


# ---------------------------------------------------------------------------
# Multiclass calibrators (Dirichlet -> vector temperature -> temperature)
# ---------------------------------------------------------------------------

def _weighted_log_loss(
    y: np.ndarray, proba: np.ndarray, sample_weight: np.ndarray | None = None
) -> float:
    proba = np.clip(np.asarray(proba, dtype=float), 1e-9, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)
    nll = -np.log(proba[np.arange(len(y)), np.asarray(y, dtype=int)])
    if sample_weight is None:
        return float(np.mean(nll))
    return float(np.average(nll, weights=np.asarray(sample_weight, dtype=float)))


class IdentityCalibrator:
    """No-op calibrator (the raw-probability fallback)."""

    name = "none"

    def fit(self, proba, y, sample_weight=None) -> "IdentityCalibrator":
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        return np.asarray(proba, dtype=float)


class DirichletCalibrator:
    """Dirichlet calibration (Kull et al. 2019): ``p̂ = softmax(W ln p + b)``.

    Implemented as multinomial logistic regression on ``ln p`` features
    (sklearn lbfgs is multinomial for >2 classes); ridge strength via ``C``.
    """

    name = "dirichlet"

    def __init__(self, C: float = 1.0, seed: int = 42) -> None:
        self.C = float(C)
        self.seed = int(seed)
        self._lr = None

    @staticmethod
    def _features(proba: np.ndarray) -> np.ndarray:
        return np.log(np.clip(np.asarray(proba, dtype=float), 1e-6, 1.0))

    def fit(self, proba, y, sample_weight=None) -> "DirichletCalibrator":
        from sklearn.linear_model import LogisticRegression

        self._lr = LogisticRegression(
            C=self.C, max_iter=2000, random_state=self.seed
        )
        self._lr.fit(self._features(proba), np.asarray(y, dtype=int),
                     sample_weight=sample_weight)
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        if self._lr is None:
            return np.asarray(proba, dtype=float)
        raw = self._lr.predict_proba(self._features(proba))
        out = np.zeros((len(raw), len(CLASSES)), dtype=float)
        for col, cls in enumerate(self._lr.classes_):
            out[:, int(cls)] = raw[:, col]
        return out


class VectorScaler:
    """Vector temperature scaling (3 params): ``p̂ = softmax(w ⊙ ln p)``."""

    name = "vector_temperature"

    def __init__(self) -> None:
        self.w_ = np.ones(len(CLASSES), dtype=float)

    def fit(self, proba, y, sample_weight=None) -> "VectorScaler":
        from scipy.optimize import minimize

        logp = np.log(np.clip(np.asarray(proba, dtype=float), 1e-8, 1.0))
        y = np.asarray(y, dtype=int)
        w = (
            np.ones(len(y), dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float)
        )

        def nll(vec: np.ndarray) -> float:
            z = logp * vec[None, :]
            z = z - z.max(axis=1, keepdims=True)
            p = np.exp(z)
            p = p / p.sum(axis=1, keepdims=True)
            return _weighted_log_loss(y, p, w)

        res = minimize(
            nll,
            np.ones(len(CLASSES), dtype=float),
            method="L-BFGS-B",
            bounds=[(0.05, 20.0)] * len(CLASSES),
        )
        self.w_ = np.asarray(res.x, dtype=float)
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        logp = np.log(np.clip(np.asarray(proba, dtype=float), 1e-8, 1.0))
        z = logp * self.w_[None, :]
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)


class _TemperatureAdapter:
    """Sample-weight-tolerant facade over the repo ``TemperatureScaler``."""

    name = "temperature"

    def __init__(self) -> None:
        self._ts = TemperatureScaler()

    def fit(self, proba, y, sample_weight=None) -> "_TemperatureAdapter":
        # TemperatureScaler is a 1-parameter grid search; weighting the loss
        # changes the optimum negligibly — keep the existing implementation.
        self._ts.fit(np.asarray(proba, dtype=float), np.asarray(y, dtype=int))
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        return self._ts.transform(np.asarray(proba, dtype=float))


def _make_calibrator(kind: str, *, C: float = 1.0, seed: int = 42):
    if kind == "dirichlet":
        return DirichletCalibrator(C=C, seed=seed)
    if kind == "vector_temperature":
        return VectorScaler()
    if kind == "temperature":
        return _TemperatureAdapter()
    return IdentityCalibrator()


def _grouped_cv_loss(
    kind: str,
    proba: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    groups: np.ndarray,
    *,
    C: float = 1.0,
    seed: int = 42,
    n_splits: int = 3,
) -> float:
    """Grouped-CV held-out weighted log-loss of a calibrator kind on cal."""
    from sklearn.model_selection import GroupKFold

    n_groups = len(np.unique(groups))
    splits = min(int(n_splits), n_groups)
    if splits < 2:
        return float("inf")
    losses: list[float] = []
    weights: list[float] = []
    for fit_idx, val_idx in GroupKFold(n_splits=splits).split(proba, y, groups):
        if kind == "identity":
            p_val = proba[val_idx]
        else:
            if len(np.unique(y[fit_idx])) < 2:
                return float("inf")
            try:
                cal = _make_calibrator(kind, C=C, seed=seed)
                cal.fit(proba[fit_idx], y[fit_idx],
                        sample_weight=sample_weight[fit_idx])
                p_val = cal.transform(proba[val_idx])
            except Exception:
                return float("inf")
        losses.append(_weighted_log_loss(y[val_idx], p_val,
                                         sample_weight[val_idx]))
        weights.append(float(sample_weight[val_idx].sum()))
    return float(np.average(losses, weights=weights))


def _fit_calibrator_ladder(
    proba: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int = 42,
    c_grid: Sequence[float] = (0.1, 1.0, 10.0),
) -> tuple[Any, str, dict[str, Any]]:
    """Fit the Dirichlet -> vector-temp -> temperature ladder on cal rows.

    Each candidate is scored by grouped 3-fold CV *within cal* (weighted
    log-loss); the first candidate that does NOT worsen the raw (identity)
    CV log-loss wins and is refit on all cal rows.  Returns
    ``(calibrator_or_None, kind, info)``.
    """
    info: dict[str, Any] = {"candidates": {}}
    y = np.asarray(y, dtype=int)
    n_classes_cal = len(np.unique(y))
    n_groups = len(np.unique(groups))
    if len(y) < 30 or n_groups < 3 or n_classes_cal < 2:
        info["skip_reason"] = (
            f"cal too small (n={len(y)}, groups={n_groups}, "
            f"classes={n_classes_cal})"
        )
        return None, "none", info

    raw_loss = _grouped_cv_loss("identity", proba, y, sample_weight, groups,
                                seed=seed)
    info["cal_cv_logloss_raw"] = raw_loss

    ladder: list[tuple[str, float]] = []
    if n_classes_cal == len(CLASSES):
        # Dirichlet / vector scaling can zero-out an unseen class — only
        # offered when all 3 classes are present in cal.
        best_c, best_c_loss = None, float("inf")
        for C in c_grid:
            loss = _grouped_cv_loss("dirichlet", proba, y, sample_weight,
                                    groups, C=C, seed=seed)
            info["candidates"][f"dirichlet_C={C}"] = loss
            if loss < best_c_loss:
                best_c, best_c_loss = float(C), float(loss)
        info["dirichlet_best_C"] = best_c
        ladder.append(("dirichlet", best_c_loss))
        vec_loss = _grouped_cv_loss("vector_temperature", proba, y,
                                    sample_weight, groups, seed=seed)
        info["candidates"]["vector_temperature"] = vec_loss
        ladder.append(("vector_temperature", vec_loss))
    temp_loss = _grouped_cv_loss("temperature", proba, y, sample_weight,
                                 groups, seed=seed)
    info["candidates"]["temperature"] = temp_loss
    ladder.append(("temperature", temp_loss))

    for kind, loss in ladder:
        if np.isfinite(loss) and loss <= raw_loss + 1e-12:
            C = info.get("dirichlet_best_C") or 1.0
            calibrator = _make_calibrator(kind, C=C, seed=seed)
            calibrator.fit(proba, y, sample_weight=sample_weight)
            info["cal_cv_logloss_chosen"] = float(loss)
            return calibrator, kind, info
    info["skip_reason"] = "no candidate improved raw cal-CV log-loss"
    return None, "none", info


# ---------------------------------------------------------------------------
# Model bundle helpers
# ---------------------------------------------------------------------------

def _fit_multiclass_model(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray,
    params: dict[str, Any],
    *,
    n_estimators: int,
    eval_data: tuple[pd.DataFrame, np.ndarray, np.ndarray] | None = None,
    early_stopping_rounds: int = 100,
    n_jobs: int = 8,
    seed: int = 42,
) -> dict[str, Any]:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier

    classes = np.unique(y)
    if len(classes) < 2:
        proba = np.zeros(len(CLASSES), dtype=float)
        if len(classes) == 1:
            proba[int(classes[0])] = 1.0
        return {"kind": "constant", "proba": proba.tolist()}

    model = LGBMClassifier(
        objective="multiclass",
        n_estimators=int(n_estimators),
        learning_rate=float(params["learning_rate"]),
        num_leaves=int(params["num_leaves"]),
        min_child_samples=int(params["min_child_samples"]),
        feature_fraction=0.7,
        bagging_fraction=0.8,
        bagging_freq=1,
        reg_alpha=0.1,
        reg_lambda=float(params["reg_lambda"]),
        random_state=seed,
        deterministic=True,
        force_row_wise=True,
        n_jobs=n_jobs,
        verbose=-1,
    )
    if eval_data is not None:
        X_val, y_val, w_val = eval_data
        model.fit(
            X, y,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            eval_sample_weight=[w_val],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
    else:
        model.fit(X, y, sample_weight=sample_weight)
    return {"kind": "lightgbm", "model": model}


def _predict_proba_full(model_bundle: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    """(N, 3) raw probabilities in CLASSES order, regardless of how many
    classes the fitted model saw."""
    n = len(X)
    if model_bundle["kind"] == "constant":
        return np.tile(np.asarray(model_bundle["proba"], dtype=float), (n, 1))
    model = model_bundle["model"]
    raw = model.predict_proba(X)
    out = np.zeros((n, len(CLASSES)), dtype=float)
    for col, cls in enumerate(model.classes_):
        out[:, int(cls)] = raw[:, col]
    return out


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------

@dataclass
class MulticlassFollowupArtifact:
    """Persisted fusion_v8 follow-up head: LightGBM multiclass(3) + the
    Dirichlet-ladder calibrator (global, plus optional per-survey)."""

    feature_cols: list[str]
    model_bundle: Any
    calibrator: Any = None
    survey_calibrators: dict[str, Any] = field(default_factory=dict)
    classes: tuple[str, ...] = CLASSES

    def predict_proba_raw(self, df: pd.DataFrame) -> np.ndarray:
        X = _prepare_frame(df, self.feature_cols)
        return _predict_proba_full(self.model_bundle, X)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Calibrated (N, 3) probabilities in ``CLASSES`` order.

        Rows are routed to their survey's calibrator when one was trained
        (>=150 cal objects in that survey), else to the global calibrator,
        else returned raw.  Output rows are renormalized to sum to 1.
        """
        raw = self.predict_proba_raw(df)
        out = raw.copy()
        routed = np.zeros(len(df), dtype=bool)
        if self.survey_calibrators and "survey" in df.columns:
            surveys = df["survey"].astype(str).to_numpy()
            for survey, calibrator in self.survey_calibrators.items():
                mask = surveys == survey
                if mask.any() and calibrator is not None:
                    out[mask] = calibrator.transform(raw[mask])
                    routed[mask] = True
        if self.calibrator is not None and (~routed).any():
            out[~routed] = self.calibrator.transform(raw[~routed])
        out = np.clip(out, 1e-9, 1.0)
        return out / out.sum(axis=1, keepdims=True)

    def save(self, out_dir: str) -> None:
        model_dir = Path(out_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "model.pkl", "wb") as fh:
            pickle.dump(self.model_bundle, fh)
        with open(model_dir / "calibrator.pkl", "wb") as fh:
            pickle.dump(
                {"global": self.calibrator, "per_survey": self.survey_calibrators},
                fh,
            )
        with open(model_dir / "metadata.json", "w") as fh:
            json.dump(
                {
                    "feature_cols": self.feature_cols,
                    "classes": list(self.classes),
                    "calibrator_kind": (
                        getattr(self.calibrator, "name", None)
                        if self.calibrator is not None
                        else None
                    ),
                    "survey_calibrator_kinds": {
                        survey: getattr(cal, "name", None)
                        for survey, cal in self.survey_calibrators.items()
                    },
                },
                fh,
                indent=2,
            )

    @classmethod
    def load(cls, out_dir: str) -> "MulticlassFollowupArtifact":
        model_dir = Path(out_dir)
        with open(model_dir / "metadata.json") as fh:
            metadata = json.load(fh)
        with open(model_dir / "model.pkl", "rb") as fh:
            model_bundle = pickle.load(fh)
        calibrator = None
        survey_calibrators: dict[str, Any] = {}
        calibrator_path = model_dir / "calibrator.pkl"
        if calibrator_path.exists():
            with open(calibrator_path, "rb") as fh:
                payload = pickle.load(fh)
            calibrator = payload.get("global")
            survey_calibrators = payload.get("per_survey", {}) or {}
        return cls(
            feature_cols=[str(c) for c in metadata["feature_cols"]],
            model_bundle=model_bundle,
            calibrator=calibrator,
            survey_calibrators=survey_calibrators,
            classes=tuple(metadata.get("classes", CLASSES)),
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _multiclass_metrics(y: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import balanced_accuracy_score

    metrics: dict[str, Any] = {"n_rows": int(len(y))}
    if len(y) == 0:
        metrics.update({"log_loss": None, "macro_ovr_auc": None,
                        "balanced_accuracy": None})
        return metrics
    metrics["log_loss"] = _weighted_log_loss(y, proba)
    per_class: dict[str, float | None] = {}
    aucs: list[float] = []
    for c, name in enumerate(CLASSES):
        y_bin = (y == c).astype(int)
        if len(np.unique(y_bin)) < 2:
            per_class[name] = None
            continue
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y_bin, proba[:, c]))
        per_class[name] = auc
        aucs.append(auc)
    metrics["ovr_auc"] = per_class
    metrics["macro_ovr_auc"] = float(np.mean(aucs)) if aucs else None
    metrics["balanced_accuracy"] = float(
        balanced_accuracy_score(y, np.argmax(proba, axis=1))
    )
    return metrics


def _hyperparam_grid(grid_small: bool) -> list[dict[str, Any]]:
    """Design §5.2 grid (36 configs) or a 2-config smoke grid."""
    if grid_small:
        return [
            {"num_leaves": nl, "min_child_samples": 20,
             "learning_rate": 0.05, "reg_lambda": 0.1}
            for nl in (15, 31)
        ]
    grid: list[dict[str, Any]] = []
    for num_leaves in (15, 31, 63):
        for min_child_samples in (20, 50, 100):
            for learning_rate in (0.03, 0.05):
                for reg_lambda in (0.1, 1.0):
                    grid.append({
                        "num_leaves": num_leaves,
                        "min_child_samples": min_child_samples,
                        "learning_rate": learning_rate,
                        "reg_lambda": reg_lambda,
                    })
    return grid


# ---------------------------------------------------------------------------
# Training entry point (pinned interface)
# ---------------------------------------------------------------------------

def train_multiclass_followup(
    snapshots: pd.DataFrame,
    train_ids: set,
    cal_ids: set,
    test_ids: set,
    out_dir: str,
    *,
    weak_weight: float = 0.1,
    context_weight: float = 0.15,
    tns_untyped_weight: float = 0.6,
    bts_weight: float = 1.0,
    aug_frac: float = 0.25,
    aug_weight: float = 0.3,
    grid_small: bool = False,
    n_jobs: int = 8,
    seed: int = 42,
) -> dict:
    """Train the fusion_v8 multiclass follow-up head; save artifact to
    ``out_dir``; return a JSON-serializable metrics dict.

    Protocol (no-leakage):
      1. split-integrity asserts (no object in two splits);
      2. provenance masking on TRAIN rows (spec correction #1);
      3. base weights ``w_qual × 1/n_rows(object) × w_bts``; class weights
         from inverse weighted class frequency on TRAIN;
      4. inner grouped early-stopping fold carved from TRAIN
         (GroupShuffleSplit, 20% of train objects) — cal never touched by
         fitting; hyperparams selected on inner-fold weighted log-loss;
      5. expert-dropout augmentation (spec correction #4a) applied to the
         fitting rows only, never to the inner validation fold;
      6. refit on full TRAIN (+regenerated augmentation) at the selected
         config's early-stopped iteration count;
      7. Dirichlet-ladder calibration on CAL (spec+tns_untyped rows,
         object-weighted), plus per-survey calibrators where the survey has
         >= 150 cal objects;
      8. raw vs calibrated metrics on the locked TEST split.
    """
    train_ids = {str(i) for i in train_ids}
    cal_ids = {str(i) for i in cal_ids}
    test_ids = {str(i) for i in test_ids}

    # --- Split integrity validation (loud failures) ---
    assert len(train_ids & cal_ids) == 0, "train/cal overlap detected"
    assert len(train_ids & test_ids) == 0, "train/test overlap detected"
    assert len(cal_ids & test_ids) == 0, "cal/test overlap detected"

    labelled = snapshots[snapshots["target_class"].isin(CLASSES)].copy()
    labelled["object_id"] = labelled["object_id"].astype(str)
    if len(labelled) == 0:
        raise ValueError("No rows with a usable ternary target_class")

    feature_cols = _numeric_feature_cols(labelled)
    _assert_no_label_features(feature_cols)

    oid = labelled["object_id"]
    train_df = labelled[oid.isin(train_ids)].copy()
    cal_df = labelled[oid.isin(cal_ids)].copy()
    test_df = labelled[oid.isin(test_ids)].copy()
    if len(train_df) == 0:
        raise ValueError("No multiclass follow-up training rows available")

    # --- Provenance masking: TRAIN rows only.  Cal calibration uses
    # spec/tns_untyped labels (never context/self-derived) and test must see
    # production features, so masking train is sufficient and honest. ---
    train_df = apply_provenance_masking(train_df)
    masking_info = dict(train_df.attrs.get("provenance_masking", {}))

    y_train = train_df["target_class"].map(CLASS_TO_IDX).to_numpy(dtype=int)
    base_w = compute_base_weights(
        train_df,
        weak_weight=weak_weight,
        context_weight=context_weight,
        tns_untyped_weight=tns_untyped_weight,
        bts_weight=bts_weight,
    )
    keep = base_w > 0
    if not keep.all():
        train_df = train_df[keep].copy()
        y_train = y_train[keep]
        base_w = base_w[keep]
    if len(train_df) == 0:
        raise ValueError("All training rows have zero weight")

    w_cls = _class_weight_vector(y_train, base_w)

    # --- Inner grouped early-stopping fold carved from train ---
    from sklearn.model_selection import GroupShuffleSplit

    groups = train_df["object_id"].to_numpy()
    n_groups = len(np.unique(groups))
    if n_groups >= 5:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        inner_fit_idx, inner_val_idx = next(gss.split(train_df, y_train, groups))
    else:
        inner_fit_idx = np.arange(len(train_df))
        inner_val_idx = np.zeros(0, dtype=int)

    def _augmented_matrix(
        frame: pd.DataFrame, y: np.ndarray, w_base: np.ndarray
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
        """frame (+expert-dropout copies) -> (X, y, final sample weights)."""
        aug_df, aug_w, aug_info = expert_dropout_augment(
            frame, w_base, aug_frac=aug_frac, aug_weight=aug_weight, seed=seed
        )
        if len(aug_df) > 0:
            full = pd.concat([frame, aug_df], ignore_index=True, sort=False)
            y_full = np.concatenate([
                y, aug_df["target_class"].map(CLASS_TO_IDX).to_numpy(dtype=int)
            ])
            w_full = np.concatenate([w_base, aug_w])
        else:
            full, y_full, w_full = frame, y, w_base
        X_full = _prepare_frame(full, feature_cols)
        return X_full, y_full, w_full * w_cls[y_full], aug_info

    grid = _hyperparam_grid(grid_small)
    trace: list[dict[str, Any]] = []
    best_entry: dict[str, Any] | None = None

    if len(inner_val_idx) > 0:
        inner_fit_df = train_df.iloc[inner_fit_idx]
        X_it, y_it, w_it, aug_info_inner = _augmented_matrix(
            inner_fit_df, y_train[inner_fit_idx], base_w[inner_fit_idx]
        )
        X_iv = _prepare_frame(train_df.iloc[inner_val_idx], feature_cols)
        y_iv = y_train[inner_val_idx]
        w_iv = base_w[inner_val_idx] * w_cls[y_iv]
        for params in grid:
            bundle = _fit_multiclass_model(
                X_it, y_it, w_it, params,
                n_estimators=2000,
                eval_data=(X_iv, y_iv, w_iv),
                n_jobs=n_jobs, seed=seed,
            )
            if bundle["kind"] == "constant":
                inner_loss = _weighted_log_loss(
                    y_iv, np.tile(bundle["proba"], (len(y_iv), 1)), w_iv
                )
                best_iteration = 0
            else:
                model = bundle["model"]
                best_iteration = int(model.best_iteration_ or model.n_estimators)
                inner_loss = float(
                    model.best_score_["valid_0"]["multi_logloss"]
                )
            entry = {
                "params": dict(params),
                "best_iteration": best_iteration,
                "inner_val_logloss": inner_loss,
            }
            trace.append(entry)
            if best_entry is None or inner_loss < best_entry["inner_val_logloss"]:
                best_entry = entry
        aug_info = aug_info_inner
    else:
        # Degenerate tiny-train fallback: no inner fold possible.
        best_entry = {
            "params": dict(grid[0]),
            "best_iteration": 200,
            "inner_val_logloss": None,
        }
        aug_info = {}

    assert best_entry is not None
    best_params = best_entry["params"]
    best_iteration = max(int(best_entry["best_iteration"]), 1)

    # --- Final refit on FULL train (+regenerated augmentation) ---
    X_tr, y_tr, w_tr, aug_info_full = _augmented_matrix(train_df, y_train, base_w)
    model_bundle = _fit_multiclass_model(
        X_tr, y_tr, w_tr, best_params,
        n_estimators=best_iteration,
        eval_data=None, n_jobs=n_jobs, seed=seed,
    )

    # --- Raw probabilities for cal/test ---
    proba_cal_raw = _predict_proba_full(
        model_bundle, _prepare_frame(cal_df, feature_cols)
    ) if len(cal_df) else np.zeros((0, len(CLASSES)))
    proba_test_raw = _predict_proba_full(
        model_bundle, _prepare_frame(test_df, feature_cols)
    ) if len(test_df) else np.zeros((0, len(CLASSES)))

    # --- Dirichlet-ladder calibration on cal (spec + tns_untyped only) ---
    if "label_quality" in cal_df.columns:
        cal_fit_mask = (
            cal_df["label_quality"].astype(str).str.lower().isin(_SPEC_QUALITIES)
        ).to_numpy()
    else:
        cal_fit_mask = np.ones(len(cal_df), dtype=bool)
    cal_fit_df = cal_df[cal_fit_mask]
    y_cal_fit = cal_fit_df["target_class"].map(CLASS_TO_IDX).to_numpy(dtype=int)
    # Object weighting (1/n_rows) per design §4.1 — quality is uniform-spec
    # here, so this is pure object normalization.
    w_cal_fit = compute_base_weights(
        cal_fit_df,
        weak_weight=weak_weight,
        context_weight=context_weight,
        tns_untyped_weight=tns_untyped_weight,
        bts_weight=1.0,
    ) if len(cal_fit_df) else np.zeros(0)
    groups_cal = cal_fit_df["object_id"].to_numpy() if len(cal_fit_df) else np.zeros(0)

    calibrator, calibrator_kind, cal_info = (
        _fit_calibrator_ladder(
            proba_cal_raw[cal_fit_mask], y_cal_fit, w_cal_fit, groups_cal,
            seed=seed,
        )
        if len(cal_fit_df) > 0
        else (None, "none", {"skip_reason": "no cal rows"})
    )

    survey_calibrators: dict[str, Any] = {}
    survey_cal_info: dict[str, Any] = {}
    if len(cal_fit_df) > 0 and "survey" in cal_fit_df.columns:
        for survey in sorted(cal_fit_df["survey"].astype(str).unique()):
            mask = (cal_fit_df["survey"].astype(str) == survey).to_numpy()
            n_objects = cal_fit_df.loc[mask, "object_id"].nunique()
            if n_objects < 150:
                survey_cal_info[survey] = {
                    "kind": "none",
                    "skip_reason": f"cal_objects<150 (have {int(n_objects)})",
                }
                continue
            cal_s, kind_s, info_s = _fit_calibrator_ladder(
                proba_cal_raw[cal_fit_mask][mask],
                y_cal_fit[mask],
                w_cal_fit[mask],
                groups_cal[mask],
                seed=seed,
            )
            survey_cal_info[survey] = {"kind": kind_s, **{
                k: v for k, v in info_s.items() if k != "candidates"
            }}
            if cal_s is not None:
                survey_calibrators[survey] = cal_s

    artifact = MulticlassFollowupArtifact(
        feature_cols=feature_cols,
        model_bundle=model_bundle,
        calibrator=calibrator,
        survey_calibrators=survey_calibrators,
    )
    artifact.save(out_dir)

    # --- Honest reporting: spec(+tns_untyped) rows on cal and locked test ---
    def _quality_mask(df: pd.DataFrame) -> np.ndarray:
        if "label_quality" not in df.columns:
            return np.ones(len(df), dtype=bool)
        return (
            df["label_quality"].astype(str).str.lower().isin(_SPEC_QUALITIES)
        ).to_numpy()

    test_mask = _quality_mask(test_df)
    y_test = (
        test_df.loc[test_mask, "target_class"].map(CLASS_TO_IDX)
        .to_numpy(dtype=int)
    )
    proba_test_calibrated = (
        artifact.predict_proba(test_df) if len(test_df) else proba_test_raw
    )
    proba_cal_calibrated = (
        artifact.predict_proba(cal_df) if len(cal_df) else proba_cal_raw
    )

    metrics: dict[str, Any] = {
        "classes": list(CLASSES),
        "n_train_rows": int(len(train_df)),
        "n_cal_rows": int(len(cal_df)),
        "n_test_rows": int(len(test_df)),
        "n_train_objects": int(train_df["object_id"].nunique()),
        "n_cal_objects": int(cal_df["object_id"].nunique()) if len(cal_df) else 0,
        "n_test_objects": int(test_df["object_id"].nunique()) if len(test_df) else 0,
        "n_features": int(len(feature_cols)),
        "class_weights": [float(v) for v in w_cls],
        "provenance_masking": masking_info,
        "augmentation": {
            "aug_frac": float(aug_frac),
            "aug_weight": float(aug_weight),
            "inner_fold": {
                k: int(v) for k, v in aug_info.items()
                if k != "source_positions"
            },
            "full_train": {
                k: int(v) for k, v in aug_info_full.items()
                if k != "source_positions"
            },
        },
        "grid": {
            "grid_small": bool(grid_small),
            "n_configs": len(grid),
            "trace": trace,
            "best_params": best_params,
            "best_iteration": best_iteration,
        },
        "calibration": {
            "kind": calibrator_kind,
            "cal_cv_logloss_raw": cal_info.get("cal_cv_logloss_raw"),
            "cal_cv_logloss_chosen": cal_info.get("cal_cv_logloss_chosen"),
            "candidates": cal_info.get("candidates", {}),
            "skip_reason": cal_info.get("skip_reason"),
            "per_survey": survey_cal_info,
        },
        "test_raw": _multiclass_metrics(y_test, proba_test_raw[test_mask]),
        "test_calibrated": _multiclass_metrics(
            y_test, proba_test_calibrated[test_mask]
        ),
        "artifact_dir": str(out_dir),
    }
    cal_mask_rep = _quality_mask(cal_df)
    y_cal_rep = (
        cal_df.loc[cal_mask_rep, "target_class"].map(CLASS_TO_IDX)
        .to_numpy(dtype=int)
    )
    metrics["cal_raw"] = _multiclass_metrics(
        y_cal_rep, proba_cal_raw[cal_mask_rep]
    )
    metrics["cal_calibrated"] = _multiclass_metrics(
        y_cal_rep, proba_cal_calibrated[cal_mask_rep]
    )
    return metrics
