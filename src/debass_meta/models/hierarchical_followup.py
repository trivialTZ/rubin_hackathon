"""Hierarchical follow-up head for fusion_v11  [work package P3].

Replaces the flat 3-class Stage-B (``multiclass_followup.py``) with a two-head
factorization:

* **Head-1  P(SN | x)** — binary LightGBM, ``snia + nonIa_snlike`` vs ``other``.
  Trained on ALL label tiers (spec, weak-SN, catalog-others, tns_untyped): the
  weak "SN" labels finally supervise exactly the level they constrain (fusion
  bug #1: a weak stamp-SN label knows SN-ness, not subtype).  Preregistered
  default is a POOLED head (with ``survey_is_lsst`` as a feature); a per-survey
  realization is a cal-gated arm that stays OFF when the is_sn-level LSST cal
  frame is small (spec §2.1 / B1 — it is small in practice, so pooled is the
  operating default).

* **Head-2  P(Ia | SN, x)** — binary LightGBM, trained ONLY on
  ``label_quality == 'spectroscopic'`` SN rows, **starting ZTF-only** (B1).
  It **drops ``survey_is_lsst`` and every survey-degenerate column by DEFAULT**
  (not a gate — the gate is undecidable with no LSST spec cal and the v10 harm
  was survey-flag-mediated).  On LSST rows the default routing is a per-survey
  constant base rate ``P(Ia|SN)`` clipped to ``[0.05, 0.95]``; the shared head-2
  is only applied to LSST rows when a gate pays (LSST spec cal/OOF n >= 30 AND
  Spearman(p_snia, is_Ia) >= 0).

**Composition** (spec §2.1, ``multiclass_followup.py:624`` clip precedent):
each head is clipped to ``[1e-6, 1-1e-6]`` BEFORE composing::

    p_snia         = P1 * P2
    p_nonIa_snlike = P1 * (1 - P2)
    p_other        = 1 - P1

which sums to 1 by construction (``P1*P2 + P1*(1-P2) + (1-P1) == 1``) — no
per-class recalibration of the composed marginals (only a simplex-preserving
Dirichlet rung, out of P3 scope, may follow composition downstream).

**Calibration is per-head** (never a recalibration of the composed simplex):
per-survey binary calibrators — isotonic when the survey cal frame has
``n >= survey_cal_min`` (constructor param, default 40), else ``PlattCalibrator``
(``pooled_trust.PlattCalibrator``) — **not** temperature: temperature cannot
shift the intercept, and intercept bias was the v10 failure mode.  Opt-in
(``head1_calibrator={"lsst": "beta"}``, v13g): a per-survey family override;
``beta`` is a smooth monotone map (``calibrate.BetaCalibrator``) without the
tie plateaus of isotonic.  Head-2's
calibrator is fit on true-SN spec rows and applied everywhere (covariate shift
accepted, documented here per spec §2.1).

``predict_proba_raw`` composes the UNcalibrated heads; ``predict_proba`` composes
the calibrated heads.  Both are required by the v8 scorer
(``score_fusion_v8.py:340-345``) and mirror the ``MulticlassFollowupArtifact``
interface (``feature_cols`` per head in ``metadata.json``; survey routing by the
literal lowercase ``survey`` string; ``_prepare_frame`` NaN-fills missing cols).

Everything imported from ``multiclass_followup`` / ``calibrate`` / ``pooled_trust``
is used read-only — this module never forks those helpers.

fusion v13 (docs/fusion_v13_plan.md; every option defaults to the v11/v12
behaviour so existing artifacts and job scripts are unchanged):

* B1 ``head1_exclude_qualities`` — label tiers (e.g. ``("weak",)``) dropped
  from head-1 training, the head-1 calibrators and the per-survey gate count.
* B2 ``equalize_lsst_provenance=False`` switches off the LSST equalization
  that masked the whole ALeRCE family (incl. the LOCAL ``alerce_lc`` and the
  Rubin stamp) on every LSST weak+context row — class-pure missingness once
  the weak rows are gone.  ``context_mask_scope="survey"`` masks the context
  family (``lasair/sherlock``, ``babamul``) on EVERY row of a survey that has
  context-labelled rows, because masking it only on the catalogue-"other"
  rows is class-pure too.  ``head1_survey_masks`` masks named experts on a
  whole survey (for genuine availability gaps in the gold).  Guard G8
  (``g8_max_corr``) fails the fit when any expert's weighted
  |corr(avail, is_SN)| on the final head-1 LSST/ZTF frame exceeds the threshold.
* B3 ``dropout`` (:class:`AvailabilityDropout`) — availability-dropout copies
  (structured no-broker / no-local / no-expert regimes on LSST rows, random
  subset drops on all rows); each object's total weight is unchanged.
* B4 ``cross_fit_folds`` — GroupKFold out-of-fold head predictions on train;
  the head calibrators are then fit on OOF-train ∪ cal (spec + context, in the
  same regime mixture) and :meth:`predict_proba_oof` lets the orchestrator fit
  α / evaluate G2 out of sample.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

# Read-only imports of the shared ladder / helpers (spec §3, P3 line 220-221).
from .calibrate import BetaCalibrator, IsotonicCalibrator
from .multiclass_followup import (
    CLASSES,
    _ALERCE_FAMILY,
    _CONTEXT_FAMILY,
    _mask_expert_blocks,
    _numeric_feature_cols,
    _prepare_frame,
    apply_provenance_masking,
    blank_expert_blocks,
    compute_base_weights,
    expert_dropout_augment,
)
from .pooled_trust import PlattCalibrator

# Ternary class order — must match ``multiclass_followup.CLASSES`` so the
# composed (N, 3) matrix drops straight into the v8 scorer.
assert CLASSES == ("snia", "nonIa_snlike", "other")

_SN_CLASSES = ("snia", "nonIa_snlike")
_SPEC_QUALITIES = ("spectroscopic", "tns_untyped")
_HEAD2_SPEC_QUALITY = "spectroscopic"

# Guard G7 (B0): a head-2 row is TYPED iff ``tns_type`` is non-empty OR
# ``bts_type`` is a real subtype (not in this untyped/sentinel set).
_UNTYPED_BTS = ("", "-", "nan", "none")

# Composition clip (multiclass_followup.py:624 precedent) — applied to BOTH
# heads BEFORE composing, so the simplex is exact and log-loss stays finite.
_HEAD_CLIP = (1e-6, 1.0 - 1e-6)

# Columns head-2 drops by DEFAULT (survey flag ablation is a default, not a gate).
_HEAD2_DROP_COLS = ("survey_is_lsst",)

# Default LightGBM params for each binary head (mirrors the v8 multiclass grid
# point that consistently won on cal).
_HEAD_PARAMS = {
    "num_leaves": 31,
    "min_child_samples": 20,
    "learning_rate": 0.05,
    "reg_lambda": 0.1,
}

# fusion v13: when cross-fitting, the head-1 calibrators see OOF-train ∪ cal
# rows of exactly the label tiers head 1 trains on (``head1_exclude_qualities``
# applied to both, survey-scoped).  A hard "spectroscopic + context" list
# would leave the ZTF calibrator single-class: ZTF's only "other" labels are
# weak rows (v12 calibrated ZTF on them too).
_G8_SURVEYS = ("lsst", "ztf")
_REGIME_GROUPS = {"no_broker": "brokers", "no_local": "local", "no_expert": "all"}


class G8Error(AssertionError):
    """Guard G8: class-correlated expert availability in the head-1 frame."""


@dataclass(frozen=True)
class AvailabilityDropout:
    """Availability-dropout settings (fusion v13 B3).

    ``regimes``       structured copies, applied to rows of ``regime_surveys``:
                      ``no_broker`` / ``no_local`` / ``no_expert`` blank the
                      ``features.availability`` groups (brokers / local / all).
    ``regime_weight`` weight SHARE each structured copy takes from its source
                      row (the original keeps ``1 - Σ shares``, so every
                      object's total weight is unchanged).
    ``random_frac``   fraction of source rows (all surveys) that get ONE
                      random-subset-drop copy (``expert_dropout_augment``,
                      full-block blanking); its share is ``random_weight``
                      (default = ``regime_weight``).
    ``keep_one_frac`` share of random copies in keep-exactly-one mode.
    ``seed``          RNG seed of the random regime.
    """

    regimes: tuple[str, ...] = ("no_broker", "no_local", "no_expert")
    regime_weight: float = 0.1
    random_frac: float = 0.25
    random_weight: float | None = None
    keep_one_frac: float = 0.0
    regime_surveys: tuple[str, ...] = ("lsst",)
    seed: int = 42

    def __post_init__(self) -> None:
        bad = [r for r in self.regimes if r not in _REGIME_GROUPS]
        if bad:
            raise ValueError(f"unknown dropout regimes {bad}; "
                             f"choose from {sorted(_REGIME_GROUPS)}")
        total = self.regime_weight * len(self.regimes) + self.random_share
        if not (0.0 <= self.regime_weight and 0.0 <= self.random_share) or total >= 1.0:
            raise ValueError(
                f"dropout weight shares must be >= 0 with a total < 1 "
                f"(got {len(self.regimes)} x {self.regime_weight} + "
                f"{self.random_share} = {total})")

    @property
    def random_share(self) -> float:
        return float(self.regime_weight if self.random_weight is None
                     else self.random_weight)

    def to_dict(self) -> dict[str, Any]:
        return {
            "regimes": list(self.regimes),
            "regime_weight": float(self.regime_weight),
            "random_frac": float(self.random_frac),
            "random_weight": self.random_weight,
            "random_share": self.random_share,
            "keep_one_frac": float(self.keep_one_frac),
            "regime_surveys": list(self.regime_surveys),
            "seed": int(self.seed),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AvailabilityDropout":
        return cls(
            regimes=tuple(d.get("regimes", ("no_broker", "no_local", "no_expert"))),
            regime_weight=float(d.get("regime_weight", 0.1)),
            random_frac=float(d.get("random_frac", 0.25)),
            random_weight=d.get("random_weight"),
            keep_one_frac=float(d.get("keep_one_frac", 0.0)),
            regime_surveys=tuple(d.get("regime_surveys", ("lsst",))),
            seed=int(d.get("seed", 42)),
        )


# ---------------------------------------------------------------------------
# Binary LightGBM head bundle
# ---------------------------------------------------------------------------

def _fit_binary_model(
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
    """Fit one binary head; returns a bundle dict.

    Degenerate single-class input -> a constant bundle emitting that class's
    probability (mirrors ``_fit_multiclass_model``'s constant branch).
    """
    import lightgbm as lgb
    from lightgbm import LGBMClassifier

    y = np.asarray(y, dtype=int)
    classes = np.unique(y)
    if len(classes) < 2:
        proba = float(classes[0]) if len(classes) == 1 else 0.0
        return {"kind": "constant", "proba": proba}

    model = LGBMClassifier(
        objective="binary",
        n_estimators=int(n_estimators),
        learning_rate=float(params["learning_rate"]),
        num_leaves=int(params["num_leaves"]),
        min_child_samples=int(params["min_child_samples"]),
        feature_fraction=0.7,
        bagging_fraction=0.8,
        bagging_freq=1,
        reg_alpha=0.1,
        reg_lambda=float(params["reg_lambda"]),
        is_unbalance=True,
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
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
    else:
        model.fit(X, y, sample_weight=sample_weight)
    return {"kind": "lightgbm", "model": model}


def _predict_binary(bundle: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    """P(y == 1) for the binary head, robust to constant bundles / empty X."""
    n = len(X)
    if n == 0:
        return np.zeros(0, dtype=float)
    if bundle["kind"] == "constant":
        return np.full(n, float(bundle["proba"]), dtype=float)
    model = bundle["model"]
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)].astype(float)
    # model never saw the positive class -> all-zero probability
    return np.zeros(n, dtype=float)


def _train_binary_head(
    frame: pd.DataFrame,
    y: np.ndarray,
    base_w: np.ndarray,
    feature_cols: list[str],
    *,
    params: dict[str, Any],
    n_jobs: int,
    seed: int,
) -> dict[str, Any]:
    """Grouped early-stopping inner fold -> refit on full frame (v8 protocol)."""
    from sklearn.model_selection import GroupShuffleSplit

    X = _prepare_frame(frame, feature_cols)
    groups = frame["object_id"].astype(str).to_numpy()
    n_groups = len(np.unique(groups))
    if n_groups >= 5 and len(np.unique(y)) >= 2:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        fit_idx, val_idx = next(gss.split(frame, y, groups))
    else:
        fit_idx, val_idx = np.arange(len(frame)), np.zeros(0, dtype=int)

    best_iteration = 200
    if len(val_idx) > 0 and len(np.unique(y[fit_idx])) >= 2:
        probe = _fit_binary_model(
            X.iloc[fit_idx], y[fit_idx], base_w[fit_idx], params,
            n_estimators=1000,
            eval_data=(X.iloc[val_idx], y[val_idx], base_w[val_idx]),
            n_jobs=n_jobs, seed=seed,
        )
        if probe["kind"] == "lightgbm":
            model = probe["model"]
            best_iteration = int(model.best_iteration_ or model.n_estimators)
    best_iteration = max(int(best_iteration), 1)

    return _fit_binary_model(
        X, y, base_w, params, n_estimators=best_iteration,
        eval_data=None, n_jobs=n_jobs, seed=seed,
    )


# ---------------------------------------------------------------------------
# Binary calibrators — isotonic (n >= survey_cal_min) else Platt, never temp.
# ---------------------------------------------------------------------------

class _IdentityBinaryCalibrator:
    """No-op binary calibrator (returned when a class is missing / n < 2)."""

    name = "identity"

    def fit(self, y_prob, y_true) -> "_IdentityBinaryCalibrator":
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        return np.asarray(y_prob, dtype=float)


class _WeightedIsotonicCalibrator(IsotonicCalibrator):
    """``IsotonicCalibrator`` with per-row weights (fusion v13 cross-fit
    calibration on the object-normalized OOF-train ∪ cal frame)."""

    def fit(self, y_prob, y_true, sample_weight=None) -> "_WeightedIsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression

        y_prob = np.asarray(y_prob, dtype=float)
        y_true = np.asarray(y_true, dtype=int)
        self._ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self._ir.fit(y_prob, y_true, sample_weight=sample_weight)
        return self


class _WeightedPlattCalibrator(PlattCalibrator):
    """``PlattCalibrator`` with per-row weights (fusion v13)."""

    def fit(self, y_prob, y_true, sample_weight=None) -> "_WeightedPlattCalibrator":
        from sklearn.linear_model import LogisticRegression

        x = self._logit(y_prob).reshape(-1, 1)
        y = np.asarray(y_true, dtype=int)
        self._lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=42)
        self._lr.fit(x, y, sample_weight=sample_weight)
        return self


HEAD1_CALIBRATOR_KINDS = ("beta", "platt", "isotonic")


def parse_head1_calibrator_specs(specs: Sequence[str]) -> dict[str, str]:
    """``["lsst:beta", ...]`` -> ``{"lsst": "beta"}`` (the ``--head1-calibrator``
    syntax; the survey key is lower-cased as for the survey masks)."""
    out: dict[str, str] = {}
    for spec in specs:
        sv, _, kind = str(spec).partition(":")
        sv, kind = sv.strip().lower(), kind.strip().lower()
        if not sv or kind not in HEAD1_CALIBRATOR_KINDS:
            raise ValueError(
                f"--head1-calibrator expects SURVEY:KIND with KIND in {HEAD1_CALIBRATOR_KINDS}, got {spec!r}")
        out[sv] = kind
    return out


def _fit_binary_calibrator(
    p: np.ndarray, y: np.ndarray, *, n_min: int, sample_weight: np.ndarray | None = None,
    kind: str | None = None,
):
    """Isotonic when ``n >= n_min`` else Platt (spec §2.1); identity when a
    class is absent or the frame is empty.  Platt (not temperature) is the
    tiny-n fallback because temperature cannot shift the intercept bias that
    killed v10 (spec deviation #21).

    ``sample_weight`` (fusion v13, optional) fits the weighted variants; with
    ``None`` the original unweighted classes are used.  ``kind`` (opt-in, one
    of ``HEAD1_CALIBRATOR_KINDS``) overrides the ``n_min`` rule: ``beta`` is the
    smooth beta calibration, ``platt`` / ``isotonic`` force that family."""
    if kind is not None and kind not in HEAD1_CALIBRATOR_KINDS:
        raise ValueError(f"unknown calibrator kind {kind!r}; expected one of {HEAD1_CALIBRATOR_KINDS}")
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    ok = np.isfinite(p)
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if w is not None:
        ok &= np.isfinite(w) & (w > 0)
        w = w[ok]
    p, y = p[ok], y[ok]
    if len(y) < 2 or len(np.unique(y)) < 2:
        return _IdentityBinaryCalibrator()
    if kind == "beta":
        return BetaCalibrator().fit(p, y, sample_weight=w)
    isotonic = (len(y) >= int(n_min)) if kind is None else kind == "isotonic"
    if w is None:
        return IsotonicCalibrator().fit(p, y) if isotonic else PlattCalibrator().fit(p, y)
    if isotonic:
        return _WeightedIsotonicCalibrator().fit(p, y, sample_weight=w)
    return _WeightedPlattCalibrator().fit(p, y, sample_weight=w)


# ---------------------------------------------------------------------------
# Row selection helpers (public — tests assert weak-label routing on them)
# ---------------------------------------------------------------------------

def survey_series(df: pd.DataFrame) -> np.ndarray:
    """Positional lowercase survey strings (``'ztf'`` / ``'lsst'`` / other)."""
    if "survey" in df.columns:
        return df["survey"].astype(str).str.lower().to_numpy()
    return np.array([""] * len(df), dtype=object)


def is_sn_target(df: pd.DataFrame) -> np.ndarray:
    """Head-1 target: 1 when ``target_class`` is an SN subtype, else 0."""
    return df["target_class"].isin(_SN_CLASSES).to_numpy().astype(int)


def head2_training_mask(
    df: pd.DataFrame, *, surveys: Sequence[str] = ("ztf",)
) -> np.ndarray:
    """Head-2 (Ia|SN) training rows: spectroscopic **SN** rows on the given
    surveys ONLY.  Weak / context / tns_untyped SN rows are NEVER admitted —
    a weak label knows SN-ness, not subtype (fusion bug #1).  Starting
    ``surveys=('ztf',)`` realizes the ZTF-only start (B1)."""
    sn = df["target_class"].isin(_SN_CLASSES).to_numpy()
    if "label_quality" in df.columns:
        spec = (
            df["label_quality"].astype(str).str.lower() == _HEAD2_SPEC_QUALITY
        ).to_numpy()
    else:
        spec = np.ones(len(df), dtype=bool)
    sv = survey_series(df)
    want = {str(s).lower() for s in surveys}
    on_survey = np.isin(sv, list(want))
    return sn & spec & on_survey


def has_provenance_cols(df: pd.DataFrame) -> bool:
    """True when the frame carries at least one subtype-provenance column —
    i.e. it was built from ``object_truth_v11.parquet`` (guard G7 presupposes
    this; a frame lacking BOTH columns cannot be checked)."""
    return ("tns_type" in df.columns) or ("bts_type" in df.columns)


def typed_provenance_mask(df: pd.DataFrame) -> np.ndarray:
    """Guard-G7 admission mask: True where a row carries a CONCRETE subtype.

    Admitted iff ``tns_type`` is non-empty OR ``bts_type`` ∉ {'-', '', 'nan',
    'none'}.  A missing provenance column contributes no positive evidence, so
    on a frame lacking BOTH columns every row is untyped (``has_provenance_cols``
    gates whether the assert is even evaluated).
    """
    n = len(df)
    typed = np.zeros(n, dtype=bool)
    if "tns_type" in df.columns:
        t = df["tns_type"].astype("string").str.strip()
        typed |= (
            t.notna() & (t.str.len() > 0)
            & ~t.str.lower().isin(["nan", "none"])
        ).to_numpy()
    if "bts_type" in df.columns:
        b = df["bts_type"].astype("string").str.strip()
        typed |= (
            b.notna() & ~b.str.lower().isin(list(_UNTYPED_BTS))
        ).to_numpy()
    return typed


def parse_quality_specs(qualities: Sequence[str]) -> list[tuple[str | None, str]]:
    """``"weak"`` -> (None, "weak") [all surveys]; ``"lsst:weak"`` -> ("lsst", "weak")."""
    out: list[tuple[str | None, str]] = []
    for q in qualities:
        q = str(q).strip().lower()
        if not q:
            continue
        if ":" in q:
            sv, qual = q.split(":", 1)
            out.append((sv.strip() or None, qual.strip()))
        else:
            out.append((None, q))
    return out


def excluded_quality_mask(df: pd.DataFrame, qualities: Sequence[str]) -> np.ndarray:
    """True where a row's (survey, label_quality) matches an exclusion spec
    (B1).  Specs are ``"<quality>"`` (every survey) or ``"<survey>:<quality>"``
    (that survey only, e.g. ``"lsst:weak"`` keeps the ZTF weak rows that are
    ZTF's only source of "other" labels)."""
    n = len(df)
    drop = np.zeros(n, dtype=bool)
    specs = parse_quality_specs(qualities)
    if not specs or "label_quality" not in df.columns:
        return drop
    lq = df["label_quality"].astype(str).str.lower().to_numpy()
    sv = survey_series(df)
    for survey, qual in specs:
        m = lq == qual
        if survey is not None:
            m &= sv == survey
        drop |= m
    return drop


def exclude_qualities(df: pd.DataFrame, qualities: Sequence[str]) -> pd.DataFrame:
    """Rows of ``df`` not matched by the exclusion specs (see
    :func:`excluded_quality_mask`)."""
    drop = excluded_quality_mask(df, qualities)
    return df if not drop.any() else df[~drop]


def row_keys(df: pd.DataFrame, regime: str | np.ndarray | pd.Series = "") -> np.ndarray:
    """``object_id|n_det|regime`` keys identifying a gold row and its
    availability-dropout copy (``regime == ""`` for the original row)."""
    n_det = pd.to_numeric(df["n_det"], errors="coerce")
    nd = np.where(n_det.notna(), n_det.fillna(-1).astype(int).astype(str), "nan")
    if isinstance(regime, str):
        reg = np.full(len(df), regime, dtype=object)
    else:
        reg = pd.Series(regime, index=df.index).fillna("").astype(str).to_numpy()
    oid = df["object_id"].astype(str).to_numpy()
    return np.array([f"{o}|{n}|{r}" for o, n, r in zip(oid, nd, reg)], dtype=object)


def _with_row_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "is_aug" not in out.columns:
        out["is_aug"] = 0.0
    if "aug_regime" not in out.columns:
        out["aug_regime"] = ""
    out["row_key"] = row_keys(out, out["aug_regime"])
    return out


def _group_sans(group: str, columns: Sequence[str]) -> list[str]:
    """Sanitized keys of the experts of ``features.availability`` group
    ``group`` that have an ``avail__`` column in ``columns``."""
    from debass_meta.features.availability import GROUPS
    from debass_meta.projectors.base import sanitize_expert_key

    present = {c[len("avail__"):] for c in columns if c.startswith("avail__")}
    return sorted(present & {sanitize_expert_key(k) for k in GROUPS[group]})


def augment_availability(
    frame: pd.DataFrame,
    base_w: np.ndarray,
    spec: AvailabilityDropout,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Append availability-dropout copies to ``frame`` (fusion v13 B3).

    Returns ``(frame_aug, w_aug, info)``: the originals (``is_aug=0``,
    ``aug_regime=""``) followed by the copies, with ``row_key`` on every row.
    A structured copy blanks the FULL gold block of every expert of its group
    (``blank_expert_blocks``: proj/avail/exact/traj/q/mapped_pred_class/...;
    ``q_prior__`` stays; ``traj_x__*`` recomputed from the surviving experts
    when the per-expert trajectory block is complete, else blanked) on the
    rows of ``spec.regime_surveys`` where at least one expert of the group
    fired; the random copy uses ``expert_dropout_augment(full_block=True)``.
    Weights: a copy takes its share of the source row's weight and the
    original keeps the remainder, so ``Σ w`` per object is unchanged
    (``info["object_weight_max_abs_delta"]`` records the check).
    """
    base_w = np.asarray(base_w, dtype=float)
    assert len(base_w) == len(frame), "base_w must align with frame rows"
    orig = _with_row_keys(frame)
    orig["is_aug"] = 0.0
    orig["aug_regime"] = ""
    orig["row_key"] = row_keys(orig, "")
    n = len(orig)
    shares = np.zeros(n, dtype=float)
    parts: list[pd.DataFrame] = []
    weights: list[np.ndarray] = []
    info: dict[str, Any] = {
        "spec": spec.to_dict(), "n_source_rows": int(n), "regimes": {},
        "traj_x": {},
    }
    sv = survey_series(orig)
    on_survey = np.isin(sv, [s.lower() for s in spec.regime_surveys])

    avail_cols = [c for c in orig.columns if c.startswith("avail__")]
    avail = (
        orig[avail_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        .to_numpy(float) > 0.5
    ) if avail_cols else np.zeros((n, 0), dtype=bool)

    for regime in spec.regimes:
        sans = _group_sans(_REGIME_GROUPS[regime], orig.columns)
        if not sans:
            info["regimes"][regime] = {"n_rows": 0, "reason": "no expert of the group in frame"}
            continue
        cols_idx = [avail_cols.index(f"avail__{s}") for s in sans]
        rows = on_survey & avail[:, cols_idx].any(axis=1)
        if not rows.any():
            info["regimes"][regime] = {"n_rows": 0, "reason": "no eligible rows"}
            continue
        cp = orig.loc[rows].copy()
        note = blank_expert_blocks(cp, np.ones(len(cp), dtype=bool), sans,
                                   recompute_cross_traj=True)
        cp["is_aug"] = 1.0
        cp["aug_regime"] = regime
        cp["row_key"] = row_keys(cp, regime)
        parts.append(cp)
        weights.append(base_w[rows] * float(spec.regime_weight))
        shares[rows] += float(spec.regime_weight)
        info["regimes"][regime] = {
            "n_rows": int(rows.sum()), "weight_share": float(spec.regime_weight),
            "n_experts_blanked": len(sans),
        }
        info["traj_x"][regime] = note["traj_x"]

    if spec.random_frac > 0 and avail_cols:
        aug_df, aug_w, ainfo = expert_dropout_augment(
            orig, base_w, aug_frac=float(spec.random_frac),
            aug_weight=spec.random_share, keep_one_frac=float(spec.keep_one_frac),
            seed=int(spec.seed), full_block=True,
        )
        if len(aug_df):
            aug_df["is_aug"] = 1.0
            aug_df["aug_regime"] = "random"
            aug_df["row_key"] = row_keys(aug_df, "random")
            parts.append(aug_df)
            weights.append(np.asarray(aug_w, dtype=float))
            shares[ainfo["source_positions"]] += spec.random_share
        info["regimes"]["random"] = {
            "n_rows": int(len(aug_df)), "weight_share": spec.random_share,
            "n_eligible": int(ainfo.get("n_eligible", 0)),
            "n_keep_one": int(ainfo.get("n_keep_one", 0)),
        }
        info["traj_x"]["random"] = ainfo.get("traj_x", "untouched")

    assert shares.max(initial=0.0) < 1.0, "dropout shares must leave the original weight > 0"
    w_orig = base_w * (1.0 - shares)
    frame_aug = pd.concat([orig] + parts, ignore_index=True, sort=False) if parts else orig
    w_aug = np.concatenate([w_orig] + weights) if parts else w_orig

    before = pd.Series(base_w).groupby(orig["object_id"].astype(str).to_numpy()).sum()
    after = pd.Series(w_aug).groupby(frame_aug["object_id"].astype(str).to_numpy()).sum()
    delta = float((after.reindex(before.index).fillna(0.0) - before).abs().max()) if n else 0.0
    info["object_weight_max_abs_delta"] = delta
    info["n_rows_after"] = int(len(frame_aug))
    return frame_aug, w_aug, info


def _weighted_corr(a: np.ndarray, y: np.ndarray, w: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    sw = float(w.sum())
    if sw <= 0:
        return None
    ma = float((w * a).sum() / sw)
    my = float((w * y).sum() / sw)
    va = float((w * (a - ma) ** 2).sum() / sw)
    vy = float((w * (y - my) ** 2).sum() / sw)
    if va < 1e-12 or vy < 1e-12:
        return None
    return float((w * (a - ma) * (y - my)).sum() / sw / np.sqrt(va * vy))


def availability_label_corr(
    frame: pd.DataFrame, y: np.ndarray, w: np.ndarray,
    *, surveys: Sequence[str] = _G8_SURVEYS,
) -> dict[str, dict[str, Any]]:
    """Per survey: weighted |corr(avail__<expert>, is_SN)| for every expert
    whose availability varies on that survey (guard G8, fusion v13)."""
    sv = survey_series(frame)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    out: dict[str, dict[str, Any]] = {}
    avail_cols = [c for c in frame.columns if c.startswith("avail__")]
    for survey in surveys:
        m = sv == survey
        entry: dict[str, Any] = {
            "n_rows": int(m.sum()),
            "n_objects": int(frame.loc[m, "object_id"].astype(str).nunique()) if m.any() else 0,
            "weight_sn_frac": (float((w[m] * y[m]).sum() / w[m].sum())
                               if m.any() and w[m].sum() > 0 else None),
            "per_expert": {},
        }
        if m.sum() >= 10 and len(np.unique(y[m])) >= 2:
            ym, wm = y[m], w[m]
            w_sn, w_ot = float((wm * ym).sum()), float((wm * (1 - ym)).sum())
            for c in avail_cols:
                a = pd.to_numeric(frame.loc[m, c], errors="coerce").fillna(0.0).to_numpy(float)
                r = _weighted_corr(a, ym, wm)
                if r is not None:
                    entry["per_expert"][c] = {
                        "corr": abs(r),
                        "avail_sn": float((wm * ym * a).sum() / w_sn) if w_sn > 0 else None,
                        "avail_other": float((wm * (1 - ym) * a).sum() / w_ot) if w_ot > 0 else None,
                    }
        out[survey] = entry
    return out


# ---------------------------------------------------------------------------
# The artifact / estimator
# ---------------------------------------------------------------------------

@dataclass
class HierarchicalFollowup:
    """Two-head hierarchical follow-up head (fusion_v11 P3).

    Construct -> :meth:`fit` -> :meth:`save`; or :meth:`load` a persisted one.
    Mirrors ``MulticlassFollowupArtifact``: ``predict_proba``/``predict_proba_raw``
    return ``(N, 3)`` in ``CLASSES`` order; ``feature_cols`` per head live in
    ``metadata.json``.
    """

    # --- hyperparameters (constructor) ---
    survey_cal_min: int = 40
    weak_weight: float = 0.1          # head-1 weak-SN weight (gated {0.1, 0.3})
    context_weight: float = 0.15
    tns_untyped_weight: float = 0.6
    bts_weight: float = 1.0
    head1_per_survey: bool = False    # preregistered default = POOLED
    head1_per_survey_min: int = 300   # per-survey realization needs this cal n
    head2_surveys: tuple[str, ...] = ("ztf",)   # ZTF-only start (B1)
    head2_lsst_min: int = 30          # LSST head-2 enable threshold
    base_rate_clip: tuple[float, float] = (0.05, 0.95)
    equalize_lsst_provenance: bool = True
    grid_small: bool = False
    n_jobs: int = 8
    seed: int = 42
    # --- fusion v13 options (defaults = v11/v12 behaviour) ---
    head1_exclude_qualities: tuple[str, ...] = ()      # B1, e.g. ("weak",)
    context_mask_scope: str = "rows"                   # B2: "rows" | "survey"
    head1_survey_masks: dict[str, tuple[str, ...]] = field(default_factory=dict)
    dropout: AvailabilityDropout | None = None         # B3
    cross_fit_folds: int = 0                           # B4 (0 = in-sample)
    g8_max_corr: float | None = None                   # G8 (None = disabled)
    g8_override: bool = False
    drop_experts: tuple[str, ...] = ()                 # global expert drop (both heads)
    # v13b: weights of the cross-fitted head-1 calibrators.  "train" = the head-1
    # training weights (v13); "object" = the same without the label-quality
    # factor (context 0.15, ...), so calibrated P(SN) follows the object mix
    # instead of the training emphasis (docs/fusion_v13_plan.md, Results).
    head1_cal_weights: str = "train"
    # v13d: feature-name prefixes kept out of both heads.  ``event_count__`` and
    # ``exact__`` only restate availability / pipeline bookkeeping; in v13b the
    # untimed local-expert rows put event_count__alerce_lc at ~160 on LSST
    # training rows against 8 at serving (docs/fusion_v13_plan.md, v13d).
    feature_drop_prefixes: tuple[str, ...] = ()
    # v13g: per-survey head-1 calibrator family override ({"lsst": "beta"}; kinds
    # in HEAD1_CALIBRATOR_KINDS).  A survey not named keeps the isotonic /
    # Platt rule on ``survey_cal_min``; the global calibrator is never overridden.
    head1_calibrator: dict[str, str] = field(default_factory=dict)

    # --- fitted state (populated by fit / load) ---
    head1_feature_cols: list[str] = field(default_factory=list)
    head2_feature_cols: list[str] = field(default_factory=list)
    head1_mode: str = "pooled"
    head1_bundles: dict[str, Any] = field(default_factory=dict)
    head2_bundle: Any = None
    head1_calibrators: dict[str, Any] = field(default_factory=dict)
    head1_global_calibrator: Any = None
    head2_calibrator: Any = None
    head1_calibrator_kinds: dict[str, str] = field(default_factory=dict)
    head2_calibrator_kind: str = "identity"
    base_rate: dict[str, float] = field(default_factory=dict)
    base_rate_global: float = 0.5
    head2_lsst_enabled: bool = False
    classes: tuple[str, ...] = CLASSES
    report_: dict[str, Any] = field(default_factory=dict)
    # --- fusion v13 fitted state (cross-fit) ---
    head1_serving_masks: dict[str, tuple[str, ...]] = field(default_factory=dict)  # resolved at fit; applied at serve
    head1_oof_: pd.Series | None = None      # row_key -> OOF raw P1 (train rows + copies)
    head2_oof_: pd.Series | None = None      # row_key -> OOF raw P2 (head-2 train rows + copies)
    oof_frame_: pd.DataFrame | None = None   # head-1 train mixture incl. p1_oof_raw, sample_weight (memory only)

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        df: pd.DataFrame,
        train_ids: Sequence,
        cal_ids: Sequence,
        test_ids: Sequence | None = None,
    ) -> "HierarchicalFollowup":
        """Fit both heads + per-head/per-survey calibration.

        Object-level splits are honored: heads fit on TRAIN rows only, per-head
        calibrators fit on CAL rows only.  Returns ``self``; a JSON-serializable
        fit report is stored on ``self.report_`` (the P6 orchestrator reads it).
        """
        train_ids = {str(i) for i in train_ids}
        cal_ids = {str(i) for i in cal_ids}
        test_ids = {str(i) for i in (test_ids or [])}
        assert not (train_ids & cal_ids), "train/cal overlap detected"
        assert not (train_ids & test_ids), "train/test overlap detected"
        assert not (cal_ids & test_ids), "cal/test overlap detected"

        labelled = df[df["target_class"].isin(CLASSES)].copy()
        labelled["object_id"] = labelled["object_id"].astype(str)
        if len(labelled) == 0:
            raise ValueError("No rows with a usable ternary target_class")

        oid = labelled["object_id"]
        train_df = labelled[oid.isin(train_ids)].copy()
        cal_df = labelled[oid.isin(cal_ids)].copy()
        if len(train_df) == 0:
            raise ValueError("No hierarchical follow-up training rows available")

        report: dict[str, Any] = {"gate_verdicts": []}
        if self.context_mask_scope not in ("rows", "survey"):
            raise ValueError("context_mask_scope must be 'rows' or 'survey'")
        bad = {k: v for k, v in self.head1_calibrator.items() if v not in HEAD1_CALIBRATOR_KINDS}
        if bad:
            raise ValueError(f"head1_calibrator kinds must be in {HEAD1_CALIBRATOR_KINDS}, got {bad}")
        report["v13"] = self._v13_settings()

        self._fit_head1(train_df, cal_df, report)
        self._fit_head2(train_df, cal_df, report)

        report["n_train_rows"] = int(len(train_df))
        report["n_cal_rows"] = int(len(cal_df))
        report["survey_cal_min"] = int(self.survey_cal_min)
        self.report_ = report
        return self

    def _v13_settings(self) -> dict[str, Any]:
        return {
            "head1_exclude_qualities": list(self.head1_exclude_qualities),
            "equalize_lsst_provenance": bool(self.equalize_lsst_provenance),
            "context_mask_scope": self.context_mask_scope,
            "head1_survey_masks": {k: list(v) for k, v in self.head1_survey_masks.items()},
            "dropout": self.dropout.to_dict() if self.dropout else None,
            "cross_fit_folds": int(self.cross_fit_folds),
            "g8_max_corr": self.g8_max_corr,
            "g8_override": bool(self.g8_override),
            "drop_experts": list(self.drop_experts),
            "head1_serving_masks": {k: list(v) for k, v in self.head1_serving_masks.items()},
            "head1_cal_weights": self.head1_cal_weights,
            **({"feature_drop_prefixes": list(self.feature_drop_prefixes)}
               if self.feature_drop_prefixes else {}),
            **({"head1_calibrator": dict(self.head1_calibrator)} if self.head1_calibrator else {}),
        }

    def _drop_prefixed(self, cols: list[str]) -> list[str]:
        if not self.feature_drop_prefixes:
            return cols
        return [c for c in cols if not c.startswith(tuple(self.feature_drop_prefixes))]

    # ------------------------------------------------ v13 serving-side masks

    def _serving_mask_frame(self, df: pd.DataFrame, *, head: str) -> pd.DataFrame:
        """The PERSISTENT masks a head sees identically at fit and at serve:
        ``drop_experts`` (both heads, every row; q_prior__ too) and, for head
        1, ``head1_serving_masks`` (per survey).  The label-provenance masks
        (``apply_provenance_masking`` / LSST equalization) are training-only
        by design and are NOT applied here.  Returns ``df`` itself when
        nothing is to be masked (the v11/v12 path stays byte-identical)."""
        from debass_meta.projectors.base import sanitize_expert_key

        drop = [sanitize_expert_key(k) for k in self.drop_experts]
        masks = self.head1_serving_masks if head == "head1" else {}
        if not drop and not masks:
            return df
        out = df.copy()
        if drop:
            blank_expert_blocks(out, np.ones(len(out), dtype=bool), drop,
                                recompute_cross_traj=True, include_q_prior=True)
        if masks:
            sv = survey_series(out)
            explicit = {str(k).lower(): {sanitize_expert_key(e) for e in v}
                        for k, v in self.head1_survey_masks.items()}
            for survey, experts in masks.items():
                rows = sv == str(survey).lower()
                if not rows.any():
                    continue
                keys = [sanitize_expert_key(k) for k in experts]
                # explicitly named masks (v13b, e.g. lsst:supernnova) blank the
                # q_prior too; the context-family masks keep it (v13 behaviour)
                named = [k for k in keys if k in explicit.get(str(survey).lower(), set())]
                rest = [k for k in keys if k not in named]
                if named:
                    blank_expert_blocks(out, rows, named, recompute_cross_traj=True,
                                        include_q_prior=True)
                if rest:
                    blank_expert_blocks(out, rows, rest, recompute_cross_traj=True)
        return out

    def _resolve_serving_masks(self, masked: pd.DataFrame) -> None:
        """Fix ``head1_serving_masks`` from the options + the training frame:
        explicit ``head1_survey_masks`` plus, with ``context_mask_scope ==
        "survey"``, the context family on every survey that has
        context-labelled rows."""
        masks: dict[str, list[str]] = {
            str(k).lower(): list(v) for k, v in self.head1_survey_masks.items()
        }
        if self.context_mask_scope == "survey":
            sv = survey_series(masked)
            lq = (
                masked["label_quality"].astype(str).str.lower().to_numpy()
                if "label_quality" in masked.columns else np.array([""] * len(masked))
            )
            for survey in sorted(set(sv.tolist())):
                if survey in ("", "nan"):
                    continue
                if ((sv == survey) & (lq == "context")).any():
                    masks.setdefault(survey, [])
                    masks[survey] = list(dict.fromkeys(masks[survey] + list(_CONTEXT_FAMILY)))
        self.head1_serving_masks = {k: tuple(v) for k, v in masks.items()}

    # ----------------------------------------------------------- head-1 fit

    def _prepare_head1(
        self, train_df: pd.DataFrame, cal_df: pd.DataFrame, report: dict[str, Any]
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
        """Everything before model fitting: B1 exclusion, provenance masking,
        (optional) equalization, serving masks, weights, feature discovery,
        the per-survey gate and the availability-dropout augmentation.
        Returns ``(frame, y, base_w, production, cal_df)``; ``production`` is
        the same source rows as seen at serving (serving masks, no
        provenance masks, no copies) — the G8 reference."""
        # B1 (v13): drop excluded label tiers from head-1 training AND from the
        # cal rows the head-1 calibrators / per-survey gate see.
        n_before = int(len(train_df))
        train_df = exclude_qualities(train_df, self.head1_exclude_qualities)
        cal_df = exclude_qualities(cal_df, self.head1_exclude_qualities)
        report["head1_excluded_train_rows"] = n_before - int(len(train_df))
        if len(train_df) == 0:
            raise ValueError("No head-1 training rows left after head1_exclude_qualities")

        # Provenance masking (reused from v8) + LSST equalization (spec §2.1 /
        # deviation #25): on LSST weak+context rows mask the ALeRCE family on
        # ALL such rows (not only alerce-labeled ones) so masking cannot create
        # a class-correlated availability pattern.
        #
        # v13 finding: that equalization is itself class-pure once the weak
        # rows are gone (every LSST catalogue-"other" row masked, no
        # spectroscopic SN row masked) and it hits the LOCAL `alerce_lc` and
        # the Rubin stamp — see multiclass_followup._ALERCE_FAMILY.  Switch it
        # off with ``equalize_lsst_provenance=False``; guard G8 checks the result.
        masked = apply_provenance_masking(train_df)
        masking_info = dict(masked.attrs.get("provenance_masking", {}))
        equalized = 0
        if self.equalize_lsst_provenance:
            sv = survey_series(masked)
            lq = (
                masked["label_quality"].astype(str).str.lower()
                if "label_quality" in masked.columns
                else pd.Series([""] * len(masked), index=masked.index)
            )
            lsst_weakish = pd.Series(
                (sv == "lsst") & lq.isin(("weak", "context")).to_numpy(),
                index=masked.index,
            )
            equalized = int(lsst_weakish.sum())
            _mask_expert_blocks(masked, lsst_weakish, _ALERCE_FAMILY)
        masking_info["n_lsst_equalized"] = equalized

        # v13 serving-side masks (applied identically at fit AND at serve, see
        # ``_serving_mask_frame``).  The label-source masking above blanks the
        # context family (lasair/sherlock, babamul) on catalogue-labelled rows
        # only; on LSST those are ALL the "other" rows, so `avail__babamul`
        # would be a perfect label proxy (v12 audit: corr 1.0).  With
        # ``context_mask_scope="survey"`` the family is masked on EVERY head-1
        # row of a survey that has context-labelled rows: the head can no
        # longer learn "context expert absent => other", at the price of not
        # using those experts on that survey (they are anchor-excluded anyway).
        # ``head1_survey_masks`` does the same for named experts whose
        # availability in the gold is a class-correlated pipeline artifact
        # (G8 names them); ``drop_experts`` removes an expert everywhere.
        self._resolve_serving_masks(masked)
        masked = self._serving_mask_frame(masked, head="head1")
        sv_m = survey_series(masked)
        masking_info["survey_masks"] = {
            survey: {"experts": list(experts), "n_rows": int((sv_m == survey).sum())}
            for survey, experts in self.head1_serving_masks.items()
        }
        masking_info["drop_experts"] = list(self.drop_experts)

        y = is_sn_target(masked)
        base_w = compute_base_weights(
            masked,
            weak_weight=self.weak_weight,
            context_weight=self.context_weight,
            tns_untyped_weight=self.tns_untyped_weight,
            bts_weight=self.bts_weight,
        )
        keep = base_w > 0
        frame = masked[keep].copy()
        y, base_w = y[keep], base_w[keep]
        if len(frame) == 0:
            raise ValueError("All head-1 training rows have zero weight")
        # production availability of the same rows (serving masks, no
        # provenance masks, no copies) — the G8 reference that tells
        # masking-induced from genuine correlation.
        production = self._serving_mask_frame(train_df, head="head1")[keep]

        self.head1_feature_cols = self._drop_prefixed(_numeric_feature_cols(frame))

        # Gate: per-survey vs pooled (default pooled when the LSST is_sn cal
        # frame is small — B1).  LSST cal-object count decides.
        lsst_cal_n = 0
        if len(cal_df):
            lsst_cal_n = int(
                cal_df.loc[survey_series(cal_df) == "lsst", "object_id"].nunique()
            )
        use_per_survey = bool(self.head1_per_survey) and (
            lsst_cal_n >= int(self.head1_per_survey_min)
        )
        self.head1_mode = "per_survey" if use_per_survey else "pooled"
        report["gate_verdicts"].append({
            "gate": "head1_per_survey_vs_pooled",
            "decision": self.head1_mode,
            "detail": f"lsst_is_sn_cal_objects={lsst_cal_n} "
                      f"(min={self.head1_per_survey_min}, "
                      f"per_survey_requested={self.head1_per_survey})",
        })
        report["head1_provenance_masking"] = masking_info

        # B3 (v13): availability-dropout copies (object totals unchanged).
        if self.dropout is not None:
            frame, base_w, aug_info = augment_availability(frame, base_w, self.dropout)
            y = is_sn_target(frame)
            report["head1_dropout"] = aug_info
        elif self.cross_fit_folds > 0:
            frame = _with_row_keys(frame)
        return frame, y, base_w, production, cal_df

    def g8_dry_run(
        self, df: pd.DataFrame, train_ids: Sequence, cal_ids: Sequence,
        *, threshold: float | None = None,
    ) -> dict[str, Any]:
        """Build the final head-1 frames with every v13 setting (exclusions,
        masks, drops, weights, dropout copies) and evaluate G8 on LSST and
        ZTF WITHOUT fitting anything.  Returns the G8 report (never raises);
        ``per_survey[<survey>]["per_expert"]`` carries every expert's final
        and production |corr| plus its weighted availability per class."""
        train_ids = {str(i) for i in train_ids}
        cal_ids = {str(i) for i in cal_ids}
        labelled = df[df["target_class"].isin(CLASSES)].copy()
        labelled["object_id"] = labelled["object_id"].astype(str)
        train_df = labelled[labelled["object_id"].isin(train_ids)].copy()
        cal_df = labelled[labelled["object_id"].isin(cal_ids)].copy()
        report: dict[str, Any] = {"gate_verdicts": [], "v13": self._v13_settings()}
        frame, y, w, production, _ = self._prepare_head1(train_df, cal_df, report)
        thr = float(self.g8_max_corr if threshold is None else threshold)
        g8 = self._evaluate_g8(frame, y, w, production, threshold=thr, raise_on_fail=False)
        g8["dry_run"] = True
        g8["n_head1_rows"] = int(len(frame))
        g8["head1_provenance_masking"] = report.get("head1_provenance_masking")
        g8["head1_dropout"] = {k: v for k, v in report.get("head1_dropout", {}).items()
                               if k != "spec"}
        g8["v13"] = report["v13"]
        return g8

    def _fit_head1(
        self, train_df: pd.DataFrame, cal_df: pd.DataFrame, report: dict[str, Any]
    ) -> None:
        frame, y, base_w, production, cal_df = self._prepare_head1(train_df, cal_df, report)
        use_per_survey = self.head1_mode == "per_survey"

        # G8 (v13): no expert's availability may be a label proxy on the FINAL
        # head-1 frame (after masking + augmentation), per survey.
        if self.g8_max_corr is not None:
            report["g8"] = self._evaluate_g8(frame, y, base_w, production)

        # B4 (v13): out-of-fold head-1 predictions for every train row/copy.
        if self.cross_fit_folds > 0:
            p1_oof, cf_info = self._cross_fit_binary(
                frame, y, base_w, self.head1_feature_cols, head="head1")
            self.head1_oof_ = pd.Series(p1_oof, index=frame["row_key"].to_numpy())
            oof_frame = frame.copy()
            oof_frame["p1_oof_raw"] = p1_oof
            oof_frame["sample_weight"] = base_w
            self.oof_frame_ = oof_frame
            report["cross_fit_head1"] = cf_info

        # Always train the pooled backbone (also the per-survey fallback).
        self.head1_bundles = {
            "__pooled__": _train_binary_head(
                frame, y, base_w, self.head1_feature_cols,
                params=_HEAD_PARAMS, n_jobs=self.n_jobs, seed=self.seed,
            )
        }
        if use_per_survey:
            sv = survey_series(frame)
            for survey in ("ztf", "lsst"):
                m = sv == survey
                if m.sum() >= 2 and len(np.unique(y[m])) >= 2:
                    self.head1_bundles[survey] = _train_binary_head(
                        frame[m], y[m], base_w[m], self.head1_feature_cols,
                        params=_HEAD_PARAMS, n_jobs=self.n_jobs, seed=self.seed,
                    )

        # Availability audit (deviation #25): dependence of each expert's
        # avail-pattern on y within LSST train rows.
        report["availability_audit"] = self._availability_audit(frame, y)

        # Per-survey head-1 calibration on CAL rows (isotonic / Platt); with
        # cross-fitting on OOF-train ∪ cal in the same regime mixture (B4).
        self._fit_head1_calibrators(cal_df, report)

    # ------------------------------------------------------------ v13 guards

    def _evaluate_g8(
        self, frame: pd.DataFrame, y: np.ndarray, w: np.ndarray,
        production: pd.DataFrame, *, threshold: float | None = None,
        raise_on_fail: bool = True,
    ) -> dict[str, Any]:
        thr = float(self.g8_max_corr if threshold is None else threshold)
        final = availability_label_corr(frame, y, w)
        # reference: same source rows, production availability, base weights
        y_prod = is_sn_target(production)
        w_prod = compute_base_weights(
            production, weak_weight=self.weak_weight, context_weight=self.context_weight,
            tns_untyped_weight=self.tns_untyped_weight, bts_weight=self.bts_weight)
        prod = availability_label_corr(production, y_prod, w_prod)
        violations: list[dict[str, Any]] = []
        for survey, entry in final.items():
            ref = prod.get(survey, {}).get("per_expert", {})
            for col, e in entry["per_expert"].items():
                r = e["corr"]
                pr = ref.get(col)
                entry["per_expert"][col] = {
                    "final": r, "production": None if pr is None else pr["corr"],
                    "avail_sn": e["avail_sn"], "avail_other": e["avail_other"],
                    "avail_sn_production": None if pr is None else pr["avail_sn"],
                    "avail_other_production": None if pr is None else pr["avail_other"],
                }
                if r > thr:
                    p_corr = None if pr is None else pr["corr"]
                    origin = ("masking_or_augmentation" if (p_corr is None or p_corr <= thr)
                              else "genuine_availability_gap_in_gold")
                    violations.append({"survey": survey, "expert": col,
                                       "corr_final": r, "corr_production": p_corr,
                                       "origin": origin})
        max_abs = max((v["corr_final"] for v in violations), default=None)
        if max_abs is None:
            vals = [r["final"] for e in final.values() for r in e["per_expert"].values()]
            max_abs = max(vals) if vals else None
        override = bool(self.g8_override) or not raise_on_fail
        status = "PASS" if not violations else ("OVERRIDDEN" if override else "FAIL")
        out = {"guard": "G8", "status": status, "threshold": thr,
               "max_abs_corr": max_abs, "violations": violations, "per_survey": final}
        if violations:
            lines = [
                f"  {v['survey']}: {v['expert']} |corr|={v['corr_final']:.3f} "
                f"(production {v['corr_production']}) — {v['origin']}"
                for v in violations]
            msg = (
                f"G8 {'overridden' if self.g8_override else 'violation'}: expert availability is a "
                f"label proxy on the head-1 train frame (threshold {thr}):\n" + "\n".join(lines)
                + "\n  masking_or_augmentation: the provenance/equalization masking is class-"
                "correlated on that survey — exclude the ALeRCE-labelled tier "
                "(head1_exclude_qualities), switch equalize_lsst_provenance off, or use "
                "context_mask_scope='survey'.\n  genuine_availability_gap_in_gold: the expert "
                "was only run on one class of objects — run it on the missing objects or "
                "mask it on that survey for head 1 (head1_survey_masks).")
            out["message"] = msg
            if not override:
                raise G8Error(msg)
        return out

    def _cross_fit_binary(
        self, frame: pd.DataFrame, y: np.ndarray, w: np.ndarray,
        feature_cols: list[str], *, head: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """GroupKFold (object-level; copies share their object) out-of-fold raw
        predictions; NaN where a fold could not be fit.  Fold models are the
        pooled realization (the per-survey arm, when gated on, only affects the
        deployed head)."""
        from debass_meta.models.folds import StableGroupKFold as GroupKFold  # CPU-independent folds (v13f)

        groups = frame["object_id"].astype(str).to_numpy()
        n_groups = len(np.unique(groups))
        k = int(min(self.cross_fit_folds, n_groups))
        oof = np.full(len(frame), np.nan)
        info: dict[str, Any] = {"head": head, "folds_requested": int(self.cross_fit_folds),
                                "folds_used": k, "n_rows": int(len(frame)),
                                "n_objects": int(n_groups)}
        if k < 2 or len(np.unique(y)) < 2:
            info["status"] = "skipped_too_few_groups_or_classes"
            return oof, info
        for i, (fit_idx, out_idx) in enumerate(GroupKFold(n_splits=k).split(frame, y, groups)):
            if len(np.unique(y[fit_idx])) < 2:
                oof[out_idx] = float(np.mean(y[fit_idx]))
                continue
            bundle = _train_binary_head(
                frame.iloc[fit_idx], y[fit_idx], w[fit_idx], feature_cols,
                params=_HEAD_PARAMS, n_jobs=self.n_jobs, seed=self.seed + i,
            )
            oof[out_idx] = _predict_binary(
                bundle, _prepare_frame(frame.iloc[out_idx], feature_cols))
        info["status"] = "ok"
        info["n_oof_finite"] = int(np.isfinite(oof).sum())
        return oof, info

    def _availability_audit(
        self, frame: pd.DataFrame, y: np.ndarray
    ) -> dict[str, Any]:
        sv = survey_series(frame)
        m = sv == "lsst"
        avail_cols = [c for c in frame.columns if c.startswith("avail__")]
        if m.sum() < 10 or not avail_cols or len(np.unique(y[m])) < 2:
            return {"n_lsst_rows": int(m.sum()), "max_abs_corr": None,
                    "per_expert": {}}
        yy = y[m].astype(float)
        per_expert: dict[str, float] = {}
        for c in avail_cols:
            a = (
                pd.to_numeric(frame.loc[m, c], errors="coerce")
                .fillna(0.0).to_numpy(float)
            )
            if np.std(a) < 1e-12:
                continue
            per_expert[c] = float(abs(np.corrcoef(a, yy)[0, 1]))
        max_abs = max(per_expert.values()) if per_expert else None
        return {
            "n_lsst_rows": int(m.sum()),
            "max_abs_corr": max_abs,
            "per_expert": per_expert,
        }

    def _fit_head1_calibrators(
        self, cal_df: pd.DataFrame, report: dict[str, Any]
    ) -> None:
        self.head1_calibrators = {}
        self.head1_calibrator_kinds = {}
        self.head1_global_calibrator = None
        if self.oof_frame_ is not None:
            self._fit_head1_calibrators_crossfit(cal_df, report)
            return
        if len(cal_df) == 0:
            return
        # Head-1 calibration must see PRODUCTION features (no train masking).
        p_raw = self._head1_raw(cal_df)
        y = is_sn_target(cal_df)
        sv = survey_series(cal_df)
        for survey in sorted(set(sv.tolist())):
            if survey in ("", "nan"):
                continue
            m = sv == survey
            cal = _fit_binary_calibrator(
                p_raw[m], y[m], n_min=self.survey_cal_min,
                kind=self.head1_calibrator.get(survey)
            )
            self.head1_calibrators[survey] = cal
            self.head1_calibrator_kinds[survey] = getattr(cal, "name", "identity")
        glob = _fit_binary_calibrator(p_raw, y, n_min=self.survey_cal_min)
        self.head1_global_calibrator = glob
        self.head1_calibrator_kinds["__global__"] = getattr(
            glob, "name", "identity"
        )

    def calibration_mixture(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Rows of ``df`` (production features) plus their availability-dropout
        copies under the fitted ``dropout`` spec, with object-normalized base
        weights rescaled so object totals are unchanged (v13: the mixture the
        head-1 calibrators and α see for cal rows).  Without a dropout spec the
        rows are returned as-is with their base weights."""
        w = compute_base_weights(
            df, weak_weight=self.weak_weight, context_weight=self.context_weight,
            tns_untyped_weight=self.tns_untyped_weight, bts_weight=self.bts_weight)
        if self.dropout is None or len(df) == 0:
            out = _with_row_keys(df)
            return out, w
        mix, w_mix, _ = augment_availability(df, w, self.dropout)
        return mix, w_mix

    def _quality_factor(self, df: pd.DataFrame) -> np.ndarray:
        """The label-quality factor of :func:`compute_base_weights` per row
        (spectroscopic 1, tns_untyped / context / weak per the head's
        weights; unknown = weak), so ``w / factor`` keeps only the object
        normalization, the BTS factor and the dropout shares."""
        qual_map = {"spectroscopic": 1.0, "tns_untyped": float(self.tns_untyped_weight),
                    "context": float(self.context_weight), "weak": float(self.weak_weight)}
        if "label_quality" not in df.columns:
            return np.ones(len(df), dtype=float)
        f = df["label_quality"].astype(str).str.lower().map(qual_map).fillna(float(self.weak_weight))
        return np.maximum(f.to_numpy(float), 1e-12)

    def _fit_head1_calibrators_crossfit(
        self, cal_df: pd.DataFrame, report: dict[str, Any]
    ) -> None:
        """B4: per-survey calibrators on OOF-train ∪ cal (the tiers head 1
        trains on, weighted; cal rows in the same availability mixture as
        train)."""
        oof = self.oof_frame_
        assert oof is not None
        keep = np.isfinite(oof["p1_oof_raw"].to_numpy(float))
        w_oof = oof.loc[keep, "sample_weight"].to_numpy(float)
        if self.head1_cal_weights == "object":
            w_oof = w_oof / self._quality_factor(oof[keep])
        parts = [pd.DataFrame({
            "p": oof.loc[keep, "p1_oof_raw"].to_numpy(float),
            "y": is_sn_target(oof[keep]),
            "w": w_oof,
            "survey": survey_series(oof[keep]),
            "source": "oof_train",
        })]
        cal_rows = cal_df  # already filtered by head1_exclude_qualities in _fit_head1
        if len(cal_rows):
            mix, w_mix = self.calibration_mixture(cal_rows)
            if self.head1_cal_weights == "object":
                w_mix = w_mix / self._quality_factor(mix)
            parts.append(pd.DataFrame({
                "p": self._head1_raw(mix), "y": is_sn_target(mix), "w": w_mix,
                "survey": survey_series(mix), "source": "cal",
            }))
        rows = pd.concat(parts, ignore_index=True)
        ledger: dict[str, Any] = {
            "excluded_qualities": list(self.head1_exclude_qualities),
            "weights": self.head1_cal_weights, "per_survey": {}}
        for survey in sorted(set(rows["survey"].tolist())):
            if survey in ("", "nan"):
                continue
            m = (rows["survey"] == survey).to_numpy()
            cal = _fit_binary_calibrator(
                rows.loc[m, "p"].to_numpy(), rows.loc[m, "y"].to_numpy(),
                n_min=self.survey_cal_min, sample_weight=rows.loc[m, "w"].to_numpy(),
                kind=self.head1_calibrator.get(survey))
            self.head1_calibrators[survey] = cal
            self.head1_calibrator_kinds[survey] = getattr(cal, "name", "identity")
            ledger["per_survey"][survey] = {
                "n_rows": int(m.sum()),
                "n_oof_train": int((rows.loc[m, "source"] == "oof_train").sum()),
                "n_cal": int((rows.loc[m, "source"] == "cal").sum()),
                "weight_sn_frac": float((rows.loc[m, "w"] * rows.loc[m, "y"]).sum()
                                        / max(rows.loc[m, "w"].sum(), 1e-12)),
            }
        glob = _fit_binary_calibrator(
            rows["p"].to_numpy(), rows["y"].to_numpy(), n_min=self.survey_cal_min,
            sample_weight=rows["w"].to_numpy())
        self.head1_global_calibrator = glob
        self.head1_calibrator_kinds["__global__"] = getattr(glob, "name", "identity")
        report["head1_calibration_crossfit"] = ledger

    # ----------------------------------------------------------- head-2 fit

    def _fit_head2(
        self, train_df: pd.DataFrame, cal_df: pd.DataFrame, report: dict[str, Any]
    ) -> None:
        # Per-survey base rate P(Ia|SN) on spec SN rows (clipped) — the LSST
        # default routing and the no-Ia-info fallback.
        self._fit_base_rates(train_df, cal_df, report)

        h2_mask = head2_training_mask(train_df, surveys=self.head2_surveys)
        h2_df = self._serving_mask_frame(train_df[h2_mask].copy(), head="head2")
        report["head2_n_train_rows"] = int(len(h2_df))
        report["head2_train_surveys"] = list(self.head2_surveys)

        # --- GUARD G7 (spec §4 / B0): every head-2 TRAINING row must carry
        # concrete subtype provenance.  On ``object_truth_v11.parquet`` the
        # BTS-untyped filler that force-mapped into the spec nonIa class is
        # demoted to the ``bts_untyped`` tier (label_quality != spectroscopic),
        # so it never reaches this mask; a hard assert here catches any frame
        # still built on the pre-B0 truth (where 61% of the spec nonIa class was
        # untyped filler).  Only evaluable when the provenance columns are
        # present — a synthetic frame lacking them records a not-evaluable note
        # rather than force-failing. ---
        if has_provenance_cols(h2_df):
            typed = typed_provenance_mask(h2_df)
            n_untyped = int((~typed).sum())
            report["g7"] = {
                "status": "enforced",
                "n_head2_rows": int(len(h2_df)),
                "n_untyped_provenance": n_untyped,
            }
            assert n_untyped == 0, (
                f"G7 violation: {n_untyped} head-2 training row(s) carry "
                "UNTYPED provenance (tns_type empty AND bts_type in "
                "{'-',''}). Head-2 is spec-only and presupposes "
                "data/truth/object_truth_v11.parquet, where BTS-untyped filler "
                "is demoted to the 'bts_untyped' tier (Head-1 only, B0). "
                "Rebuild gold with --truth object_truth_v11.parquet."
            )
        else:
            report["g7"] = {
                "status": "not_evaluable_no_provenance_cols",
                "n_head2_rows": int(len(h2_df)),
            }

        # Head-2 feature set: auto-discover on the (ZTF-only) spec SN frame,
        # then DROP survey_is_lsst by default.  Survey-degenerate columns are
        # already pruned (constant/all-NaN) by _numeric_feature_cols on this
        # single-survey frame.
        if len(h2_df) >= 2 and h2_df["target_class"].nunique() >= 1:
            cols = _numeric_feature_cols(h2_df)
        else:
            cols = []
        self.head2_feature_cols = [c for c in self._drop_prefixed(cols) if c not in _HEAD2_DROP_COLS]

        h2_oof_rows: pd.DataFrame | None = None
        if len(h2_df) >= 2:
            y2 = (h2_df["target_class"] == "snia").to_numpy().astype(int)
            base_w2 = compute_base_weights(
                h2_df,
                weak_weight=self.weak_weight,
                context_weight=self.context_weight,
                tns_untyped_weight=self.tns_untyped_weight,
                bts_weight=self.bts_weight,
            )
            keep = base_w2 > 0
            if keep.any() and self.head2_feature_cols:
                f2, w2 = h2_df[keep], base_w2[keep]
                y2k = y2[keep]
                # B3 (v13): the same availability dropout on head-2's own rows
                # (structured regimes only touch ``regime_surveys`` rows, so a
                # ZTF-only head-2 frame gets the random-subset copies).
                if self.dropout is not None:
                    f2, w2, aug2 = augment_availability(f2, w2, self.dropout)
                    y2k = (f2["target_class"] == "snia").to_numpy().astype(int)
                    report["head2_dropout"] = aug2
                elif self.cross_fit_folds > 0:
                    f2 = _with_row_keys(f2)
                # B4 (v13): OOF head-2 predictions (α / G2 honesty on train rows).
                if self.cross_fit_folds > 0:
                    p2_oof, cf2 = self._cross_fit_binary(
                        f2, y2k, w2, self.head2_feature_cols, head="head2")
                    self.head2_oof_ = pd.Series(p2_oof, index=f2["row_key"].to_numpy())
                    h2_oof_rows = pd.DataFrame({
                        "p": p2_oof, "y": y2k, "w": w2, "survey": survey_series(f2)})
                    report["cross_fit_head2"] = cf2
                self.head2_bundle = _train_binary_head(
                    f2, y2k, w2, self.head2_feature_cols,
                    params=_HEAD_PARAMS, n_jobs=self.n_jobs, seed=self.seed,
                )
        if self.head2_bundle is None:
            # No trainable head-2 -> constant global base rate.
            self.head2_bundle = {"kind": "constant", "proba": self.base_rate_global}

        # Head-2 covariate-shift calibrator: fit on ZTF spec SN cal rows, is_Ia
        # (v13 cross-fit: on OOF-train ∪ cal spec SN rows, weighted).
        self._fit_head2_calibrator(cal_df, report, oof_rows=h2_oof_rows)

        # head-2-on-LSST routing gate (default OFF; enable needs n>=30 AND
        # non-inversion on LSST spec cal/OOF).
        self._decide_head2_lsst(cal_df, train_df, report)

    def _fit_base_rates(
        self, train_df: pd.DataFrame, cal_df: pd.DataFrame, report: dict[str, Any]
    ) -> None:
        lo, hi = self.base_rate_clip
        combined = pd.concat(
            [d for d in (train_df, cal_df) if len(d)], ignore_index=True, sort=False
        )
        # Global base rate over spec SN rows (any survey), used as fallback.
        spec_sn = combined[
            head2_training_mask(combined, surveys=("ztf", "lsst"))
        ]
        if len(spec_sn):
            self.base_rate_global = float(
                np.clip((spec_sn["target_class"] == "snia").mean(), lo, hi)
            )
        else:
            self.base_rate_global = float(np.clip(0.5, lo, hi))

        self.base_rate = {}
        sv = survey_series(spec_sn)
        for survey in ("ztf", "lsst"):
            m = sv == survey
            if m.sum() >= 1:
                rate = float(
                    (spec_sn.loc[m, "target_class"] == "snia").mean()
                )
                self.base_rate[survey] = float(np.clip(rate, lo, hi))
            else:
                self.base_rate[survey] = self.base_rate_global
        report["base_rate"] = dict(self.base_rate)
        report["base_rate_global"] = self.base_rate_global

    def _fit_head2_calibrator(
        self, cal_df: pd.DataFrame, report: dict[str, Any],
        oof_rows: pd.DataFrame | None = None,
    ) -> None:
        self.head2_calibrator = _IdentityBinaryCalibrator()
        self.head2_calibrator_kind = "identity"
        if oof_rows is not None:
            self._fit_head2_calibrator_crossfit(cal_df, report, oof_rows)
            return
        if len(cal_df) == 0:
            return
        c_mask = head2_training_mask(cal_df, surveys=self.head2_surveys)
        c_df = cal_df[c_mask]
        if len(c_df) < 2:
            return
        p = _predict_binary(self.head2_bundle, self._X2(c_df))
        y = (c_df["target_class"] == "snia").to_numpy().astype(int)
        cal = _fit_binary_calibrator(p, y, n_min=self.survey_cal_min)
        self.head2_calibrator = cal
        self.head2_calibrator_kind = getattr(cal, "name", "identity")

    def _X2(self, df: pd.DataFrame) -> pd.DataFrame:
        """Head-2 design matrix with the persistent serving masks applied."""
        return _prepare_frame(self._serving_mask_frame(df, head="head2"),
                              self.head2_feature_cols)

    def _fit_head2_calibrator_crossfit(
        self, cal_df: pd.DataFrame, report: dict[str, Any], oof_rows: pd.DataFrame
    ) -> None:
        ok = np.isfinite(oof_rows["p"].to_numpy(float))
        parts = [oof_rows[ok].assign(source="oof_train")]
        if len(cal_df):
            c_df = cal_df[head2_training_mask(cal_df, surveys=self.head2_surveys)]
            if len(c_df):
                mix, w_mix = self.calibration_mixture(c_df)
                parts.append(pd.DataFrame({
                    "p": _predict_binary(self.head2_bundle, self._X2(mix)),
                    "y": (mix["target_class"] == "snia").to_numpy().astype(int),
                    "w": w_mix, "survey": survey_series(mix), "source": "cal",
                }))
        rows = pd.concat(parts, ignore_index=True)
        if len(rows) < 2:
            return
        cal = _fit_binary_calibrator(
            rows["p"].to_numpy(), rows["y"].to_numpy(), n_min=self.survey_cal_min,
            sample_weight=rows["w"].to_numpy())
        self.head2_calibrator = cal
        self.head2_calibrator_kind = getattr(cal, "name", "identity")
        report["head2_calibration_crossfit"] = {
            "n_rows": int(len(rows)),
            "n_oof_train": int((rows["source"] == "oof_train").sum()),
            "n_cal": int((rows["source"] == "cal").sum()),
        }

    def _decide_head2_lsst(
        self,
        cal_df: pd.DataFrame,
        train_df: pd.DataFrame,
        report: dict[str, Any],
    ) -> None:
        # Eligible frame = LSST spec SN cal rows (approx of cal/OOF spec-Ia).
        lsst_spec = cal_df[
            head2_training_mask(cal_df, surveys=("lsst",))
        ]
        n = int(len(lsst_spec))
        enabled = False
        spearman = None
        if n >= int(self.head2_lsst_min):
            p = _predict_binary(self.head2_bundle, self._X2(lsst_spec))
            y = (lsst_spec["target_class"] == "snia").to_numpy().astype(int)
            if len(np.unique(y)) >= 2 and np.std(p) > 1e-12:
                from scipy.stats import spearmanr

                rho = spearmanr(p, y).correlation
                spearman = None if rho is None or np.isnan(rho) else float(rho)
                # Non-inversion gate: shared head-2 must not be anti-correlated.
                enabled = spearman is not None and spearman >= 0.0
        self.head2_lsst_enabled = bool(enabled)
        report["gate_verdicts"].append({
            "gate": "head2_on_lsst",
            "decision": "shared_head2" if enabled else "constant_base_rate",
            "detail": f"lsst_spec_cal_n={n} (min={self.head2_lsst_min}), "
                      f"spearman={spearman}",
        })

    # -------------------------------------------------------------- predict

    def _head1_raw(self, df: pd.DataFrame) -> np.ndarray:
        df = self._serving_mask_frame(df, head="head1")
        X = _prepare_frame(df, self.head1_feature_cols)
        pooled = self.head1_bundles["__pooled__"]
        if self.head1_mode != "per_survey" or len(self.head1_bundles) == 1:
            return _predict_binary(pooled, X)
        sv = survey_series(df)
        p1 = np.empty(len(df), dtype=float)
        assigned = np.zeros(len(df), dtype=bool)
        for survey, bundle in self.head1_bundles.items():
            if survey == "__pooled__":
                continue
            m = sv == survey
            if m.any():
                p1[m] = _predict_binary(bundle, X.iloc[np.flatnonzero(m)])
                assigned[m] = True
        if (~assigned).any():
            p1[~assigned] = _predict_binary(
                pooled, X.iloc[np.flatnonzero(~assigned)]
            )
        return p1

    def _head1_p(self, df: pd.DataFrame, *, calibrated: bool) -> np.ndarray:
        p_raw = self._head1_raw(df)
        if not calibrated:
            return p_raw
        out = p_raw.copy()
        sv = survey_series(df)
        routed = np.zeros(len(df), dtype=bool)
        for survey, cal in self.head1_calibrators.items():
            m = sv == survey
            if m.any() and cal is not None:
                out[m] = cal.transform(p_raw[m])
                routed[m] = True
        if self.head1_global_calibrator is not None and (~routed).any():
            out[~routed] = self.head1_global_calibrator.transform(p_raw[~routed])
        return out

    def _head2_p(self, df: pd.DataFrame, *, calibrated: bool) -> np.ndarray:
        sv = survey_series(df)
        is_lsst = sv == "lsst"
        p2 = np.empty(len(df), dtype=float)

        # Rows routed through the shared head-2 model: everything that is not
        # LSST, plus LSST iff the gate enabled it.  (Use boolean logic, not
        # bitwise ops on the Python bool ``head2_lsst_enabled``.)
        enabled = bool(self.head2_lsst_enabled)
        model_mask = ~is_lsst if not enabled else np.ones(len(df), dtype=bool)
        if model_mask.any():
            X2 = self._X2(df.iloc[np.flatnonzero(model_mask)])
            pm = _predict_binary(self.head2_bundle, X2)
            if calibrated and self.head2_calibrator is not None:
                pm = self.head2_calibrator.transform(pm)
            p2[model_mask] = pm

        # LSST rows on the default route -> per-survey constant base rate.
        base_mask = is_lsst & (not enabled)
        if base_mask.any():
            p2[base_mask] = self.base_rate.get("lsst", self.base_rate_global)
        return p2

    @staticmethod
    def _compose(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        lo, hi = _HEAD_CLIP
        p1 = np.clip(np.asarray(p1, dtype=float), lo, hi)
        p2 = np.clip(np.asarray(p2, dtype=float), lo, hi)
        out = np.empty((len(p1), 3), dtype=float)
        out[:, 0] = p1 * p2            # snia
        out[:, 1] = p1 * (1.0 - p2)    # nonIa_snlike
        out[:, 2] = 1.0 - p1           # other
        return out

    def predict_proba_raw(self, df: pd.DataFrame) -> np.ndarray:
        """(N, 3) composition of the UNcalibrated heads (v8 scorer contract)."""
        p1 = self._head1_p(df, calibrated=False)
        p2 = self._head2_p(df, calibrated=False)
        return self._compose(p1, p2)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """(N, 3) composition of the per-head-calibrated heads."""
        p1 = self._head1_p(df, calibrated=True)
        p2 = self._head2_p(df, calibrated=True)
        return self._compose(p1, p2)

    # ------------------------------------------------------ v13 OOF predict

    def _oof_lookup(self, df: pd.DataFrame, table: pd.Series | None) -> np.ndarray:
        """OOF raw prediction per row of ``df`` (NaN where none): exact
        ``row_key`` first (train row or its dropout copy), else the row's
        ORIGINAL key (``object_id|n_det|``) so a copy the other head never
        made still gets an out-of-fold value."""
        out = np.full(len(df), np.nan)
        if table is None or len(table) == 0 or "n_det" not in df.columns:
            return out
        tbl = table[~table.index.duplicated(keep="first")]
        regime = df["aug_regime"] if "aug_regime" in df.columns else ""
        keys = row_keys(df, regime)
        hit = tbl.reindex(keys).to_numpy(float)
        miss = ~np.isfinite(hit)
        if miss.any():
            base = row_keys(df, "")
            hit_base = tbl.reindex(base).to_numpy(float)
            hit[miss] = hit_base[miss]
        return hit

    def oof_coverage(self, df: pd.DataFrame) -> dict[str, int]:
        """How many rows of ``df`` would get OOF head-1 / head-2 values."""
        return {
            "n_rows": int(len(df)),
            "n_oof_head1": int(np.isfinite(self._oof_lookup(df, self.head1_oof_)).sum()),
            "n_oof_head2": int(np.isfinite(self._oof_lookup(df, self.head2_oof_)).sum()),
        }

    def predict_proba_oof(self, df: pd.DataFrame) -> np.ndarray:
        """(N, 3) like :meth:`predict_proba`, but a row that was in a head's
        cross-fit frame (train row or dropout copy) uses that head's
        OUT-OF-FOLD raw prediction; every other row (e.g. cal) falls back to
        the deployed head.  Calibration and composition are the deployed ones.
        Without cross-fitting this is exactly ``predict_proba``."""
        p1_raw = self._head1_raw(df)
        o1 = self._oof_lookup(df, self.head1_oof_)
        p1_raw = np.where(np.isfinite(o1), o1, p1_raw)
        p1 = p1_raw.copy()
        sv = survey_series(df)
        routed = np.zeros(len(df), dtype=bool)
        for survey, cal in self.head1_calibrators.items():
            m = sv == survey
            if m.any() and cal is not None:
                p1[m] = cal.transform(p1_raw[m])
                routed[m] = True
        if self.head1_global_calibrator is not None and (~routed).any():
            p1[~routed] = self.head1_global_calibrator.transform(p1_raw[~routed])

        p2 = self._head2_p(df, calibrated=True)
        o2 = self._oof_lookup(df, self.head2_oof_)
        use = np.isfinite(o2)
        if use.any():
            # OOF raw P2 only where the deployed route is the shared head-2
            # (LSST rows on the base-rate route keep the base rate).
            is_lsst = sv == "lsst"
            model_route = ~is_lsst | bool(self.head2_lsst_enabled)
            use &= model_route
            o2c = o2[use]
            if self.head2_calibrator is not None:
                o2c = self.head2_calibrator.transform(o2c)
            p2[use] = o2c
        return self._compose(p1, p2)

    # --------------------------------------------------------------- io

    def save(self, out_dir: str) -> None:
        model_dir = Path(out_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "model.pkl", "wb") as fh:
            pickle.dump(
                {
                    "head1_bundles": self.head1_bundles,
                    "head2_bundle": self.head2_bundle,
                    "head1_mode": self.head1_mode,
                },
                fh,
            )
        with open(model_dir / "calibrator.pkl", "wb") as fh:
            pickle.dump(
                {
                    "head1_per_survey": self.head1_calibrators,
                    "head1_global": self.head1_global_calibrator,
                    "head2": self.head2_calibrator,
                },
                fh,
            )
        with open(model_dir / "metadata.json", "w") as fh:
            json.dump(self._metadata(), fh, indent=2)
        with open(model_dir / "fit_report.json", "w") as fh:
            json.dump(self.report_, fh, indent=2, default=_json_default)
        for name, table in (("head1_oof", self.head1_oof_), ("head2_oof", self.head2_oof_)):
            if table is not None and len(table):
                pd.DataFrame({"row_key": table.index.astype(str), "p_oof_raw": table.to_numpy(float)}) \
                    .to_parquet(model_dir / f"{name}.parquet", index=False)

    def _metadata(self) -> dict[str, Any]:
        return {
            "artifact": "HierarchicalFollowup",
            "classes": list(self.classes),
            "survey_cal_min": int(self.survey_cal_min),
            "head1_feature_cols": list(self.head1_feature_cols),
            "head2_feature_cols": list(self.head2_feature_cols),
            "head1_mode": self.head1_mode,
            "head2_surveys": list(self.head2_surveys),
            "head2_lsst_enabled": bool(self.head2_lsst_enabled),
            "head1_calibrator_kinds": dict(self.head1_calibrator_kinds),
            "head2_calibrator_kind": self.head2_calibrator_kind,
            "base_rate": dict(self.base_rate),
            "base_rate_global": float(self.base_rate_global),
            "base_rate_clip": list(self.base_rate_clip),
            "gate_verdicts": list(self.report_.get("gate_verdicts", [])),
            "hyperparameters": {
                "weak_weight": float(self.weak_weight),
                "context_weight": float(self.context_weight),
                "tns_untyped_weight": float(self.tns_untyped_weight),
                "bts_weight": float(self.bts_weight),
                "head1_per_survey": bool(self.head1_per_survey),
                "head1_per_survey_min": int(self.head1_per_survey_min),
                "head2_lsst_min": int(self.head2_lsst_min),
                "grid_small": bool(self.grid_small),
                "seed": int(self.seed),
                "equalize_lsst_provenance": bool(self.equalize_lsst_provenance),
            },
            "v13": self._v13_settings(),
        }

    @classmethod
    def load(cls, out_dir: str) -> "HierarchicalFollowup":
        model_dir = Path(out_dir)
        with open(model_dir / "metadata.json") as fh:
            meta = json.load(fh)
        with open(model_dir / "model.pkl", "rb") as fh:
            models = pickle.load(fh)
        head1_calibrators: dict[str, Any] = {}
        head1_global = None
        head2_calibrator = None
        cal_path = model_dir / "calibrator.pkl"
        if cal_path.exists():
            with open(cal_path, "rb") as fh:
                payload = pickle.load(fh)
            head1_calibrators = payload.get("head1_per_survey", {}) or {}
            head1_global = payload.get("head1_global")
            head2_calibrator = payload.get("head2")

        hp = meta.get("hyperparameters", {})
        v13 = dict(meta.get("v13", {}) or {})
        obj = cls(
            survey_cal_min=int(meta.get("survey_cal_min", 40)),
            weak_weight=float(hp.get("weak_weight", 0.1)),
            context_weight=float(hp.get("context_weight", 0.15)),
            tns_untyped_weight=float(hp.get("tns_untyped_weight", 0.6)),
            bts_weight=float(hp.get("bts_weight", 1.0)),
            head1_per_survey=bool(hp.get("head1_per_survey", False)),
            head1_per_survey_min=int(hp.get("head1_per_survey_min", 300)),
            head2_surveys=tuple(meta.get("head2_surveys", ("ztf",))),
            head2_lsst_min=int(hp.get("head2_lsst_min", 30)),
            base_rate_clip=tuple(meta.get("base_rate_clip", (0.05, 0.95))),
            grid_small=bool(hp.get("grid_small", False)),
            seed=int(hp.get("seed", 42)),
            equalize_lsst_provenance=bool(hp.get("equalize_lsst_provenance",
                                                 v13.get("equalize_lsst_provenance", True))),
            head1_exclude_qualities=tuple(v13.get("head1_exclude_qualities", ())),
            context_mask_scope=str(v13.get("context_mask_scope", "rows")),
            head1_survey_masks={k: tuple(v) for k, v in (v13.get("head1_survey_masks") or {}).items()},
            dropout=(AvailabilityDropout.from_dict(v13["dropout"]) if v13.get("dropout") else None),
            cross_fit_folds=int(v13.get("cross_fit_folds", 0)),
            g8_max_corr=v13.get("g8_max_corr"),
            g8_override=bool(v13.get("g8_override", False)),
            drop_experts=tuple(v13.get("drop_experts", ())),
            head1_cal_weights=str(v13.get("head1_cal_weights", "train")),
            feature_drop_prefixes=tuple(v13.get("feature_drop_prefixes", ())),
            head1_calibrator={str(k): str(v) for k, v in (v13.get("head1_calibrator") or {}).items()},
        )
        obj.head1_serving_masks = {
            str(k): tuple(v) for k, v in (v13.get("head1_serving_masks") or {}).items()}
        for name, attr in (("head1_oof", "head1_oof_"), ("head2_oof", "head2_oof_")):
            path = model_dir / f"{name}.parquet"
            if path.exists():
                t = pd.read_parquet(path)
                setattr(obj, attr, pd.Series(t["p_oof_raw"].to_numpy(float),
                                             index=t["row_key"].astype(str).to_numpy()))
        obj.head1_feature_cols = [str(c) for c in meta.get("head1_feature_cols", [])]
        obj.head2_feature_cols = [str(c) for c in meta.get("head2_feature_cols", [])]
        obj.head1_mode = str(meta.get("head1_mode", models.get("head1_mode", "pooled")))
        obj.head1_bundles = models["head1_bundles"]
        obj.head2_bundle = models["head2_bundle"]
        obj.head1_calibrators = head1_calibrators
        obj.head1_global_calibrator = head1_global
        obj.head2_calibrator = head2_calibrator
        obj.head1_calibrator_kinds = dict(meta.get("head1_calibrator_kinds", {}))
        obj.head2_calibrator_kind = str(meta.get("head2_calibrator_kind", "identity"))
        obj.base_rate = {str(k): float(v) for k, v in meta.get("base_rate", {}).items()}
        obj.base_rate_global = float(meta.get("base_rate_global", 0.5))
        obj.head2_lsst_enabled = bool(meta.get("head2_lsst_enabled", False))
        obj.classes = tuple(meta.get("classes", CLASSES))
        obj.report_ = {"gate_verdicts": meta.get("gate_verdicts", [])}
        report_path = model_dir / "fit_report.json"
        if report_path.exists():
            with open(report_path) as fh:
                obj.report_ = json.load(fh)
        return obj


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
