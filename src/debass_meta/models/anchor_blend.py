"""fusion_v11 anchored blend — trust-weighted ternary anchor + per-cell α blend.

The anchor is a *trust-weighted linear ternary pool of the fired experts*
(extends ``score_fusion_v8.trust_weighted_p_snia``), decomposed on two axes so
that only Ia-capable experts move the Ia|SN ratio:

  SN axis   P(SN) = Σ_e q_e·(p_snia^e + p_nonia^e) / Σ_e q_e      (all fired experts)
  Ia|SN     r     = Σ_e q_e·(p_snia^e/(p_snia^e+p_nonia^e)) / Σ_e q_e
                                                       (Ia-capable fired experts only)

  anchor = ( P(SN)·r ,  P(SN)·(1−r) ,  1−P(SN) )   →  ε-clipped (1e-6) + renorm.

``lasair/sherlock`` and ``babamul`` (context flags, no ternary SN claim) are
excluded from the anchor entirely.  When no Ia-capable expert fired the Ia axis
falls back to the per-survey calibration base rate P(Ia|SN), clipped [0.05,0.95].

Trust semantics (v8 precedent): SN-filter experts' q targets ``is_sn`` and
ternary experts' q targets ``is_topclass_correct`` — both pool on the SN axis;
the Ia|SN axis is weighted only by Ia-capable experts' q.

The deployed probabilities are ``p_final = α·p_model + (1−α)·p_anchor`` with α
chosen per fixed ``(survey × n_experts_fired{0,1,2+} × lc_cov{<0.25,≥0.25})`` cell
from the grid {0,.25,.5,.75,1} by cal log-loss + a 1-SE rule that *prefers the
anchor* (smaller α).  n_experts_fired==0 rows have no anchor and are pinned α≡1
(excluded from the G3 guard).  Out-of-support cells fall back cell → survey →
global → (α=0 if an anchor exists else 1).

G3 (§4): per fitted cell cal log-loss(blend) ≤ cal log-loss(ε-clipped anchor) is
guaranteed because α=0 (pure anchor) is in the grid and the fit takes the grid
minimum.  A post-fit per-survey pooled verification (log-loss AND macro-OvR AUC)
collapses a losing survey to a single α, then to α=0 (which terminates because
α=0 ⇒ blend==anchor).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from debass_meta.projectors.base import ALL_EXPERT_KEYS, sanitize_expert_key

try:  # canonical class order (snia, nonIa_snlike, other)
    from debass_meta.models.multiclass_followup import CLASSES
except Exception:  # pragma: no cover — parallel build
    CLASSES = ("snia", "nonIa_snlike", "other")

try:
    from debass_meta.features.lightcurve import FEATURE_NAMES as _BASE51
except Exception:  # pragma: no cover
    _BASE51 = []

EPS = 1e-6
ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
CELL_N_MIN = 50
BASE_RATE_CLIP = (0.05, 0.95)
LC_COV_EDGE = 0.25

# Anchor output column names (scorer uses p_snia/p_nonia/p_other spelling).
_ANCHOR_COLS = ("p_snia_anchor", "p_nonia_anchor", "p_other_anchor")
_CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}

# ── Ia-capability mask (§2.2 / deviation #11) ────────────────────────────────
# Experts whose projector moves the Ia|SN ratio with the input.  `fink/snn` is
# INCLUDED (strongest ZTF Ia broker; design omitted it); `fink_lsst/cats` is OUT:
# its projection always splits the SN share 50/50 between Ia and non-Ia, so it
# carries no Ia|SN information. (Before 2026-09-24 the projector misread CATS
# class 21 = Periodic as a non-Ia SN claim; see docs/metadebass_cats_bug.md.)
IA_CAPABLE: frozenset[str] = frozenset({
    "fink/snn",
    "fink/rf_ia",
    "fink_lsst/early_snia",
    "alerce/lc_classifier_transient",
    "alerce/lc_classifier_BHRF_forced_phot_transient",
    "alerce/LC_classifier_ATAT_forced_phot(beta)",
    "pittgoogle/supernnova_lsst",
    "pittgoogle/supernnova_ztf",
    "parsnip",
    "supernnova",
    "alerce_lc",
    "salt3_chi2",
    "lc_features_bv",
    "seq_v9",
    "seq_v11",
    "antares/oracle",
    "antares/superphot_plus",
    "ampel/parsnip_followme",
    "oracle_lsst",
})

# Context-only experts excluded from the anchor entirely (no ternary SN claim).
ANCHOR_EXCLUDED: frozenset[str] = frozenset({"lasair/sherlock", "babamul"})

# Registered by sibling package P5; tolerated as "pending" in the subset guard
# until the seq_v11 registry entry lands (cross-package ordering).
_PENDING_REGISTRATION: frozenset[str] = frozenset({"seq_v11"})

# Label sources that would make the α fit circular (B4 honesty filter; mirrors
# pooled_trust.py:369-375 — broker_consensus + alerce_self_label are dropped).
_CIRCULAR_LABEL_SOURCES = frozenset({"broker_consensus", "alerce_self_label"})


def assert_ia_capable_subset(
    registry_keys: list[str] | None = None, *, allow_pending: bool = True
) -> None:
    """Drift guard (§2.2): every IA_CAPABLE key must be a registered expert.

    ``allow_pending`` tolerates keys registered by a sibling package that may
    land after P4 (``seq_v11`` via P5); the integrated system registers them so
    the strict subset holds at train time.  A typo in any *other* key still
    fails loudly here and at train time.
    """
    keys = set(ALL_EXPERT_KEYS if registry_keys is None else registry_keys)
    missing = set(IA_CAPABLE) - keys
    if allow_pending:
        missing -= _PENDING_REGISTRATION
    if missing:
        raise AssertionError(
            f"IA_CAPABLE members not in expert registry: {sorted(missing)}"
        )


def _key_by_san() -> dict[str, str]:
    return {sanitize_expert_key(k): k for k in ALL_EXPERT_KEYS}


def _num(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _clip_renorm(mat: np.ndarray) -> np.ndarray:
    """ε-clip to [EPS, 1] then renormalize rows to sum 1; all-NaN rows kept NaN."""
    out = mat.copy()
    finite = np.isfinite(out).all(axis=1)
    sub = np.clip(out[finite], EPS, None)
    s = sub.sum(axis=1, keepdims=True)
    out[finite] = sub / s
    return out


# ── anchor ───────────────────────────────────────────────────────────────────

def compute_anchor(
    df: pd.DataFrame,
    *,
    base_rate_by_survey: dict[str, float] | None = None,
    default_base_rate: float = 0.5,
    ia_capable: frozenset[str] = IA_CAPABLE,
    exclude_experts: frozenset[str] | tuple[str, ...] = (),
    call_weighted_experts: frozenset[str] | tuple[str, ...] = (),
) -> pd.DataFrame:
    """Add ``p_{snia,nonia,other}_anchor`` and ``n_experts_fired`` to ``df``.

    An expert *fires* on a row when it is available (``avail__<san>`` truthy, or
    a finite ``proj__<san>__p_snia`` when no avail column exists), its ternary
    projection is finite, and its trust ``q__<san>`` is finite and > 0 (q
    defaults to 1 when the column is absent, matching ``trust_weighted_p_snia``).
    Rows with zero fired anchor-eligible experts get NaN anchor + n_experts_fired=0.
    ``exclude_experts`` (fusion v13 ``drop_experts``) never fire, whatever their
    columns hold — an expert dropped from the whole stack must not re-enter the
    anchor at serving with the q=1 default of a missing trust head.

    ``call_weighted_experts`` (fusion v13d, default none) are weighted by the
    trust of the call they make instead of q.  Their trust heads target
    ``is_sn``, so q estimates P(SN), not P(expert right): an SN-filter expert
    that correctly says "not SN" gets q near 0 and dropped out of the pool,
    while one wrongly saying "SN" kept its weight.  The call trust is q where
    the expert says SN (p_sn >= 0.5) and 1 - q where it says not SN.
    """
    df = df.copy()
    n = len(df)
    key_by_san = _key_by_san()
    excluded = set(ANCHOR_EXCLUDED) | {str(k) for k in exclude_experts}
    call_weighted = {str(k) for k in call_weighted_experts}
    sans = sorted({
        m.group(1) for c in df.columns
        for m in [re.match(r"proj__(.+)__p_snia$", c)] if m
    })

    num_sn = np.zeros(n)
    den_sn = np.zeros(n)
    n_fired = np.zeros(n, dtype=int)
    num_r = np.zeros(n)
    den_r = np.zeros(n)

    for san in sans:
        key = key_by_san.get(san)
        if key is None or key in excluded:
            continue
        p_snia = _num(df, f"proj__{san}__p_snia")
        p_nonia = _num(df, f"proj__{san}__p_nonIa_snlike")
        p_nonia = np.where(np.isfinite(p_nonia), p_nonia, 0.0)
        p_sn = p_snia + p_nonia
        q = _num(df, f"q__{san}")
        q = np.where(np.isfinite(q), q, 1.0) if f"q__{san}" in df.columns else np.ones(n)
        if key in call_weighted:
            q = np.where(p_sn >= 0.5, q, 1.0 - q)
        avail_col = f"avail__{san}"
        avail = (
            df[avail_col].fillna(0).astype(bool).to_numpy()
            if avail_col in df.columns else np.isfinite(p_snia)
        )
        fired = avail & np.isfinite(p_sn) & np.isfinite(q) & (q > 0)
        num_sn[fired] += q[fired] * p_sn[fired]
        den_sn[fired] += q[fired]
        n_fired[fired] += 1
        if key in ia_capable:
            denom = p_snia + p_nonia
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.where(denom > 0, p_snia / np.maximum(denom, 1e-12), np.nan)
            rf = fired & np.isfinite(r)
            num_r[rf] += q[rf] * r[rf]
            den_r[rf] += q[rf]

    with np.errstate(invalid="ignore", divide="ignore"):
        p_sn_anchor = np.where(den_sn > 0, num_sn / np.maximum(den_sn, 1e-12), np.nan)
        r_pool = np.where(den_r > 0, num_r / np.maximum(den_r, 1e-12), np.nan)

    # base-rate fallback for the Ia axis where no Ia-capable expert fired
    surveys = (
        df["survey"].astype(str).str.lower().to_numpy()
        if "survey" in df.columns else np.full(n, "unknown", dtype=object)
    )
    base = np.full(n, float(default_base_rate))
    if base_rate_by_survey:
        for sv, rate in base_rate_by_survey.items():
            base[surveys == str(sv).lower()] = float(rate)
    base = np.clip(base, *BASE_RATE_CLIP)
    r = np.where(np.isfinite(r_pool), r_pool, base)
    r = np.clip(r, 0.0, 1.0)

    p_snia_a = p_sn_anchor * r
    p_nonia_a = p_sn_anchor * (1.0 - r)
    p_other_a = 1.0 - p_sn_anchor
    anchor = _clip_renorm(np.column_stack([p_snia_a, p_nonia_a, p_other_a]))

    for i, col in enumerate(_ANCHOR_COLS):
        df[col] = anchor[:, i]
    df["n_experts_fired"] = n_fired
    return df


# ── blend spec ─────────────────────────────────────────────────────────────

def _n_experts_bucket(n: np.ndarray) -> np.ndarray:
    out = np.full(len(n), "2+", dtype=object)
    out[n == 0] = "0"
    out[n == 1] = "1"
    return out


def _lc_cov(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in _BASE51 if c in df.columns]
    if not cols:
        return np.zeros(len(df))
    block = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return np.isfinite(block).mean(axis=1)


def _lc_cov_bucket(cov: np.ndarray) -> np.ndarray:
    return np.where(cov < LC_COV_EDGE, "<0.25", ">=0.25").astype(object)


def cell_key(survey: str, n_exp_bucket: str, cov_bucket: str) -> str:
    return f"{survey}|{n_exp_bucket}|{cov_bucket}"


ALPHA_OBJECTIVES = ("multiclass", "sn_binary")
ALPHA_SE_UNITS = ("row", "object")
ALPHA_RULES = ("1se", "best")
_SN_INDEX = (_CLASS_INDEX["snia"], _CLASS_INDEX["nonIa_snlike"])


def _sample_losses(model: np.ndarray, anchor: np.ndarray, alpha: float, y: np.ndarray,
                   objective: str = "multiclass") -> np.ndarray:
    """Per-row log-loss of the α-blend.  ``sn_binary`` (fusion v13d) scores
    only the SN-vs-other axis, P(SN) = p_snia + p_nonia: the Ia|SN split is
    ignored, so it cannot drive α where the Ia axis carries no information."""
    blend = alpha * model + (1.0 - alpha) * anchor
    if objective == "sn_binary":
        p_sn = blend[:, list(_SN_INDEX)].sum(axis=1)
        p_true = np.where(np.isin(y, _SN_INDEX), p_sn, 1.0 - p_sn)
    else:
        p_true = blend[np.arange(len(y)), y]
    return -np.log(np.clip(p_true, EPS, 1.0))


def _fit_alpha_1se(
    model: np.ndarray, anchor: np.ndarray, y: np.ndarray,
    w: np.ndarray | None = None,
    *,
    objective: str = "multiclass",
    groups: np.ndarray | None = None,
    rule: str = "1se",
) -> tuple[float, dict]:
    """Grid α by mean cal log-loss; 1-SE rule preferring the anchor (smaller α).

    ``w`` (fusion v13, optional) weights the rows: the mean becomes the
    weighted mean and the SE its linearization ``sqrt(Σ w²(l-μ)²)/Σw``.  With
    ``w=None`` the arithmetic is the original unweighted one.

    fusion v13d: ``objective`` picks the loss (:func:`_sample_losses`);
    ``groups`` (e.g. object ids) makes the SE cluster-robust,
    ``sqrt(Σ_g (Σ_{i∈g} w_i (l_i-μ))²)/Σw``, since the rows of one object
    (its epochs and availability copies) are not independent.  ``rule="best"``
    takes the grid minimum (the SE is still reported): with few objects per
    cell a clustered SE makes the 1-SE rule fall back to the anchor even where
    the out-of-fold loss clearly prefers the model (fusion v13d)."""
    loss = lambda a: _sample_losses(model, anchor, a, y, objective)  # noqa: E731
    if w is None and groups is None:
        means = {a: float(loss(a).mean()) for a in ALPHA_GRID}
        best = min(means, key=lambda a: means[a])
        losses_best = loss(best)
        se = float(losses_best.std(ddof=1) / np.sqrt(len(y))) if len(y) > 1 else 0.0
    else:
        w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
        sw = float(w.sum())
        means = {a: float((w * loss(a)).sum() / sw) for a in ALPHA_GRID}
        best = min(means, key=lambda a: means[a])
        dev = w * (loss(best) - means[best])
        if groups is not None:
            dev = pd.Series(dev).groupby(pd.Series(np.asarray(groups)).to_numpy()).sum().to_numpy()
        se = float(np.sqrt((dev ** 2).sum()) / sw) if len(y) > 1 else 0.0
    thresh = means[best] + se
    candidates = [a for a in ALPHA_GRID if means[a] <= thresh + 1e-12]
    chosen = best if rule == "best" else min(candidates)  # α=0 == pure anchor
    info = {
        "means": means,
        "best_grid": best,
        "se": se,
        "loss_blend": means[chosen],
        "loss_anchor": means[0.0],
    }
    return chosen, info


_V13D_DEFAULTS: dict[str, Any] = {"anchor_call_experts": [], "alpha_objective": "multiclass",
                                  "alpha_se": "row", "base_rate_unit": "row", "alpha_rule": "1se"}


@dataclass
class BlendSpec:
    """Serialized α table + anchor base rates.  → ``blend.json``."""

    alpha_cells: dict[str, dict[str, Any]] = field(default_factory=dict)
    alpha_survey: dict[str, dict[str, Any]] = field(default_factory=dict)
    alpha_global: dict[str, Any] = field(default_factory=lambda: {"alpha": 0.0, "n": 0})
    base_rates: dict[str, float] = field(default_factory=dict)
    default_base_rate: float = 0.5
    grid: tuple[float, ...] = ALPHA_GRID
    bucket_edges: dict[str, Any] = field(default_factory=lambda: {
        "n_experts": ["0", "1", "2+"], "lc_cov": LC_COV_EDGE,
    })
    n_min: int = CELL_N_MIN
    g3: dict[str, Any] = field(default_factory=dict)
    drop_experts: list[str] = field(default_factory=list)   # fusion v13
    # fusion v13d (defaults = v11..v13c behaviour; see fit_alpha)
    anchor_call_experts: list[str] = field(default_factory=list)
    alpha_objective: str = "multiclass"
    alpha_se: str = "row"
    base_rate_unit: str = "row"
    alpha_rule: str = "1se"

    def to_dict(self) -> dict[str, Any]:
        d = {
            "alpha_cells": self.alpha_cells,
            "alpha_survey": self.alpha_survey,
            "alpha_global": self.alpha_global,
            "base_rates": self.base_rates,
            "default_base_rate": self.default_base_rate,
            "grid": list(self.grid),
            "bucket_edges": self.bucket_edges,
            "n_min": self.n_min,
            "g3": self.g3,
        }
        if self.drop_experts:  # v11/v12 blend.json stays byte-identical
            d["drop_experts"] = list(self.drop_experts)
        v13d = {k: v for k, v in (("anchor_call_experts", list(self.anchor_call_experts)),
                                  ("alpha_objective", self.alpha_objective),
                                  ("alpha_se", self.alpha_se),
                                  ("base_rate_unit", self.base_rate_unit),
                                  ("alpha_rule", self.alpha_rule))
                if v != _V13D_DEFAULTS[k]}
        if v13d:
            d["v13d"] = v13d
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BlendSpec":
        return cls(
            alpha_cells=d.get("alpha_cells", {}),
            alpha_survey=d.get("alpha_survey", {}),
            alpha_global=d.get("alpha_global", {"alpha": 0.0, "n": 0}),
            base_rates=d.get("base_rates", {}),
            default_base_rate=float(d.get("default_base_rate", 0.5)),
            grid=tuple(d.get("grid", ALPHA_GRID)),
            bucket_edges=d.get("bucket_edges", {}),
            n_min=int(d.get("n_min", CELL_N_MIN)),
            g3=d.get("g3", {}),
            drop_experts=[str(k) for k in d.get("drop_experts", [])],
            anchor_call_experts=[str(k) for k in (d.get("v13d") or {}).get("anchor_call_experts", [])],
            alpha_objective=str((d.get("v13d") or {}).get("alpha_objective", "multiclass")),
            alpha_se=str((d.get("v13d") or {}).get("alpha_se", "row")),
            base_rate_unit=str((d.get("v13d") or {}).get("base_rate_unit", "row")),
            alpha_rule=str((d.get("v13d") or {}).get("alpha_rule", "1se")),
        )

    def anchor_kwargs(self) -> dict[str, Any]:
        """``compute_anchor`` keyword arguments that reproduce this spec's anchor."""
        return {"base_rate_by_survey": self.base_rates,
                "default_base_rate": self.default_base_rate,
                "exclude_experts": tuple(self.drop_experts),
                "call_weighted_experts": tuple(self.anchor_call_experts)}

    def save(self, out_dir: str | Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "blend.json"
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)
        return path

    @classmethod
    def load(cls, in_dir: str | Path) -> "BlendSpec":
        path = Path(in_dir)
        if path.is_dir():
            path = path / "blend.json"
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def fitted_surveys(self) -> set[str]:
        """Surveys that contributed rows to the α fit (cell or survey rungs)."""
        out = {str(k) for k in self.alpha_survey}
        for ck in self.alpha_cells:
            out.add(str(ck).split("|", 1)[0])
        return out

    # -- α lookup with fallback ladder ------------------------------------
    def lookup_alpha(self, survey: str, n_exp_bucket: str, cov_bucket: str,
                     has_anchor: bool) -> tuple[float, str]:
        if n_exp_bucket == "0" or not has_anchor:
            return 1.0, "no_anchor"
        ck = cell_key(survey, n_exp_bucket, cov_bucket)
        if ck in self.alpha_cells:
            return float(self.alpha_cells[ck]["alpha"]), "cell"
        if survey in self.alpha_survey:
            return float(self.alpha_survey[survey]["alpha"]), "survey"
        # The global rung is survey-guarded: a survey the α fit never saw
        # (e.g. LSST when cal was ZTF-only) must not inherit another survey's
        # α — with an anchor present it falls back to the anchor (α=0), which
        # is the G3-safe terminal default for out-of-support rows.
        if self.alpha_global and survey in self.fitted_surveys():
            return float(self.alpha_global.get("alpha", 0.0)), "global"
        return 0.0, "anchor_default"


def _honesty_mask(
    df: pd.DataFrame, *, exclude_qualities: tuple[str, ...] = ()
) -> np.ndarray:
    """B4 filter: drop broker_consensus / alerce_self_label rows and rows whose
    label_source is an anchor-member expert (never fit α on a broker's own labels).

    ``exclude_qualities`` (fusion v13 B1, default none) additionally drops rows
    whose ``label_quality`` is in the tuple — e.g. ``("weak",)`` keeps every
    stamp-derived weak label out of the α fit, whatever its label_source."""
    keep = np.ones(len(df), dtype=bool)
    if "label_source" in df.columns:
        ls = df["label_source"].astype(str)
        member_keys = set(ALL_EXPERT_KEYS) - set(ANCHOR_EXCLUDED)
        drop = set(_CIRCULAR_LABEL_SOURCES) | member_keys
        keep &= ~ls.isin(drop).to_numpy()
    keep &= ~_excluded_quality_rows(df, exclude_qualities)
    return keep


def _excluded_quality_rows(df: pd.DataFrame, exclude_qualities: tuple[str, ...]) -> np.ndarray:
    """Rows matching ``"<quality>"`` (any survey) or ``"<survey>:<quality>"``
    specs — the same grammar as ``hierarchical_followup.excluded_quality_mask``."""
    drop = np.zeros(len(df), dtype=bool)
    if not exclude_qualities or "label_quality" not in df.columns:
        return drop
    lq = df["label_quality"].astype(str).str.lower().to_numpy()
    sv = (df["survey"].astype(str).str.lower().to_numpy() if "survey" in df.columns
          else np.full(len(df), "", dtype=object))
    for spec in exclude_qualities:
        spec = str(spec).strip().lower()
        if not spec:
            continue
        survey, _, qual = spec.rpartition(":")
        m = lq == qual
        if survey:
            m &= sv == survey
        drop |= m
    return drop


def _label_index(df: pd.DataFrame) -> np.ndarray:
    y = df["target_class"].astype(str).map(_CLASS_INDEX)
    return y.to_numpy()


def _macro_ovr_auc(
    y: np.ndarray, proba: np.ndarray, w: np.ndarray | None = None
) -> float | None:
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:  # pragma: no cover
        return None
    present = np.unique(y)
    if len(present) < 2:
        return None
    kw = {} if w is None else {"sample_weight": np.asarray(w, dtype=float)}
    try:
        return float(roc_auc_score(y, proba, multi_class="ovr", average="macro",
                                   labels=list(range(len(CLASSES))), **kw))
    except Exception:
        aucs = []
        for c in range(len(CLASSES)):
            yc = (y == c).astype(int)
            if yc.min() == yc.max():
                continue
            aucs.append(roc_auc_score(yc, proba[:, c], **kw))
        return float(np.mean(aucs)) if aucs else None


def fit_alpha(
    cal_df: pd.DataFrame,
    *,
    out_dir: str | Path | None = None,
    default_base_rate: float = 0.5,
    apply_honesty: bool = True,
    model_cols: tuple[str, str, str] = ("p_snia", "p_nonia", "p_other"),
    exclude_qualities: tuple[str, ...] = (),
    weight_col: str | None = None,
    drop_experts: tuple[str, ...] = (),
    anchor_call_experts: tuple[str, ...] = (),
    alpha_objective: str = "multiclass",
    alpha_se: str = "row",
    base_rate_unit: str = "row",
    alpha_rule: str = "1se",
    original_rows_surveys: tuple[str, ...] = (),
) -> BlendSpec:
    """Fit the per-cell α table on a calibration frame.

    ``cal_df`` must carry the model probabilities (``model_cols``), the
    ``proj__*``/``avail__*``/``q__*`` expert columns, ``survey``,
    ``target_class`` and (optionally) ``label_source``.  The anchor is computed
    internally from a per-survey base rate estimated on true-SN rows.

    fusion v13 (both default to the original behaviour):
    ``exclude_qualities`` extends the honesty filter to whole label tiers
    (B1: ``("lsst:weak",)`` = LSST weak rows only, ``("weak",)`` = every
    survey); ``weight_col`` names a per-row weight used by the
    grid/1-SE/verification log-losses and AUCs (B4: the OOF-train ∪ cal frame
    carries object-normalized base weights and availability-dropout shares) —
    cell/survey/global ``n`` stay ROW counts, so ``CELL_N_MIN`` keeps its
    meaning.  ``spec.g3["fit_frame"]`` records both settings.  ``drop_experts``
    are excluded from the anchor here AND persisted on the spec so
    :func:`apply` excludes them at serving too.

    fusion v13d (defaults = the v13c behaviour, all persisted on the spec):
    ``anchor_call_experts`` are weighted in the anchor by the trust of their
    call (:func:`compute_anchor`); ``alpha_objective="sn_binary"`` fits α (and
    runs the per-survey verification) on the SN-vs-other log-loss and AUC;
    ``alpha_se="object"`` clusters the 1-SE rule's SE by ``object_id``;
    ``base_rate_unit="object"`` estimates the anchor's P(Ia|SN) fallback with
    one vote per object instead of per row (objects with more epochs no longer
    dominate it); ``alpha_rule="best"`` takes the grid minimum instead of the
    1-SE rule.

    fusion v13h: rows of the surveys in ``original_rows_surveys`` that are
    availability-dropout copies (``is_aug`` > 0) are dropped before anything
    is fitted, so those surveys' α, base rates and verification see real rows
    only.  Serving rows are real rows; a missing input already moves a row to
    another n_experts / lc_cov cell.  Recorded in ``spec.g3["fit_frame"]``.
    """
    if alpha_objective not in ALPHA_OBJECTIVES:
        raise ValueError(f"alpha_objective must be one of {ALPHA_OBJECTIVES}")
    if alpha_rule not in ALPHA_RULES:
        raise ValueError(f"alpha_rule must be one of {ALPHA_RULES}")
    if alpha_se not in ALPHA_SE_UNITS or base_rate_unit not in ALPHA_SE_UNITS:
        raise ValueError(f"alpha_se / base_rate_unit must be one of {ALPHA_SE_UNITS}")
    df = cal_df.copy()
    hon = (_honesty_mask(df, exclude_qualities=tuple(exclude_qualities))
           if apply_honesty else np.ones(len(df), dtype=bool))
    if not apply_honesty and exclude_qualities:
        hon &= ~_excluded_quality_rows(df, tuple(exclude_qualities))
    df = df[hon].reset_index(drop=True)
    orig_sv = sorted({str(s).lower() for s in original_rows_surveys})
    n_copies_dropped = 0
    if orig_sv:
        if "is_aug" not in df.columns:
            raise KeyError("fit_alpha: original_rows_surveys needs an is_aug column")
        sv_col = (df["survey"].astype(str).str.lower() if "survey" in df.columns
                  else pd.Series("unknown", index=df.index))
        copy = ((pd.to_numeric(df["is_aug"], errors="coerce").fillna(0.0) > 0)
                & sv_col.isin(orig_sv)).to_numpy()
        n_copies_dropped = int(copy.sum())
        df = df[~copy].reset_index(drop=True)
    w_all: np.ndarray | None = None
    if weight_col is not None:
        if weight_col not in df.columns:
            raise KeyError(f"fit_alpha: weight_col {weight_col!r} not in frame")
        w_all = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).to_numpy(float)

    # per-survey base rate P(Ia|SN) on true-SN rows (post honesty filter)
    surveys = df["survey"].astype(str).str.lower().to_numpy() if "survey" in df.columns \
        else np.full(len(df), "unknown", dtype=object)
    tc = df["target_class"].astype(str).to_numpy()
    is_sn = np.isin(tc, ["snia", "nonIa_snlike"])
    is_ia = tc == "snia"
    base_rates: dict[str, float] = {}
    for sv in np.unique(surveys):
        m = (surveys == sv) & is_sn
        if m.sum() > 0:
            if base_rate_unit == "object" and "object_id" in df.columns:
                rate = pd.Series(is_ia[m]).groupby(
                    df.loc[m, "object_id"].astype(str).to_numpy()).first().mean()
            else:
                rate = is_ia[m].mean()
            base_rates[sv] = float(np.clip(rate, *BASE_RATE_CLIP))

    df = compute_anchor(df, base_rate_by_survey=base_rates,
                        default_base_rate=default_base_rate,
                        exclude_experts=tuple(drop_experts),
                        call_weighted_experts=tuple(anchor_call_experts))

    model = np.column_stack([_num(df, c) for c in model_cols])
    anchor = df[list(_ANCHOR_COLS)].to_numpy(dtype=float)
    y = _label_index(df)

    n_exp = df["n_experts_fired"].to_numpy()
    n_exp_b = _n_experts_bucket(n_exp)
    cov_b = _lc_cov_bucket(_lc_cov(df))

    # rows usable for α fit: valid model + anchor + label, and n_experts>=1
    usable = (
        np.isfinite(model).all(axis=1)
        & np.isfinite(anchor).all(axis=1)
        & np.isfinite(y.astype(float))
        & (n_exp >= 1)
    )
    if w_all is not None:
        usable &= w_all > 0

    def _w(m: np.ndarray) -> np.ndarray | None:
        return None if w_all is None else w_all[m]

    groups_all = (df["object_id"].astype(str).to_numpy()
                  if alpha_se == "object" and "object_id" in df.columns else None)

    def _fit(m: np.ndarray) -> tuple[float, dict]:
        return _fit_alpha_1se(model[m], anchor[m], y[m], _w(m), objective=alpha_objective,
                              groups=None if groups_all is None else groups_all[m], rule=alpha_rule)

    spec = BlendSpec(base_rates=base_rates, default_base_rate=default_base_rate,
                     drop_experts=[str(k) for k in drop_experts],
                     anchor_call_experts=sorted(str(k) for k in anchor_call_experts),
                     alpha_objective=alpha_objective, alpha_se=alpha_se,
                     base_rate_unit=base_rate_unit, alpha_rule=alpha_rule)
    g3_cells: dict[str, Any] = {}

    # per-cell fits
    for sv in np.unique(surveys):
        for neb in ("1", "2+"):
            for cb in ("<0.25", ">=0.25"):
                m = usable & (surveys == sv) & (n_exp_b == neb) & (cov_b == cb)
                if m.sum() < CELL_N_MIN:
                    continue
                a, info = _fit(m)
                ck = cell_key(sv, neb, cb)
                spec.alpha_cells[ck] = {"alpha": a, "n": int(m.sum()),
                                        "grid_best": info["best_grid"]}
                if (alpha_objective, alpha_se, alpha_rule) != ("multiclass", "row", "1se"):
                    spec.alpha_cells[ck]["loss_by_alpha"] = {
                        str(k): round(v, 6) for k, v in info["means"].items()}
                    spec.alpha_cells[ck]["se"] = round(info["se"], 6)
                g3_cells[ck] = {
                    "loss_blend": info["loss_blend"],
                    "loss_anchor": info["loss_anchor"],
                    "pass": info["loss_blend"] <= info["loss_anchor"] + 1e-12,
                }

    # per-survey fallback fits
    for sv in np.unique(surveys):
        m = usable & (surveys == sv)
        if m.sum() >= 1:
            a, info = _fit(m)
            spec.alpha_survey[sv] = {"alpha": a, "n": int(m.sum())}

    # global fallback fit
    if usable.sum() >= 1:
        a, _ = _fit(usable)
        spec.alpha_global = {"alpha": a, "n": int(usable.sum())}

    # ── post-fit per-survey pooled verification (G3) ─────────────────────
    verify: dict[str, Any] = {}
    for sv in np.unique(surveys):
        m = usable & (surveys == sv)
        if m.sum() < 2:
            continue
        alphas = np.array([
            spec.lookup_alpha(sv, neb, cb, True)[0]
            for neb, cb in zip(n_exp_b[m], cov_b[m])
        ])[:, None]
        blend = alphas * model[m] + (1.0 - alphas) * anchor[m]
        wm = _w(m)
        ll_blend = float(_neg_logloss(blend, y[m], wm, alpha_objective))
        ll_anchor = float(_neg_logloss(anchor[m], y[m], wm, alpha_objective))
        auc_blend = _verify_auc(y[m], blend, wm, alpha_objective)
        auc_anchor = _verify_auc(y[m], anchor[m], wm, alpha_objective)
        ok = ll_blend <= ll_anchor + 1e-9 and (
            auc_blend is None or auc_anchor is None or auc_blend >= auc_anchor - 1e-9
        )
        verdict = "cell"
        if not ok and sv in spec.alpha_survey:
            # collapse to single per-survey α
            a_sv = spec.alpha_survey[sv]["alpha"]
            blend_sv = a_sv * model[m] + (1.0 - a_sv) * anchor[m]
            ll_sv = float(_neg_logloss(blend_sv, y[m], wm, alpha_objective))
            auc_sv = _verify_auc(y[m], blend_sv, wm, alpha_objective)
            if ll_sv <= ll_anchor + 1e-9 and (
                auc_sv is None or auc_anchor is None or auc_sv >= auc_anchor - 1e-9
            ):
                # force all cells of this survey to the survey α
                for ck in list(spec.alpha_cells):
                    if ck.startswith(f"{sv}|"):
                        spec.alpha_cells[ck]["alpha"] = a_sv
                        spec.alpha_cells[ck]["collapsed"] = "survey"
                verdict = "collapsed_survey"
            else:
                # terminate at α=0 (blend == anchor)
                spec.alpha_survey[sv] = {"alpha": 0.0, "n": spec.alpha_survey[sv]["n"]}
                for ck in list(spec.alpha_cells):
                    if ck.startswith(f"{sv}|"):
                        spec.alpha_cells[ck]["alpha"] = 0.0
                        spec.alpha_cells[ck]["collapsed"] = "anchor"
                verdict = "collapsed_anchor"
        verify[sv] = {
            "loss_blend": ll_blend, "loss_anchor": ll_anchor,
            "auc_blend": auc_blend, "auc_anchor": auc_anchor,
            "verdict": verdict,
        }

    spec.g3 = {"per_cell": g3_cells, "per_survey_verify": verify}
    if exclude_qualities or weight_col is not None or orig_sv:
        spec.g3["fit_frame"] = {
            "exclude_qualities": list(exclude_qualities),
            "weight_col": weight_col,
            "n_rows_after_filters": int(len(df)),
            "n_usable": int(usable.sum()),
        }
        if orig_sv:
            spec.g3["fit_frame"]["original_rows_surveys"] = orig_sv
            spec.g3["fit_frame"]["n_copies_dropped"] = n_copies_dropped
    if out_dir is not None:
        spec.save(out_dir)
    return spec


def _neg_logloss(proba: np.ndarray, y: np.ndarray, w: np.ndarray | None = None,
                 objective: str = "multiclass") -> float:
    nll = _sample_losses(proba, proba, 1.0, y, objective)
    if w is None:
        return float(nll.mean())
    w = np.asarray(w, dtype=float)
    return float((w * nll).sum() / w.sum())


def _verify_auc(y: np.ndarray, proba: np.ndarray, w: np.ndarray | None,
                objective: str) -> float | None:
    if objective != "sn_binary":
        return _macro_ovr_auc(y, proba, w)
    from sklearn.metrics import roc_auc_score
    y_sn = np.isin(y, _SN_INDEX).astype(int)
    if y_sn.min() == y_sn.max():
        return None
    kw = {} if w is None else {"sample_weight": np.asarray(w, dtype=float)}
    return float(roc_auc_score(y_sn, proba[:, list(_SN_INDEX)].sum(axis=1), **kw))


# ── apply ─────────────────────────────────────────────────────────────────

def apply(
    df: pd.DataFrame,
    spec: BlendSpec,
    *,
    model_cols: tuple[str, str, str] = ("p_snia", "p_nonia", "p_other"),
    out_cols: tuple[str, str, str] = ("p_snia", "p_nonia", "p_other"),
) -> pd.DataFrame:
    """Blend model + anchor per the fitted α table.

    Adds ``p_{snia,nonia,other}_model`` (copies of the input model probs),
    ``alpha``, ``alpha_fallback_level`` and overwrites ``out_cols`` (the deployed
    probabilities) with ``α·model + (1−α)·anchor``.  Rows without an anchor
    (n_experts_fired==0 or NaN anchor) are pinned to the model (α=1).
    """
    df = df.copy()
    if (not all(c in df.columns for c in _ANCHOR_COLS) or spec.drop_experts
            or spec.anchor_call_experts):
        # v13: a spec with dropped experts always recomputes the anchor so a
        # caller's anchor (scorer: compute_anchor without the spec) cannot
        # let a dropped expert fire; v13d: likewise for call-trust weights.
        df = compute_anchor(df, **spec.anchor_kwargs())

    model = np.column_stack([_num(df, c) for c in model_cols])
    anchor = df[list(_ANCHOR_COLS)].to_numpy(dtype=float)
    has_anchor = np.isfinite(anchor).all(axis=1)

    n_exp = df.get("n_experts_fired", pd.Series(np.zeros(len(df)))).to_numpy()
    n_exp_b = _n_experts_bucket(np.asarray(n_exp))
    cov_b = _lc_cov_bucket(_lc_cov(df))
    surveys = (
        df["survey"].astype(str).str.lower().to_numpy()
        if "survey" in df.columns else np.full(len(df), "unknown", dtype=object)
    )

    alphas = np.ones(len(df))
    levels = np.empty(len(df), dtype=object)
    for i in range(len(df)):
        a, lvl = spec.lookup_alpha(surveys[i], n_exp_b[i], cov_b[i], bool(has_anchor[i]))
        alphas[i] = a
        levels[i] = lvl

    a = alphas[:, None]
    blend = a * model + (1.0 - a) * np.where(has_anchor[:, None], anchor, model)

    model_out = ("p_snia_model", "p_nonia_model", "p_other_model")
    for i, mc in enumerate(model_out):
        df[mc] = model[:, i]
    for i, oc in enumerate(out_cols):
        df[oc] = blend[:, i]
    df["alpha"] = alphas
    df["alpha_fallback_level"] = levels
    return df
