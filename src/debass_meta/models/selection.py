"""Science-goal-conditioned ranking + budget-aware selection for fusion_v8.

Priority score ``s = u · p̂`` with calibrated probabilities is the
Bayes-optimal expected-yield policy for any utility vector ``u`` over
``(snia, nonIa_snlike, other)`` — one model serves all science goals.

Selection policy (design §3 + spec correction #4d):
  1. eligibility filter — objects whose conformal set excludes ALL classes
     with ``u_c > 0`` are ineligible (calibrated abstention from wasting
     scope time);
  2. abstention DEMOTION — objects whose conformal set is all 3 classes
     ("I don't know") are ranked BELOW decided objects, not dropped;
  3. rank by ``s`` (tie-break: ``traj_x__mean_slope`` descending — rising
     broker scores first, favouring pre-peak targets);
  4. select top ``budget``; report ``expected_yield = Σ_selected s`` and
     ``expected_purity = expected_yield / budget`` in ``DataFrame.attrs``.

Risk-controlled variant: ``fdr_controlled_threshold`` picks the score
threshold τ on the CAL split such that the empirical FDR of
``{i : s_i >= τ}`` is <= γ (fixed-sequence step-up on the sorted cal
scores); pass it as ``score_threshold`` to ``rank_and_select`` to guarantee
purity at the cost of completeness (the spectroscopic-TAC use-case).

No model fitting happens here; everything is a pure function of the scored
prediction frame, so there is no leakage surface beyond the calibrated
inputs themselves.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

# Utility presets over (snia, nonIa_snlike, other) — pinned interface.
UTILITY_PRESETS = {
    "ia": (1, 0, 0),
    "nonia": (0, 1, 0),
    "other": (0, 0, 1),
}

# Probability column aliases, in CLASSES order (snia, nonIa_snlike, other).
_PROB_COL_ALIASES: tuple[tuple[str, ...], ...] = (
    ("p_snia",),
    ("p_nonia", "p_nonIa", "p_nonia_snlike", "p_nonIa_snlike"),
    ("p_other",),
)


def _resolve_utility(utility) -> np.ndarray:
    if isinstance(utility, str):
        key = utility.lower()
        if key not in UTILITY_PRESETS:
            raise KeyError(
                f"unknown utility preset {utility!r}; "
                f"have {sorted(UTILITY_PRESETS)}"
            )
        utility = UTILITY_PRESETS[key]
    u = np.asarray(utility, dtype=float)
    if u.shape != (3,):
        raise ValueError(f"utility must be a 3-vector, got shape {u.shape}")
    if not (u > 0).any():
        raise ValueError("utility vector must have at least one positive entry")
    return u


def _resolve_prob_matrix(pred: pd.DataFrame) -> np.ndarray:
    cols: list[str] = []
    for aliases in _PROB_COL_ALIASES:
        found = next((c for c in aliases if c in pred.columns), None)
        if found is None:
            raise KeyError(
                f"prediction frame missing probability column "
                f"(any of {aliases}); have {list(pred.columns)[:20]}..."
            )
        cols.append(found)
    return pred[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)


def fdr_controlled_threshold(
    scores_cal: np.ndarray,
    is_target_cal: np.ndarray,
    gamma: float = 0.2,
) -> float:
    """Score threshold τ with empirical cal FDR(score >= τ) <= γ.

    Fixed-sequence step-up on the cal scores sorted descending: find the
    largest prefix k whose false-discovery fraction is <= γ; τ = the k-th
    score.  Returns ``inf`` when no prefix satisfies the constraint
    (i.e. select nothing).
    """
    scores = np.asarray(scores_cal, dtype=float)
    y = np.asarray(is_target_cal, dtype=float)
    if len(scores) == 0:
        return float("inf")
    order = np.argsort(-scores, kind="stable")
    false = 1.0 - y[order]
    k = np.arange(1, len(scores) + 1)
    fdr = np.cumsum(false) / k
    passing = np.flatnonzero(fdr <= float(gamma))
    if len(passing) == 0:
        return float("inf")
    return float(scores[order][passing.max()])


def rank_and_select(
    pred: pd.DataFrame,
    utility,
    budget: int,
    *,
    set_cols: Sequence[str] = ("set_snia", "set_nonia", "set_other"),
    tiebreak_col: str = "traj_x__mean_slope",
    score_threshold: float | None = None,
) -> pd.DataFrame:
    """Rank candidates by expected utility and select the top ``budget``.

    Parameters
    ----------
    pred : frame with ``p_snia / p_nonia / p_other`` + the conformal set
        membership columns ``set_cols`` (bool; NaN treated as member —
        unknown conformal status can never make an object ineligible).
    utility : preset name in ``UTILITY_PRESETS`` or a 3-vector over
        (snia, nonIa_snlike, other).
    budget : number of selections.
    score_threshold : optional risk-control τ (see
        ``fdr_controlled_threshold``); rows with ``s < τ`` become ineligible.

    Returns a copy of ``pred`` (original row order) with added columns
    ``priority_score, set_size, eligible, demoted, rank, selected`` and
    ``attrs``: ``expected_yield``, ``expected_purity``, ``n_selected``,
    ``budget``, ``utility``.
    """
    u = _resolve_utility(utility)
    out = pred.copy()
    n = len(out)
    P = _resolve_prob_matrix(out)
    s = P @ u

    missing_sets = [c for c in set_cols if c not in out.columns]
    if missing_sets:
        raise KeyError(
            f"prediction frame missing conformal set columns: {missing_sets}"
        )
    membership = (
        out[list(set_cols)]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(1.0)  # unknown conformal status cannot exclude a class
        .to_numpy(float)
        > 0.5
    )
    set_size = membership.sum(axis=1).astype(int)

    # 1. Eligibility: the conformal set must retain at least one class the
    #    user cares about ("certainly not your target" -> drop from ranking).
    eligible = membership[:, u > 0].any(axis=1)
    if score_threshold is not None:
        eligible = eligible & (s >= float(score_threshold))

    # 2. Abstention demotion (spec correction #4d): full-set objects are
    #    uncertain, not high-priority — ranked below every decided object.
    demoted = set_size >= len(set_cols)

    if tiebreak_col in out.columns:
        tiebreak = (
            pd.to_numeric(out[tiebreak_col], errors="coerce")
            .to_numpy(float)
        )
        tiebreak = np.where(np.isfinite(tiebreak), tiebreak, -np.inf)
    else:
        tiebreak = np.zeros(n, dtype=float)

    rank = np.full(n, np.nan)
    eligible_idx = np.flatnonzero(eligible)
    if len(eligible_idx) > 0:
        # lexsort: last key is primary -> demoted asc, then s desc, tb desc.
        order = np.lexsort((
            -tiebreak[eligible_idx],
            -s[eligible_idx],
            demoted[eligible_idx].astype(int),
        ))
        rank[eligible_idx[order]] = np.arange(1, len(eligible_idx) + 1)

    with np.errstate(invalid="ignore"):
        selected = eligible & (rank <= int(budget))

    out["priority_score"] = s
    out["set_size"] = set_size
    out["eligible"] = eligible
    out["demoted"] = demoted
    out["rank"] = rank
    out["selected"] = selected

    expected_yield = float(s[selected].sum())
    out.attrs["expected_yield"] = expected_yield
    out.attrs["expected_purity"] = (
        expected_yield / float(budget) if budget > 0 else float("nan")
    )
    out.attrs["n_selected"] = int(selected.sum())
    out.attrs["budget"] = int(budget)
    out.attrs["utility"] = [float(v) for v in u]
    return out
