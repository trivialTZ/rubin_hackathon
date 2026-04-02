"""Fink LSST projector logic.

Fink LSST uses different column names than ZTF Fink:
  - ``f:clf_snnSnVsOthers_score`` — SN vs Others (binary, like ZTF's snn_sn_vs_all)
  - ``f:clf_cats_class`` — CATS integer class label (11=SN-like, 21=long, etc.)
  - ``f:clf_cats_score`` — CATS confidence score
  - ``f:clf_earlySNIa_score`` — Early SN Ia score (probability when triggered)

CATS class mapping (from Moller et al. 2024):
  11 = SN-like (supernova candidate)
  21 = Long (long-duration transient)
  31 = Fast (fast transient)
  41 = Periodic
  51 = Non-Periodic (stochastic)
"""
from __future__ import annotations

from typing import Any

from .base import summarize_ternary

# CATS class → ternary mapping
_CATS_SNIA_LIKE = {11}    # SN-like → could be Ia or non-Ia
_CATS_NONIA = {21, 31}    # Long/Fast transients → non-Ia SN-like
_CATS_OTHER = {41, 51}    # Periodic/Non-periodic → other


def project_events(expert_key: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    if expert_key == "fink_lsst/snn":
        return _project_snn_lsst(events)
    if expert_key == "fink_lsst/cats":
        return _project_cats(events)
    if expert_key == "fink_lsst/early_snia":
        return _project_early_snia(events)
    return {"prediction_type": "unknown", "reason": f"unsupported fink_lsst expert {expert_key}"}


def _project_snn_lsst(events: list[dict[str, Any]]) -> dict[str, Any]:
    """SNN SN vs Others — binary score.

    Maps: high score → SN-like (split between snia and nonIa_snlike),
    low score → other.
    """
    scores = [float(e["canonical_projection"]) for e in events
              if e.get("canonical_projection") is not None]
    if not scores:
        return {"prediction_type": "class_correctness", "reason": "no snn scores"}

    # Take the latest score (most informed)
    sn_prob = scores[-1]
    # SNN gives P(SN) — we can't distinguish Ia from non-Ia with this alone
    # Split SN probability evenly between snia and nonIa_snlike as a prior
    # (the trust model will learn the real calibration from training data)
    p_snia = sn_prob * 0.5
    p_nonia = sn_prob * 0.5
    p_other = 1.0 - sn_prob
    result = summarize_ternary(p_snia, p_nonia, p_other)
    result["raw_snn_sn_vs_others"] = sn_prob
    return result


def _project_cats(events: list[dict[str, Any]]) -> dict[str, Any]:
    """CATS — integer class label + confidence score.

    CATS gives a class (11=SN-like) with a confidence score.
    """
    # Find events with class_name (integer label)
    class_events = [e for e in events if e.get("class_name") is not None]
    if not class_events:
        return {"prediction_type": "class_correctness", "reason": "no cats events"}

    latest = class_events[-1]
    cats_class = int(latest.get("class_name", 0))
    cats_score = float(latest.get("canonical_projection", 0) or 0)

    if cats_class in _CATS_SNIA_LIKE:
        # SN-like: assign score to SN, rest to other
        p_snia = cats_score * 0.5  # can't distinguish Ia from non-Ia
        p_nonia = cats_score * 0.5
        p_other = 1.0 - cats_score
    elif cats_class in _CATS_NONIA:
        p_snia = 0.0
        p_nonia = cats_score
        p_other = 1.0 - cats_score
    else:
        p_snia = 0.0
        p_nonia = 0.0
        p_other = 1.0

    result = summarize_ternary(p_snia, p_nonia, p_other)
    result["raw_cats_class"] = cats_class
    result["raw_cats_score"] = cats_score
    return result


def _project_early_snia(events: list[dict[str, Any]]) -> dict[str, Any]:
    """EarlySNIa — probability of being SN Ia.

    Only triggered when SNN + RF both indicate SN.
    Score of -1 means not triggered (not enough evidence).
    """
    scores = [float(e["canonical_projection"]) for e in events
              if e.get("canonical_projection") is not None and float(e["canonical_projection"]) >= 0]
    if not scores:
        return {"prediction_type": "class_correctness", "reason": "early_snia not triggered"}

    p_snia = scores[-1]
    p_nonia = (1.0 - p_snia) * 0.8  # most non-Ia are still SN-like when EarlySNIa triggers
    p_other = (1.0 - p_snia) * 0.2
    result = summarize_ternary(p_snia, p_nonia, p_other)
    result["raw_early_snia_score"] = p_snia
    return result
