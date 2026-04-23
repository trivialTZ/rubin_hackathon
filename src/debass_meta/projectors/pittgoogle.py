"""Pitt-Google Broker projector.

Handles:
  ``pittgoogle/supernnova_lsst``  — LSST SuperNNova binary P(Ia)
  ``pittgoogle/supernnova_ztf``   — ZTF  SuperNNova binary P(Ia)

SuperNNova produces a binary classification: class 0 = SN Ia, class 1 = non-Ia.
The probability of class 0 (P_Ia) is the canonical score.

Projection to ternary:
  p_snia         = P_Ia
  p_nonIa_snlike = (1 - P_Ia) * 0.7   # most non-Ia are still SN-like
  p_other        = (1 - P_Ia) * 0.3

If both class 0 and class 1 probabilities are available, we normalise them
directly and assign p_other from the residual (P_Ia + P_nonIa should sum to ~1
but numeric noise may shift it slightly).
"""
from __future__ import annotations

from typing import Any

from .base import summarize_ternary


def project_events(expert_key: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    if expert_key in {"pittgoogle/supernnova_lsst", "pittgoogle/supernnova_ztf"}:
        return _project_supernnova(events, expert_key)
    if expert_key == "pittgoogle/upsilon_lsst":
        return _project_upsilon(events)
    return {"prediction_type": "unknown",
            "reason": f"unsupported pittgoogle expert: {expert_key}"}


# UPSILoN class → ternary mapping.  Every UPSILoN class is a periodic variable
# star and therefore NOT a supernova; all mass goes to p_other, modulated by
# the classifier's confidence.  The DEBASS trust head learns how reliable this
# signal is for excluding SN candidates.
_UPSILON_PERIODIC = {
    "delta_scuti",
    "deltascuti",
    "dsct",
    "rr_lyrae",
    "rrlyrae",
    "rrl",
    "rr_ab",
    "rr_c",
    "cepheid",
    "cep",
    "classical_cepheid",
    "type_ii_cepheid",
    "t2cep",
    "eclipsing_binary",
    "eb",
    "long_period_variable",
    "lpv",
    "mira",
}
_UPSILON_NON_VARIABLE = {"non_variable", "nonvar", "nonvariable", "constant"}


def _project_upsilon(events: list[dict[str, Any]]) -> dict[str, Any]:
    """UPSILoN → ternary: periodic variables map to p_other.

    If UPSILoN is confident the source is a known variable class, we put that
    confidence into p_other.  Residual mass is split between p_snia and
    p_nonia_snlike at 50/50 (non-informative) so the trust head can learn the
    true prior from the labelled training data.
    """
    valid = [e for e in events if e.get("canonical_projection") is not None or e.get("class_name")]
    if not valid:
        return {"prediction_type": "class_correctness", "reason": "no upsilon events"}
    latest = valid[-1]
    cls = str(latest.get("class_name") or "").strip().lower()
    score = latest.get("canonical_projection")
    try:
        score = float(score) if score is not None else 1.0
    except (TypeError, ValueError):
        score = 1.0
    score = max(0.0, min(1.0, score))

    if cls in _UPSILON_PERIODIC:
        p_other = score
        p_snia = (1.0 - score) * 0.5
        p_nonia = (1.0 - score) * 0.5
    elif cls in _UPSILON_NON_VARIABLE:
        # Non-variable in UPSILoN's view — mildly supports transient hypothesis;
        # let SN mass share most of the residual with balanced subtype split.
        p_other = 1.0 - score
        p_snia = score * 0.5
        p_nonia = score * 0.5
    else:
        # Unknown class — maximum entropy ternary
        p_snia = p_nonia = p_other = 1.0 / 3.0

    result = summarize_ternary(p_snia, p_nonia, p_other)
    result["raw_upsilon_class"] = cls
    result["raw_upsilon_score"] = score
    return result


def _project_supernnova(events: list[dict[str, Any]], expert_key: str) -> dict[str, Any]:
    """Project SuperNNova binary P(Ia) → ternary.

    Takes the latest score from the event list (most detections = most
    informed prediction).  If ``canonical_complement`` (P_nonIa) is
    available we use it directly; otherwise we split the complement.
    """
    score_events = [
        e for e in events
        if e.get("canonical_projection") is not None
        and e.get("field") == "supernnova_prob_class0"
    ]
    if not score_events:
        return {"prediction_type": "class_correctness",
                "reason": "no supernnova scores"}

    latest = score_events[-1]
    p_ia = float(latest["canonical_projection"])
    p_nonia_raw = latest.get("canonical_complement")

    if p_nonia_raw is not None:
        # Both class 0 and class 1 are available — use them directly.
        # SuperNNova is binary (Ia vs non-Ia), so p_other is the numeric
        # residual.  For a well-calibrated binary model this ≈ 0; apply
        # the same volumetric-rate split to the non-Ia portion instead.
        p_nonia_sn = float(p_nonia_raw)
        p_nonia_snlike = p_nonia_sn * 0.7
        p_other = p_nonia_sn * 0.3
    else:
        # Only P(Ia) available — split complement using ZTF BTS volumetric
        # rate prior (Perley+2020): ~70% of non-Ia transients are CC SNe.
        complement = 1.0 - p_ia
        p_nonia_snlike = complement * 0.7
        p_other = complement * 0.3

    result = summarize_ternary(p_ia, p_nonia_snlike, p_other)
    result["raw_supernnova_p_ia"] = p_ia
    result["n_score_events"] = len(score_events)
    return result
