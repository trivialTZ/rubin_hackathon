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
    return {"prediction_type": "unknown",
            "reason": f"unsupported pittgoogle expert: {expert_key}"}


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
        # SuperNNova is binary (Ia vs non-Ia), so p_other is just numeric
        # residual from normalisation noise.  Assign a small floor so the
        # ternary normalisation doesn't collapse "other" to exactly 0.
        p_nonia_sn = float(p_nonia_raw)
        residual = max(0.0, 1.0 - p_ia - p_nonia_sn)
        p_other = max(residual, 0.01)
        p_nonia_snlike = p_nonia_sn
    else:
        # Only P(Ia) available — split complement.
        complement = 1.0 - p_ia
        p_nonia_snlike = complement * 0.7
        p_other = complement * 0.3

    result = summarize_ternary(p_ia, p_nonia_snlike, p_other)
    result["raw_supernnova_p_ia"] = p_ia
    result["n_score_events"] = len(score_events)
    return result
