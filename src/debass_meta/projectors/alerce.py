"""ALeRCE projector logic."""
from __future__ import annotations

from typing import Any

from .base import summarize_ternary

_LC_NONIA = {"SNIbc", "SNII", "SLSN"}
_LC_OTHER = {"AGN", "VS", "asteroid", "bogus"}


def project_events(expert_key: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    if expert_key == "alerce/lc_classifier_transient":
        p_snia = 0.0
        p_nonia = 0.0
        p_other = 0.0
        found = False
        for row in events:
            cls = row.get("class_name")
            value = row.get("canonical_projection")
            if value is None:
                continue
            found = True
            value = float(value)
            if cls == "SNIa":
                p_snia += value
            elif cls in _LC_NONIA:
                p_nonia += value
            elif cls in _LC_OTHER:
                p_other += value
        if not found:
            return {"prediction_type": "class_correctness", "reason": "missing classifier probabilities"}
        return summarize_ternary(p_snia, p_nonia, p_other)

    if expert_key == "alerce/stamp_classifier":
        p_nonia = 0.0
        p_other = 0.0
        found = False
        for row in events:
            cls = row.get("class_name")
            value = row.get("canonical_projection")
            if value is None:
                continue
            found = True
            value = float(value)
            if cls == "SN":
                p_nonia += value
            else:
                p_other += value
        if not found:
            return {"prediction_type": "class_correctness", "reason": "missing stamp probabilities"}
        return summarize_ternary(0.0, p_nonia, p_other)

    return {"prediction_type": "unknown", "reason": f"unsupported alerce expert {expert_key}"}
