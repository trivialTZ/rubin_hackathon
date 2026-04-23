"""Local ORACLE (dev-ved30/Oracle) projector — hierarchical 19-class → ternary.

Shares class mapping with `antares/oracle` (see projectors/antares.py); the
local runner emits a full 19-class probability vector that we aggregate.
"""
from __future__ import annotations

import json
from typing import Any

from .base import summarize_ternary
from .antares import _ORACLE_IA, _ORACLE_NONIA, _ORACLE_OTHER_TRANSIENT, _ORACLE_AGN, _ORACLE_PERIODIC


def project_events(expert_key: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    if expert_key != "oracle_lsst":
        return {"prediction_type": "unknown", "reason": f"wrong key {expert_key}"}
    valid = [e for e in events if e.get("raw_probs_json")]
    if not valid:
        return {"prediction_type": "class_correctness", "reason": "no oracle outputs"}
    latest = valid[-1]
    try:
        probs = json.loads(latest["raw_probs_json"])
    except (json.JSONDecodeError, TypeError):
        return {"prediction_type": "class_correctness", "reason": "invalid probs"}
    if not probs:
        return {"prediction_type": "class_correctness", "reason": "empty probs"}

    p_snia = p_nonia = p_other = 0.0
    for key, val in probs.items():
        lo = str(key).strip().lower().replace(" ", "")
        try:
            w = float(val)
        except (TypeError, ValueError):
            continue
        if lo in _ORACLE_IA:
            p_snia += w
        elif lo in _ORACLE_NONIA:
            p_nonia += w
        elif lo in (_ORACLE_OTHER_TRANSIENT | _ORACLE_AGN | _ORACLE_PERIODIC):
            p_other += w
        else:
            p_other += w  # unknown → route to other

    total = p_snia + p_nonia + p_other
    if total <= 0:
        return {"prediction_type": "class_correctness", "reason": "zero mass"}
    result = summarize_ternary(p_snia / total, p_nonia / total, p_other / total)
    result["raw_oracle_top_class"] = max(probs, key=lambda k: float(probs[k]))
    return result
