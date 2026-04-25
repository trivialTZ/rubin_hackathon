"""Local ORACLE (dev-ved30/Oracle) projector — hierarchical 19-class → ternary.

Shares class-mapping with `antares/oracle` (see projectors/antares.py). Accepts
two event shapes:

  1. Per-class events (as emitted by ingest.rsp_dp1._local_outputs_to_events):
       [{"class_name": "SNIa", "probability": 0.15, ...}, ...]
  2. A single event with a serialized JSON dict in `raw_probs_json` (legacy
     ANTARES feed shape).

Either way we aggregate the 19-leaf vector into ternary {p_snia, p_nonIa_snlike,
p_other} via the canonical mapping in `antares.py`.
"""
from __future__ import annotations

import json
from typing import Any

from .base import summarize_ternary
from .antares import (
    _ORACLE_AGN,
    _ORACLE_IA,
    _ORACLE_NONIA,
    _ORACLE_OTHER_TRANSIENT,
    _ORACLE_PERIODIC,
)


def _route_class(class_name: str) -> str:
    lo = str(class_name).strip().lower().replace(" ", "")
    if lo in _ORACLE_IA:
        return "snia"
    if lo in _ORACLE_NONIA:
        return "nonia"
    if lo in (_ORACLE_OTHER_TRANSIENT | _ORACLE_AGN | _ORACLE_PERIODIC):
        return "other"
    return "other"


def _aggregate(probs: dict[str, float]) -> dict[str, float]:
    p_snia = p_nonia = p_other = 0.0
    for cls, val in probs.items():
        try:
            w = float(val)
        except (TypeError, ValueError):
            continue
        bucket = _route_class(cls)
        if bucket == "snia":
            p_snia += w
        elif bucket == "nonia":
            p_nonia += w
        else:
            p_other += w
    return {"p_snia": p_snia, "p_nonia": p_nonia, "p_other": p_other}


def project_events(expert_key: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    if expert_key != "oracle_lsst":
        return {"prediction_type": "unknown", "reason": f"wrong key {expert_key}"}
    if not events:
        return {"prediction_type": "class_correctness", "reason": "no oracle outputs"}

    # Format 1: legacy single-event with raw_probs_json
    json_events = [e for e in events if e.get("raw_probs_json")]
    if json_events:
        latest = json_events[-1]
        try:
            probs = json.loads(latest["raw_probs_json"])
        except (json.JSONDecodeError, TypeError):
            return {"prediction_type": "class_correctness", "reason": "invalid probs json"}
        if not probs:
            return {"prediction_type": "class_correctness", "reason": "empty probs"}
        aggr = _aggregate(probs)
        total = sum(aggr.values())
        if total <= 0:
            return {"prediction_type": "class_correctness", "reason": "zero mass"}
        result = summarize_ternary(
            aggr["p_snia"] / total, aggr["p_nonia"] / total, aggr["p_other"] / total
        )
        result["raw_oracle_top_class"] = max(probs, key=lambda k: float(probs[k]))
        return result

    # Format 2: per-class events from _local_outputs_to_events
    probs: dict[str, float] = {}
    for ev in events:
        cls = ev.get("class_name") or ev.get("classifier_class")
        prob = (
            ev.get("canonical_projection")
            or ev.get("raw_label_or_score")
            or ev.get("probability")
        )
        if cls is None or prob is None:
            continue
        try:
            probs[str(cls)] = float(prob)
        except (TypeError, ValueError):
            continue
    if not probs:
        return {"prediction_type": "class_correctness", "reason": "no class rows"}
    aggr = _aggregate(probs)
    total = sum(aggr.values())
    if total <= 0:
        return {"prediction_type": "class_correctness", "reason": "zero mass"}
    result = summarize_ternary(
        aggr["p_snia"] / total, aggr["p_nonia"] / total, aggr["p_other"] / total
    )
    result["raw_oracle_top_class"] = max(probs, key=lambda k: probs[k])
    return result
