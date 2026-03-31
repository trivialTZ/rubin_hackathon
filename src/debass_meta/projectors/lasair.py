"""Lasair Sherlock projector logic."""
from __future__ import annotations

from typing import Any

_KNOWN_CONTEXT_TAGS = ["SN", "NT", "AGN", "CV", "BS", "ORPHAN", "VS", "UNCLEAR"]


def project_events(expert_key: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        return {"prediction_type": "context_only", "reason": "missing sherlock context"}
    tag = (
        events[0].get("class_name")
        or events[0].get("raw_label_or_score")
        or "UNKNOWN"
    )
    out: dict[str, Any] = {
        "prediction_type": "context_only",
        "context_tag": str(tag),
    }
    for known in _KNOWN_CONTEXT_TAGS:
        out[f"context_{known}"] = 1.0 if str(tag) == known else 0.0
    return out
