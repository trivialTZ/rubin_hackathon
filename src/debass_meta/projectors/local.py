"""Local expert projector logic."""
from __future__ import annotations

from typing import Any

from .base import summarize_ternary


def _map_class(class_name: str | None) -> str:
    if class_name is None:
        return "other"
    name = str(class_name).lower()
    if "ia" in name:
        return "snia"
    if name.startswith("sn") or "slsn" in name or "ibc" in name or "iin" in name or "iib" in name:
        return "nonIa_snlike"
    return "other"


def project_events(expert_key: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    p_snia = 0.0
    p_nonia = 0.0
    p_other = 0.0
    found = False
    for row in events:
        value = row.get("canonical_projection")
        if value is None:
            continue
        found = True
        mapped = _map_class(row.get("class_name"))
        if mapped == "snia":
            p_snia += float(value)
        elif mapped == "nonIa_snlike":
            p_nonia += float(value)
        else:
            p_other += float(value)
    if not found:
        return {"prediction_type": "class_correctness", "reason": f"missing probabilities for {expert_key}"}
    return summarize_ternary(p_snia, p_nonia, p_other)
