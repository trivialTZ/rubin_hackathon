"""Local Bazin/Villar light-curve feature expert projector.

Local experts are emitted one event per class by
``_local_record_to_events`` in ``ingest/gold.py``, with:
  • ``class_name``  = "snia" / "nonIa_snlike" / "other"
  • ``canonical_projection`` = probability for that class

The LightGBM head in ``LcFeaturesExpert`` emits a ternary softmax already,
so we just sum canonical_projection values by mapped class.
"""
from __future__ import annotations

from typing import Any

from .base import summarize_ternary


def project_events(expert_key: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    if expert_key != "lc_features_bv":
        return {"prediction_type": "unknown", "reason": f"wrong key {expert_key}"}

    p_snia = 0.0
    p_nonia = 0.0
    p_other = 0.0
    found = False
    for row in events:
        value = row.get("canonical_projection")
        if value is None:
            continue
        found = True
        name = str(row.get("class_name") or "").strip().lower()
        if name == "snia":
            p_snia += float(value)
        elif name in ("nonia_snlike", "non-ia", "nonia", "non_ia"):
            p_nonia += float(value)
        else:
            p_other += float(value)
    if not found:
        return {
            "prediction_type": "class_correctness",
            "reason": "no lc feature events",
        }
    return summarize_ternary(p_snia, p_nonia, p_other)
