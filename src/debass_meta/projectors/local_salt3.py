"""SALT3 local projector — fit-based SN Ia consistency via sncosmo.

Events are emitted one per class by ``_local_record_to_events`` in
``ingest/gold.py``:
  • ``class_name``  = "Ia" or "II"
  • ``canonical_projection`` = probability for that class

The Salt3Chi2Expert writes ``class_probabilities = {"Ia": p, "II": 1-p}``
where p = sigmoid(Δχ²/2). We map:
  • Ia → p_snia
  • II → p_nonIa_snlike (Nugent-II-P is non-Ia SN-like)
  • everything else → p_other (0 by default)
"""
from __future__ import annotations

from typing import Any

from .base import summarize_ternary


def project_events(expert_key: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    if expert_key != "salt3_chi2":
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
        if name in ("ia", "snia"):
            p_snia += float(value)
        elif name in ("ii", "nonia_snlike", "non-ia", "nonia", "non_ia"):
            p_nonia += float(value)
        else:
            p_other += float(value)
    if not found:
        return {
            "prediction_type": "class_correctness",
            "reason": "no salt3 fits",
        }
    return summarize_ternary(p_snia, p_nonia, p_other)
