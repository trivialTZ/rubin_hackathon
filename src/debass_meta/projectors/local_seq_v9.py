"""Projector for the local seq_v9 sequence-classifier expert.

The expert emits the canonical ternary class probabilities
(snia / nonIa_snlike / other, folded from the v10 4-way head) plus, for 4-way
artifacts, ``p4_<class>`` extras persisted in the silver payload.  This
projector keeps the pipeline contract: it emits EXACTLY the ternary via
``summarize_ternary`` (renormalization + top1/margin/entropy).

Folding rules:
  * canonical ternary event names present → faithful passthrough (extras
    like ``p4_*`` are only used as a fallback, never double-counted);
  * otherwise 4-way names (bare ``snia/snii/other_sn/non_sn`` or ``p4_``-
    prefixed) fold as (snii + other_sn) → nonIa_snlike, non_sn → other.
"""
from __future__ import annotations

from typing import Any

from .base import summarize_ternary

# Canonical ternary event names (assignment; "snia" is shared with the 4-way).
_TERNARY_BUCKET = {
    "snia": "snia",
    "nonia_snlike": "nonIa_snlike",
    "nonia": "nonIa_snlike",
    "other": "other",
}
# 4-way event names → ternary bucket (accumulated: snii + other_sn fold).
_FOURWAY_BUCKET = {
    "snia": "snia",
    "snii": "nonIa_snlike",
    "other_sn": "nonIa_snlike",
    "non_sn": "other",
    "p4_snia": "snia",
    "p4_snii": "nonIa_snlike",
    "p4_other_sn": "nonIa_snlike",
    "p4_non_sn": "other",
}
# Names that mark a payload as ternary / 4-way ("snia" alone is ambiguous
# but folds identically either way).
_TERNARY_MARKERS = {"nonia_snlike", "nonia", "other"}
_FOURWAY_MARKERS = {"snii", "other_sn", "non_sn", "p4_snia", "p4_snii", "p4_other_sn", "p4_non_sn"}


def project_events(expert_key: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    tern = {"snia": 0.0, "nonIa_snlike": 0.0, "other": 0.0}
    four = {"snia": 0.0, "nonIa_snlike": 0.0, "other": 0.0}
    seen_ternary = seen_fourway = False
    for event in events:
        name = str(event.get("class_name") or "").strip().lower().replace(" ", "_")
        value = event.get("canonical_projection")
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if name in _TERNARY_BUCKET:
            tern[_TERNARY_BUCKET[name]] = value
            if name in _TERNARY_MARKERS:
                seen_ternary = True
        if name in _FOURWAY_BUCKET:
            four[_FOURWAY_BUCKET[name]] += value
            if name in _FOURWAY_MARKERS:
                seen_fourway = True
        if name == "snia":  # shared name — folds identically either way
            seen_ternary = True
    if not (seen_ternary or seen_fourway):
        return {"prediction_type": "unknown", "reason": "no seq_v9 class probabilities"}
    ternary_specific = any(
        str(e.get("class_name") or "").strip().lower().replace(" ", "_") in _TERNARY_MARKERS
        and e.get("canonical_projection") is not None
        for e in events
    )
    p = tern if (ternary_specific or not seen_fourway) else four
    return summarize_ternary(p["snia"], p["nonIa_snlike"], p["other"])
