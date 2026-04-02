"""Shared projector helpers."""
from __future__ import annotations

from math import log
from typing import Any

PHASE1_EXPERT_KEYS = [
    "fink/snn",
    "fink/rf_ia",
    "parsnip",
    "supernnova",
    "alerce_lc",
    "alerce/lc_classifier_transient",
    "alerce/stamp_classifier",
    "lasair/sherlock",
]


def sanitize_expert_key(expert_key: str) -> str:
    return expert_key.replace("/", "__")


def _entropy(probabilities: list[float]) -> float | None:
    probs = [float(p) for p in probabilities if p is not None and p > 0]
    if not probs:
        return None
    return float(-sum(p * log(p) for p in probs))


def summarize_ternary(
    p_snia: float | None,
    p_nonia_snlike: float | None,
    p_other: float | None,
) -> dict[str, Any]:
    probs = {
        "snia": p_snia,
        "nonIa_snlike": p_nonia_snlike,
        "other": p_other,
    }
    clean = {k: float(v) for k, v in probs.items() if v is not None}
    total = sum(clean.values())
    if clean and total > 0:
        clean = {k: v / total for k, v in clean.items()}
    p_snia = clean.get("snia")
    p_nonia_snlike = clean.get("nonIa_snlike")
    p_other = clean.get("other")
    ranked = sorted(clean.items(), key=lambda item: item[1], reverse=True)
    mapped_pred_class = ranked[0][0] if ranked else None
    top1_prob = ranked[0][1] if ranked else None
    margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else top1_prob
    return {
        "prediction_type": "class_correctness",
        "mapped_pred_class": mapped_pred_class,
        "p_snia": p_snia,
        "p_nonIa_snlike": p_nonia_snlike,
        "p_other": p_other,
        "top1_prob": top1_prob,
        "margin": margin,
        "entropy": _entropy(list(clean.values())),
    }


def collect_projected_columns(expert_key: str, projected: dict[str, Any]) -> dict[str, Any]:
    prefix = f"proj__{sanitize_expert_key(expert_key)}__"
    out: dict[str, Any] = {}
    for key, value in projected.items():
        if key in {"prediction_type", "mapped_pred_class", "context_tag", "reason"}:
            out[key] = value
        else:
            out[f"{prefix}{key}"] = value
    return out


def project_expert_events(expert_key: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    if expert_key == "fink/snn" or expert_key == "fink/rf_ia":
        from .fink import project_events

        return project_events(expert_key, events)
    if expert_key.startswith("alerce/"):
        from .alerce import project_events

        return project_events(expert_key, events)
    if expert_key == "lasair/sherlock":
        from .lasair import project_events

        return project_events(expert_key, events)
    if expert_key in {"parsnip", "supernnova", "alerce_lc"}:
        from .local import project_events

        return project_events(expert_key, events)
    return {"prediction_type": "unknown", "reason": f"no projector for {expert_key}"}
