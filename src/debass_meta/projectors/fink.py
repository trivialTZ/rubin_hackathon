"""Fink projector logic."""
from __future__ import annotations

from typing import Any

from .base import summarize_ternary


def _row_value(events: list[dict[str, Any]], field: str) -> float | None:
    for row in events:
        if row.get("field") == field and row.get("canonical_projection") is not None:
            return float(row["canonical_projection"])
    return None


def project_events(expert_key: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    if expert_key == "fink/snn":
        p_snia_vs_nonia = _row_value(events, "snn_snia_vs_nonia")
        p_sn_vs_all = _row_value(events, "snn_sn_vs_all")
        if p_snia_vs_nonia is None or p_sn_vs_all is None:
            return {"prediction_type": "class_correctness", "reason": "missing snn fields"}
        p_snia = p_snia_vs_nonia * p_sn_vs_all
        p_nonia = (1.0 - p_snia_vs_nonia) * p_sn_vs_all
        p_other = 1.0 - p_sn_vs_all
        out = summarize_ternary(p_snia, p_nonia, p_other)
        out["raw_snn_snia_vs_nonia"] = p_snia_vs_nonia
        out["raw_snn_sn_vs_all"] = p_sn_vs_all
        return out

    if expert_key == "fink/rf_ia":
        p_snia = _row_value(events, "rf_snia_vs_nonia")
        if p_snia is None:
            return {"prediction_type": "class_correctness", "reason": "missing rf_snia_vs_nonia"}
        mapped_pred = "snia" if p_snia >= 0.5 else "nonIa_snlike"
        top1_prob = max(p_snia, 1.0 - p_snia)
        return {
            "prediction_type": "class_correctness",
            "mapped_pred_class": mapped_pred,
            "p_snia_scalar": p_snia,
            "p_non_snia_scalar": 1.0 - p_snia,
            "top1_prob": top1_prob,
            "margin": abs(p_snia - 0.5) * 2.0,
            "entropy": None,
        }

    return {"prediction_type": "unknown", "reason": f"unsupported fink expert {expert_key}"}
