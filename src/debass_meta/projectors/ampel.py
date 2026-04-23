"""AMPEL projector — SNGuess binary + ParSNIP FollowMe 14-class → DEBASS ternary.

Two experts live here:
  • `ampel/snguess`         — T2MultiXGBClassifier binary "young extragalactic
                              transient" probability
  • `ampel/parsnip_followme` — T2RunParsnip 14-class probability vector
                              (SNIa, SNII, SNIbc, SNIax, 91bg, SLSN-I,
                               TDE, KN, AGN, μLens, CART, ILOT, PISN, non-SN)
"""
from __future__ import annotations

import json
from typing import Any

from .base import summarize_ternary

_PARSNIP_IA = {"snia", "sn_ia", "ia", "sniacore", "sniamain"}
_PARSNIP_NONIA = {
    "snii",
    "sniip",
    "sniil",
    "sn_ii",
    "ii",
    "snib",
    "snic",
    "snibc",
    "snibc_mag",
    "ib/c",
    "ib",
    "ic",
    "sniib",
    "sniin",
    "iin",
    "iib",
    "sniax",
    "iax",
    "sn91bg",
    "91bg",
    "slsn-i",
    "slsn",
    "slsn-ii",
}
_PARSNIP_OTHER_TRANSIENT = {
    "tde",
    "kn",
    "kilonova",
    "agn",
    "mulens",
    "cart",
    "ilot",
    "pisn",
    "non_sn",
    "nonsn",
    "other",
}


def project_events(expert_key: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    if expert_key == "ampel/snguess":
        return _project_snguess(events)
    if expert_key == "ampel/parsnip_followme":
        return _project_parsnip_followme(events)
    return {"prediction_type": "unknown", "reason": f"unsupported ampel expert {expert_key}"}


def _project_snguess(events: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [
        float(e["canonical_projection"])
        for e in events
        if e.get("canonical_projection") is not None
    ]
    if not scores:
        return {"prediction_type": "class_correctness", "reason": "no snguess scores"}
    prob = scores[-1]
    prob = max(0.0, min(1.0, prob))
    # SNGuess is a SN-like selector; split SN mass 50/50 between Ia and non-Ia
    # for absence of subtype discrimination, 1-prob → p_other.
    p_snia = prob * 0.5
    p_nonia = prob * 0.5
    p_other = 1.0 - prob
    result = summarize_ternary(p_snia, p_nonia, p_other)
    result["raw_snguess_prob"] = prob
    return result


def _project_parsnip_followme(events: list[dict[str, Any]]) -> dict[str, Any]:
    valid_events = [e for e in events if e.get("raw_probs_json")]
    if not valid_events:
        return {"prediction_type": "class_correctness", "reason": "no parsnip probs"}
    latest = valid_events[-1]
    try:
        probs = json.loads(latest["raw_probs_json"])
    except (json.JSONDecodeError, TypeError):
        return {"prediction_type": "class_correctness", "reason": "invalid probs"}
    if not probs:
        return {"prediction_type": "class_correctness", "reason": "empty probs"}

    p_snia = 0.0
    p_nonia = 0.0
    p_other = 0.0
    for key, val in probs.items():
        lower = str(key).strip().lower().replace(" ", "")
        try:
            w = float(val)
        except (TypeError, ValueError):
            continue
        if lower in _PARSNIP_IA:
            p_snia += w
        elif lower in _PARSNIP_NONIA:
            p_nonia += w
        elif lower in _PARSNIP_OTHER_TRANSIENT:
            p_other += w
        else:
            # Unknown label → route to p_other (can't vouch for SN class)
            p_other += w

    total = p_snia + p_nonia + p_other
    if total <= 0:
        return {"prediction_type": "class_correctness", "reason": "zero total mass"}
    p_snia /= total
    p_nonia /= total
    p_other /= total
    result = summarize_ternary(p_snia, p_nonia, p_other)
    result["raw_parsnip_followme_top_class"] = latest.get("class_name")
    result["raw_parsnip_followme_top_prob"] = latest.get("canonical_projection")
    return result
