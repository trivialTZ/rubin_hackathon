"""ANTARES projector — ORACLE (19-class) and Superphot+ (5-class) → ternary.

Taxonomy maps:

ORACLE (Shah+ 2025) hierarchical leaves we care about:
  SN Ia          → p_snia
  SN Ia-91bg, Iax, II, Ib/c, SLSN-I/II, IIn, IIb → p_nonIa_snlike
  KN, TDE, ILOT, CART, PISN                      → p_other (rare but not Ia)
  Periodic (Cep, RR, d Scu, EB, LPV)             → p_other
  AGN, μLens, M-dwarf flare, Dwarf Nova          → p_other

Superphot+ (de Soto+ 2024) 5 classes:
  Ia           → p_snia
  II, Ib/c, IIn, SLSN-I → p_nonIa_snlike
  (no "other"; confidence = max class prob)
"""
from __future__ import annotations

from typing import Any

from .base import summarize_ternary

# Canonical ORACLE leaf labels → ternary bucket
_ORACLE_IA = {"snia", "sniamain", "sn_ia", "ia", "sn ia"}
_ORACLE_NONIA = {
    "sn91bg",
    "sniax",
    "iax",
    "snii",
    "sn_ii",
    "ii",
    "sniip",
    "sniil",
    "snib",
    "snic",
    "snibc",
    "ib/c",
    "ib",
    "ic",
    "sniib",
    "sniin",
    "iin",
    "iib",
    "slsn-i",
    "slsn-ii",
    "slsn",
}
_ORACLE_OTHER_TRANSIENT = {
    "kn",
    "kilonova",
    "tde",
    "ilot",
    "cart",
    "pisn",
    "mulens",
    "microlens",
    "mdwarf",
    "mdwarf_flare",
    "dwarfnova",
    "dwarf_nova",
    "nova",
}
_ORACLE_AGN = {"agn", "agn_variability", "qso"}
_ORACLE_PERIODIC = {
    "cep",
    "cepheid",
    "rr",
    "rrlyr",
    "rrlyrae",
    "dscu",
    "dscut",
    "dscuti",
    "eb",
    "eclipsing_binary",
    "lpv",
    "mira",
}

_SUPERPHOT_IA = {"snia", "ia", "sn_ia"}
_SUPERPHOT_NONIA = {"snii", "ii", "snib", "snic", "snibc", "ib/c", "iin", "sniin", "slsn-i", "slsn"}


def project_events(expert_key: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    if expert_key == "antares/oracle":
        return _project_oracle(events)
    if expert_key == "antares/superphot_plus":
        return _project_superphot_plus(events)
    return {"prediction_type": "unknown", "reason": f"unsupported antares expert {expert_key}"}


def _project_oracle(events: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [e for e in events if e.get("canonical_projection") is not None or e.get("class_name")]
    if not valid:
        return {"prediction_type": "class_correctness", "reason": "no oracle tags"}
    latest = valid[-1]
    class_label = str(latest.get("class_name") or "").strip().lower().replace(" ", "")
    score = float(latest.get("canonical_projection") or 0.0)
    score = max(0.0, min(1.0, score))

    if class_label in _ORACLE_IA:
        p_snia, p_nonia, p_other = score, (1 - score) * 0.6, (1 - score) * 0.4
    elif class_label in _ORACLE_NONIA:
        p_snia, p_nonia, p_other = (1 - score) * 0.3, score, (1 - score) * 0.7
    elif class_label in _ORACLE_OTHER_TRANSIENT | _ORACLE_AGN | _ORACLE_PERIODIC:
        # Not-SN predictions get most of the mass to p_other; small residual on SN classes.
        p_snia, p_nonia, p_other = (1 - score) * 0.2, (1 - score) * 0.2, score
    else:
        # Unknown class → maximum entropy
        p_snia = p_nonia = p_other = 1.0 / 3.0

    result = summarize_ternary(p_snia, p_nonia, p_other)
    result["raw_oracle_class"] = class_label
    result["raw_oracle_score"] = score
    return result


def _project_superphot_plus(events: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [e for e in events if e.get("canonical_projection") is not None or e.get("class_name")]
    if not valid:
        return {"prediction_type": "class_correctness", "reason": "no superphot_plus tags"}
    latest = valid[-1]
    class_label = str(latest.get("class_name") or "").strip().lower().replace(" ", "")
    score = float(latest.get("canonical_projection") or 0.0)
    score = max(0.0, min(1.0, score))

    if class_label in _SUPERPHOT_IA:
        # Superphot+ contains no "other" class — residual mass split 50/50
        p_snia, p_nonia, p_other = score, (1 - score) * 0.6, (1 - score) * 0.4
    elif class_label in _SUPERPHOT_NONIA:
        p_snia, p_nonia, p_other = (1 - score) * 0.4, score, (1 - score) * 0.6
    else:
        p_snia = p_nonia = p_other = 1.0 / 3.0

    result = summarize_ternary(p_snia, p_nonia, p_other)
    result["raw_superphot_class"] = class_label
    result["raw_superphot_score"] = score
    return result
