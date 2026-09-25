"""Fink LSST projector logic.

Fink LSST uses different column names than ZTF Fink:
  - ``f:clf_snnSnVsOthers_score`` — SN vs Others (binary, like ZTF's snn_sn_vs_all)
  - ``f:clf_cats_class`` — CATS broad class (ELAsTiCC class-ID prefix, see below)
  - ``f:clf_cats_score`` — probability of that class (max of a 5-class softmax;
    the smallest value in silver is ~0.24, just above the 1/5 floor)
  - ``f:clf_earlySNIa_score`` — Early SN Ia score (probability when triggered)

CATS broad classes (Fraga et al. 2024, A&A 692, A208) use the ELAsTiCC
hierarchical class IDs: 1x = non-recurring, 2x = recurring.
  11 = SN-like       (111-115: SN Ia, Ib/c, II, Iax, 91bg)
  12 = Fast          (121-124: kilonova, M-dwarf flare, dwarf nova, microlensing)
  13 = Long          (131-135: SLSN, TDE, ILOT, CART, PISN)
  21 = Periodic      (211-215: Cepheid, RR Lyrae, delta Scuti, EB, LPV)
  22 = Non-periodic  (221: AGN)
Only these five codes occur in Fink LSST data. Before 2026-09-24 this module
assumed 11/21/31/41/51 (21 = Long) and so read Periodic as a non-Ia SN vote;
see docs/metadebass_cats_bug.md.
"""
from __future__ import annotations

from typing import Any

from .base import summarize_ternary

# CATS broad class → ternary mapping
CATS_SN_LIKE = 11
# Long (13) holds SLSNe as well as TDEs/ILOTs/CARTs; it is treated as not
# SN-like, following the CATS taxonomy. raw_cats_class stays a feature, so the
# trust and follow-up heads can still learn what Long means for our labels.
CATS_NON_SN = frozenset({12, 13, 21, 22})
CATS_N_CLASSES = 5


def cats_ternary(cats_class: int, cats_score: float) -> tuple[float, float, float] | None:
    """(p_snia, p_nonia, p_other) for one CATS output; None for an unknown code.

    SN-like: the score goes to SN, split 50/50 between Ia and non-Ia (CATS has
    no subtype), the rest to other. Any other class: the score goes to other,
    and the remainder is spread evenly over the four remaining classes, so the
    SN-like share is (1 - score) / 4 (maximum-entropy split of the softmax).
    """
    if cats_class == CATS_SN_LIKE:
        return 0.5 * cats_score, 0.5 * cats_score, 1.0 - cats_score
    if cats_class in CATS_NON_SN:
        p_sn = (1.0 - cats_score) / (CATS_N_CLASSES - 1)
        return 0.5 * p_sn, 0.5 * p_sn, 1.0 - p_sn
    return None


def project_events(expert_key: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    if expert_key == "fink_lsst/snn":
        return _project_snn_lsst(events)
    if expert_key == "fink_lsst/cats":
        return _project_cats(events)
    if expert_key == "fink_lsst/early_snia":
        return _project_early_snia(events)
    return {"prediction_type": "unknown", "reason": f"unsupported fink_lsst expert {expert_key}"}


def _project_snn_lsst(events: list[dict[str, Any]]) -> dict[str, Any]:
    """SNN SN vs Others — binary score.

    Maps: high score → SN-like (split between snia and nonIa_snlike),
    low score → other.
    """
    scores = [float(e["canonical_projection"]) for e in events
              if e.get("canonical_projection") is not None]
    if not scores:
        return {"prediction_type": "class_correctness", "reason": "no snn scores"}

    # Take the latest score (most informed)
    sn_prob = scores[-1]
    # SNN gives P(SN) — no Ia vs non-Ia subtype distinction available.
    # Uniform 50/50 split is the maximum-entropy (least-informative) prior
    # for binary subtype when the classifier provides no discrimination.
    # The trust model learns the actual Ia/non-Ia ratio from training data.
    p_snia = sn_prob * 0.5
    p_nonia = sn_prob * 0.5
    p_other = 1.0 - sn_prob
    result = summarize_ternary(p_snia, p_nonia, p_other)
    result["raw_snn_sn_vs_others"] = sn_prob
    return result


def _project_cats(events: list[dict[str, Any]]) -> dict[str, Any]:
    """CATS — broad class code + probability of that class (see cats_ternary)."""
    # Find events with class_name (integer label)
    class_events = [e for e in events if e.get("class_name") is not None]
    if not class_events:
        return {"prediction_type": "class_correctness", "reason": "no cats events"}

    latest = class_events[-1]
    cats_class = int(latest.get("class_name", 0))
    cats_score = float(latest.get("canonical_projection", 0) or 0)

    ternary = cats_ternary(cats_class, cats_score)
    if ternary is None:
        return {"prediction_type": "class_correctness", "reason": f"unknown cats class {cats_class}"}
    result = summarize_ternary(*ternary)
    result["raw_cats_class"] = cats_class
    result["raw_cats_score"] = cats_score
    return result


def _project_early_snia(events: list[dict[str, Any]]) -> dict[str, Any]:
    """EarlySNIa — probability of being SN Ia.

    Only triggered when SNN + RF both indicate SN.
    Score of -1 means not triggered (not enough evidence).
    """
    scores = [float(e["canonical_projection"]) for e in events
              if e.get("canonical_projection") is not None and float(e["canonical_projection"]) >= 0]
    if not scores:
        return {"prediction_type": "class_correctness", "reason": "early_snia not triggered"}

    p_snia = scores[-1]
    # EarlySNIa only triggers when SNN+RF both indicate SN, so the population
    # is pre-filtered to SN-like objects.  80/20 non-Ia SN-like / other
    # reflects this selection effect (fewer non-SN contaminants).
    p_nonia = (1.0 - p_snia) * 0.8
    p_other = (1.0 - p_snia) * 0.2
    result = summarize_ternary(p_snia, p_nonia, p_other)
    result["raw_early_snia_score"] = p_snia
    return result
