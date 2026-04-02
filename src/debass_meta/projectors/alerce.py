"""ALeRCE projector logic.

Handles both legacy classifiers (lc_classifier_transient, stamp_classifier) and
the newer BHRF/ATAT/Rubin-era classifiers.

Classifier family mapping:
  Legacy ZTF:
    alerce/lc_classifier_transient     → SNIa, SNIbc, SNII, SLSN, AGN, VS, ...
    alerce/stamp_classifier            → SN, AGN, VS, asteroid, bogus

  BHRF (newer ZTF, forced photometry):
    alerce/lc_classifier_BHRF_forced_phot_transient → SNIa, SESN, SNII, SNIIn, SLSN, TDE
    alerce/lc_classifier_BHRF_forced_phot          → full taxonomy (transient + periodic + stochastic)

  ATAT (transformer, newer ZTF):
    alerce/LC_classifier_ATAT_forced_phot(beta)    → SNIa, SNIbc, SNII, SNIIn, SLSN, TDE, ...

  Stamp updates:
    alerce/stamp_classifier_2025_beta              → SN, AGN, VS, asteroid, bogus, satellite
    alerce/stamp_classifier_rubin_beta             → SN, AGN, VS, asteroid, bogus (LSST-specific)
"""
from __future__ import annotations

from typing import Any

from .base import summarize_ternary

# --- Class → ternary mapping tables ---

# Legacy LC transient classes
_LC_NONIA = {"SNIbc", "SNII", "SLSN"}
_LC_OTHER = {"AGN", "VS", "asteroid", "bogus"}

# BHRF transient classes (newer taxonomy)
_BHRF_NONIA = {"SESN", "SNII", "SNIIn", "SLSN"}
_BHRF_OTHER = {"TDE"}

# ATAT classes (superset — includes periodic/stochastic too)
_ATAT_NONIA = {"SNIbc", "SESN", "SNII", "SNIIn", "SLSN"}
_ATAT_OTHER = {"TDE", "AGN", "Blazar", "QSO", "CV/Nova", "YSO", "Microlensing",
               "CEP", "DSCT", "EA", "EB/EW", "LPV", "Periodic-Other",
               "RRLab", "RRLc", "RSCVn"}


def _classify_transient_events(
    events: list[dict[str, Any]],
    nonia_set: set[str],
    other_set: set[str],
) -> dict[str, Any]:
    """Generic ternary projection: SNIa vs nonIa_snlike vs other."""
    p_snia = 0.0
    p_nonia = 0.0
    p_other = 0.0
    found = False
    for row in events:
        cls = row.get("class_name")
        value = row.get("canonical_projection")
        if value is None:
            continue
        found = True
        value = float(value)
        if cls == "SNIa":
            p_snia += value
        elif cls in nonia_set:
            p_nonia += value
        else:
            # Everything not SNIa and not in nonia_set goes to other
            p_other += value
    if not found:
        return {"prediction_type": "class_correctness", "reason": "missing classifier probabilities"}
    return summarize_ternary(p_snia, p_nonia, p_other)


def _classify_top_level_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Top-level: Transient → nonIa_snlike (SN-like context), rest → other."""
    p_transient = 0.0
    p_other = 0.0
    found = False
    for row in events:
        cls = row.get("class_name")
        value = row.get("canonical_projection")
        if value is None:
            continue
        found = True
        value = float(value)
        if cls == "Transient":
            p_transient += value
        else:
            p_other += value
    if not found:
        return {"prediction_type": "class_correctness", "reason": "missing top-level probabilities"}
    # Transient is SN-like context (no Ia/nonIa distinction)
    return summarize_ternary(0.0, p_transient, p_other)


def _classify_stamp_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Stamp classifier: SN → nonIa_snlike (no Ia/nonIa distinction), rest → other."""
    p_nonia = 0.0
    p_other = 0.0
    found = False
    for row in events:
        cls = row.get("class_name")
        value = row.get("canonical_projection")
        if value is None:
            continue
        found = True
        value = float(value)
        if cls == "SN":
            p_nonia += value
        else:
            p_other += value
    if not found:
        return {"prediction_type": "class_correctness", "reason": "missing stamp probabilities"}
    return summarize_ternary(0.0, p_nonia, p_other)


def project_events(expert_key: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    # --- Legacy LC transient classifier ---
    if expert_key == "alerce/lc_classifier_transient":
        return _classify_transient_events(events, _LC_NONIA, _LC_OTHER)

    # --- BHRF forced-phot transient classifier ---
    if expert_key == "alerce/lc_classifier_BHRF_forced_phot_transient":
        return _classify_transient_events(events, _BHRF_NONIA, _BHRF_OTHER)

    # --- BHRF forced-phot full taxonomy (includes transient + periodic + stochastic) ---
    if expert_key == "alerce/lc_classifier_BHRF_forced_phot":
        return _classify_transient_events(events, _BHRF_NONIA, _BHRF_OTHER)

    # --- BHRF top-level classifier (Transient / Periodic / Stochastic) ---
    if expert_key == "alerce/lc_classifier_BHRF_forced_phot_top":
        return _classify_top_level_events(events)

    # --- ATAT transformer classifier ---
    if expert_key in ("alerce/LC_classifier_ATAT_forced_phot(beta)",
                      "alerce/LC_classifier_ATAT_forced_phot"):
        return _classify_transient_events(events, _ATAT_NONIA, _ATAT_OTHER)

    # --- All stamp classifiers (legacy, 2025 beta, Rubin beta) ---
    if expert_key in ("alerce/stamp_classifier",
                      "alerce/stamp_classifier_2025_beta",
                      "alerce/stamp_classifier_rubin_beta"):
        return _classify_stamp_events(events)

    return {"prediction_type": "unknown", "reason": f"unsupported alerce expert {expert_key}"}
