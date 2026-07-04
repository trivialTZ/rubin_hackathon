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
      (live LSST also publishes dated variants, e.g.
       stamp_classifier_rubin_beta_20260421 — accepted via prefix match;
       the newest dated variant wins when several are present for one alert)
"""
from __future__ import annotations

import re
from typing import Any

from .base import summarize_ternary

# Rubin stamp classifier name drift: prefix-match dated variants and prefer
# the newest dated version.  The registry key stays
# `alerce/stamp_classifier_rubin_beta` (see access/alerce.py).
_RUBIN_STAMP_PREFIX = "stamp_classifier_rubin_beta"
_RUBIN_STAMP_DATE_RE = re.compile(r"^_(\d+)$")

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


def _rubin_stamp_variant_rank(classifier: str) -> tuple[int, int, str]:
    """Sort key for Rubin stamp variants: dated beats undated, newer date
    beats older; the name itself breaks ties deterministically."""
    suffix = classifier[len(_RUBIN_STAMP_PREFIX):]
    match = _RUBIN_STAMP_DATE_RE.match(suffix)
    if match:
        return (1, int(match.group(1)), classifier)
    return (0, -1, classifier)


def _select_rubin_stamp_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the newest Rubin stamp variant's events.

    Live LSST ALeRCE can return probabilities from both
    ``stamp_classifier_rubin_beta`` and dated variants
    (``stamp_classifier_rubin_beta_YYYYMMDD``) for the same alert; summing
    across variants would double-count probability mass.  Events without a
    recognizable variant name are passed through unchanged.
    """
    variants: dict[str, list[dict[str, Any]]] = {}
    for row in events:
        classifier = str(row.get("classifier") or "")
        if not classifier.startswith(_RUBIN_STAMP_PREFIX):
            continue
        variants.setdefault(classifier, []).append(row)
    if not variants:
        return events
    newest = max(variants, key=_rubin_stamp_variant_rank)
    return variants[newest]


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

    # --- All stamp classifiers (legacy, 2025 beta, Rubin beta incl. dated
    #     variants via prefix match) ---
    if expert_key.startswith(f"alerce/{_RUBIN_STAMP_PREFIX}"):
        return _classify_stamp_events(_select_rubin_stamp_events(events))
    if expert_key in ("alerce/stamp_classifier",
                      "alerce/stamp_classifier_2025_beta"):
        return _classify_stamp_events(events)

    return {"prediction_type": "unknown", "reason": f"unsupported alerce expert {expert_key}"}
