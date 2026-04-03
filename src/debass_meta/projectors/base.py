"""Shared projector helpers."""
from __future__ import annotations

from math import log
from typing import Any

# Expert registry: expert_key → (survey_scope, projector_module_name)
#   survey_scope: "ztf"  = only applies to ZTF objects
#                 "lsst" = only applies to LSST objects
#                 "any"  = applies to any survey
EXPERT_REGISTRY: dict[str, tuple[str, str]] = {
    # --- Fink (ZTF alerts only for now) ---
    "fink/snn":                        ("ztf",  "fink"),
    "fink/rf_ia":                      ("ztf",  "fink"),
    # --- Local experts (any survey) ---
    "parsnip":                         ("any",  "local"),
    "supernnova":                      ("any",  "local"),
    "alerce_lc":                       ("any",  "local"),
    # --- ALeRCE legacy (ZTF, older objects) ---
    "alerce/lc_classifier_transient":  ("ztf",  "alerce"),
    "alerce/stamp_classifier":         ("ztf",  "alerce"),
    # --- ALeRCE BHRF/ATAT (ZTF, newer objects with forced photometry) ---
    "alerce/lc_classifier_BHRF_forced_phot_transient": ("ztf", "alerce"),
    "alerce/lc_classifier_BHRF_forced_phot_top":       ("ztf", "alerce"),
    "alerce/LC_classifier_ATAT_forced_phot(beta)":     ("ztf", "alerce"),
    "alerce/stamp_classifier_2025_beta":               ("ztf", "alerce"),
    # --- ALeRCE Rubin/LSST classifiers ---
    "alerce/stamp_classifier_rubin_beta":              ("lsst", "alerce"),
    # --- Fink LSST classifiers (per-alert, exact_alert!) ---
    "fink_lsst/snn":                                   ("lsst", "fink_lsst"),
    "fink_lsst/cats":                                  ("lsst", "fink_lsst"),
    "fink_lsst/early_snia":                            ("lsst", "fink_lsst"),
    # --- Lasair (both surveys) ---
    "lasair/sherlock":                 ("any",  "lasair"),
    # --- Pitt-Google Broker (SuperNNova via BigQuery) ---
    "pittgoogle/supernnova_lsst":      ("lsst", "pittgoogle"),
    "pittgoogle/supernnova_ztf":       ("ztf",  "pittgoogle"),
}

# Backward-compatible flat list — Phase 1 experts (original 8)
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

# All registered experts (Phase 1 + new ALeRCE classifiers)
ALL_EXPERT_KEYS = list(EXPERT_REGISTRY.keys())


def get_expert_keys(survey: str | None = None) -> list[str]:
    """Return expert keys applicable to a survey.

    Parameters
    ----------
    survey : str or None
        ``"ZTF"``, ``"LSST"``, or ``None`` (return all experts).
    """
    if survey is None:
        return list(EXPERT_REGISTRY.keys())
    survey_lower = survey.lower()
    return [
        key
        for key, (scope, _) in EXPERT_REGISTRY.items()
        if scope in ("any", survey_lower)
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
    """Route events to the appropriate projector. Returns ternary projection dict.

    Never raises — returns an error dict on failure so the gold builder
    can mark the expert as unavailable without crashing the pipeline.
    """
    if not events:
        return {"prediction_type": "unknown", "reason": "no events"}
    try:
        return _dispatch_projector(expert_key, events)
    except Exception as exc:
        return {"prediction_type": "unknown", "reason": f"projector error: {exc}"}


def _dispatch_projector(expert_key: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    if expert_key == "fink/snn" or expert_key == "fink/rf_ia":
        from .fink import project_events

        return project_events(expert_key, events)
    if expert_key.startswith("fink_lsst/"):
        from .fink_lsst import project_events

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
    if expert_key.startswith("pittgoogle/"):
        from .pittgoogle import project_events

        return project_events(expert_key, events)
    return {"prediction_type": "unknown", "reason": f"no projector for {expert_key}"}
