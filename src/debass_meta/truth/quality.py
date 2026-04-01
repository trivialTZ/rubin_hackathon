"""Three-tier truth quality assignment.

Quality tiers:
  spectroscopic — TNS has spectra + a mapped classification
  consensus     — no spectra, but N independent classifiers agree at late time
  weak          — single broker self-label, or no external confirmation

The consensus tier uses a temporal firewall: only late-time (n_det >= threshold)
classifier outputs are used, while training targets early-time (n_det <= 5).
This avoids circularity since the feature space at high n_det is genuinely
different from the early-time features the model learns on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

LabelQuality = Literal["spectroscopic", "consensus", "weak"]

# Minimum number of independent experts that must agree for consensus
CONSENSUS_MIN_EXPERTS = 3

# Minimum n_det for classifier outputs to be eligible for consensus
CONSENSUS_MIN_NDET = 10

# Minimum top-1 probability for an expert vote to count toward consensus
CONSENSUS_MIN_PROB = 0.5

# Experts eligible for consensus voting (subset of PHASE1_EXPERT_KEYS)
CONSENSUS_EXPERT_KEYS = [
    "fink/snn",
    "fink/rf_ia",
    "alerce/lc_classifier_transient",
    "alerce/stamp_classifier",
]


@dataclass
class QualityAssignment:
    """Result of quality assessment for one object."""
    quality: LabelQuality
    label_source: str
    final_class_ternary: str | None
    tns_type_raw: str | None = None
    has_spectra: bool = False
    consensus_experts: list[str] | None = None
    consensus_n_agree: int = 0
    consensus_n_total: int = 0


def assign_quality_from_tns(
    *,
    tns_type: str | None,
    tns_ternary: str | None,
    has_spectra: bool,
    broker_votes: dict[str, str] | None = None,
) -> QualityAssignment:
    """Assign truth quality tier for a single object.

    Parameters
    ----------
    tns_type : str or None
        Raw TNS classification type (e.g. "SN Ia").
    tns_ternary : str or None
        Mapped ternary label from map_tns_type_to_ternary().
    has_spectra : bool
        Whether TNS reports spectroscopic confirmation.
    broker_votes : dict mapping expert_key -> ternary prediction, optional
        Late-time broker consensus votes (only from n_det >= CONSENSUS_MIN_NDET).
    """
    # Tier 1: spectroscopic
    if has_spectra and tns_ternary is not None:
        return QualityAssignment(
            quality="spectroscopic",
            label_source="tns_spectroscopic",
            final_class_ternary=tns_ternary,
            tns_type_raw=tns_type,
            has_spectra=True,
        )

    # Tier 2: consensus — TNS has a classification but no spectra,
    # AND broker votes agree
    broker_votes = broker_votes or {}
    if tns_ternary is not None and broker_votes:
        agreeing = [
            k for k, v in broker_votes.items()
            if v == tns_ternary
        ]
        n_agree = len(agreeing)
        n_total = len(broker_votes)
        # Need TNS + at least CONSENSUS_MIN_EXPERTS brokers to agree
        if n_agree >= CONSENSUS_MIN_EXPERTS:
            return QualityAssignment(
                quality="consensus",
                label_source="tns_consensus",
                final_class_ternary=tns_ternary,
                tns_type_raw=tns_type,
                has_spectra=False,
                consensus_experts=agreeing,
                consensus_n_agree=n_agree,
                consensus_n_total=n_total,
            )

    # Tier 2b: consensus from brokers alone (no TNS classification),
    # if enough independent experts agree on the same class
    if not tns_ternary and broker_votes:
        class_counts: dict[str, list[str]] = {}
        for expert_key, vote in broker_votes.items():
            class_counts.setdefault(vote, []).append(expert_key)
        for cls, experts in sorted(class_counts.items(), key=lambda x: -len(x[1])):
            if len(experts) >= CONSENSUS_MIN_EXPERTS:
                return QualityAssignment(
                    quality="consensus",
                    label_source="broker_consensus",
                    final_class_ternary=cls,
                    tns_type_raw=tns_type,
                    has_spectra=False,
                    consensus_experts=experts,
                    consensus_n_agree=len(experts),
                    consensus_n_total=len(broker_votes),
                )

    # Tier 3: weak
    # TNS has a type but no spectra and insufficient broker agreement
    if tns_ternary is not None:
        return QualityAssignment(
            quality="weak",
            label_source="tns_unconfirmed",
            final_class_ternary=tns_ternary,
            tns_type_raw=tns_type,
            has_spectra=False,
        )

    # No TNS classification at all
    return QualityAssignment(
        quality="weak",
        label_source="no_tns_match",
        final_class_ternary=None,
        tns_type_raw=None,
        has_spectra=False,
    )


def collect_broker_votes(
    events_df: Any,
    object_id: str,
    *,
    min_n_det: int = CONSENSUS_MIN_NDET,
    min_prob: float = CONSENSUS_MIN_PROB,
    expert_keys: list[str] | None = None,
) -> dict[str, str]:
    """Collect late-time ternary votes from broker events for consensus.

    Only considers events where n_det >= min_n_det (temporal firewall).
    Returns a dict mapping expert_key -> predicted ternary class.
    """
    import pandas as pd

    expert_keys = expert_keys or CONSENSUS_EXPERT_KEYS
    votes: dict[str, str] = {}

    if events_df is None or len(events_df) == 0:
        return votes

    obj_events = events_df[events_df["object_id"] == object_id]
    if len(obj_events) == 0:
        return votes

    # Only late-time events
    if "n_det" in obj_events.columns:
        obj_events = obj_events[obj_events["n_det"].fillna(0) >= min_n_det]
    if len(obj_events) == 0:
        return votes

    for expert_key in expert_keys:
        expert_events = obj_events[obj_events["expert_key"] == expert_key]
        if len(expert_events) == 0:
            continue

        # Get the latest events for this expert
        if "event_time_jd" in expert_events.columns:
            timed = expert_events[expert_events["event_time_jd"].notna()]
            if len(timed) > 0:
                max_t = timed["event_time_jd"].max()
                expert_events = timed[timed["event_time_jd"] == max_t]

        # Extract ternary probabilities from projections
        ternary_vote = _vote_from_events(expert_events, min_prob=min_prob)
        if ternary_vote is not None:
            votes[expert_key] = ternary_vote

    return votes


def _vote_from_events(
    expert_events: Any,
    *,
    min_prob: float,
) -> str | None:
    """Extract a ternary vote from a set of expert event rows.

    Looks at class_name + raw_label_or_score to find the dominant class.
    """
    import pandas as pd

    probs: dict[str, float] = {}
    for _, row in expert_events.iterrows():
        class_name = str(row.get("class_name") or "").strip()
        score = row.get("canonical_projection")
        if score is None:
            score = row.get("raw_label_or_score")
        try:
            score = float(score)
        except (ValueError, TypeError):
            continue

        if class_name in ("snia", "SN Ia", "SNIa"):
            probs["snia"] = max(probs.get("snia", 0.0), score)
        elif class_name in ("nonIa_snlike", "SN", "CC"):
            probs["nonIa_snlike"] = max(probs.get("nonIa_snlike", 0.0), score)
        elif class_name in ("other",):
            probs["other"] = max(probs.get("other", 0.0), score)

    if not probs:
        return None

    best_class = max(probs, key=lambda k: probs[k])
    if probs[best_class] >= min_prob:
        return best_class
    return None
