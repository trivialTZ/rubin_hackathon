"""Truth quality assignment with source verification.

Quality tiers:
  spectroscopic — TNS has spectra from a recognized survey program + specific type
  consensus     — multiple independent classifiers agree at late time,
                  OR TNS type from a source without verification metadata
  weak          — single broker self-label, or no external confirmation

The consensus tier uses a temporal firewall: only late-time (n_det >= threshold)
classifier outputs are used, while training targets early-time (n_det <= 5).
This avoids circularity since the feature space at high n_det is genuinely
different from the early-time features the model learns on.

Source verification:
  TNS accepts classifications from diverse contributors. We assign
  "spectroscopic" quality when the classifying group is a recognized
  survey program with a well-documented reduction pipeline. For other
  sources where we lack metadata to verify the pipeline, the label is
  still used but at a lower quality tier (consensus or weak).

  Ambiguous types like generic "SN" (no subtype) are also downgraded
  because the ternary class is uncertain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
    # Credibility metadata
    source_group: str | None = None
    credibility_tier: str | None = None
    downgrade_reason: str | None = None


def assign_quality_from_tns(
    *,
    tns_type: str | None,
    tns_ternary: str | None,
    has_spectra: bool,
    broker_votes: dict[str, str] | None = None,
    credibility_tier: str | None = None,
    source_group: str | None = None,
    type_is_ambiguous: bool = False,
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
    credibility_tier : str or None
        Source credibility: "professional", "likely_professional", or "unverified".
        If None (e.g. bulk CSV mode), treated as credible for backward compat.
    source_group : str or None
        Name of the group that submitted the classification spectrum.
    type_is_ambiguous : bool
        True if the TNS type is ambiguous (e.g. generic "SN" without subtype).
        Ambiguous types are downgraded from spectroscopic even if credible.
    """
    broker_votes = broker_votes or {}

    # ── Tier 1: spectroscopic ────────────────────────────────────────
    # Requirements: has spectrum + specific type + credible source
    if has_spectra and tns_ternary is not None:
        downgrade_reason = None

        # Check 1: Is the type ambiguous? (e.g. generic "SN")
        if type_is_ambiguous:
            downgrade_reason = f"ambiguous_type:{tns_type}"

        # Check 2: Is the source credible?
        # credibility_tier=None means we don't have metadata (bulk CSV) —
        # accept as credible for backward compatibility.
        elif credibility_tier == "unverified":
            downgrade_reason = f"unverified_source:{source_group or 'unknown'}"

        if downgrade_reason is None:
            # Full spectroscopic quality — credible source + specific type
            return QualityAssignment(
                quality="spectroscopic",
                label_source="tns_spectroscopic",
                final_class_ternary=tns_ternary,
                tns_type_raw=tns_type,
                has_spectra=True,
                source_group=source_group,
                credibility_tier=credibility_tier,
            )
        else:
            # Downgraded: has spectra but source unverified or type ambiguous.
            # Falls through to consensus check below.
            pass

    # ── Tier 2: consensus ────────────────────────────────────────────

    # 2a: TNS has a classification (possibly downgraded) + broker votes agree
    if tns_ternary is not None and broker_votes:
        agreeing = [
            k for k, v in broker_votes.items()
            if v == tns_ternary
        ]
        n_agree = len(agreeing)
        n_total = len(broker_votes)
        # Need TNS + at least CONSENSUS_MIN_EXPERTS brokers to agree
        if n_agree >= CONSENSUS_MIN_EXPERTS:
            source = "tns_consensus"
            # Note if this was downgraded from spectroscopic
            if has_spectra and (type_is_ambiguous or credibility_tier == "unverified"):
                source = "tns_downgraded_consensus"
            return QualityAssignment(
                quality="consensus",
                label_source=source,
                final_class_ternary=tns_ternary,
                tns_type_raw=tns_type,
                has_spectra=has_spectra,
                consensus_experts=agreeing,
                consensus_n_agree=n_agree,
                consensus_n_total=n_total,
                source_group=source_group,
                credibility_tier=credibility_tier,
                downgrade_reason=downgrade_reason if has_spectra else None,
            )

    # 2b: consensus from brokers alone (no TNS classification),
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

    # ── Tier 3: weak ─────────────────────────────────────────────────

    # TNS type exists but either:
    # - no spectra and insufficient broker agreement, OR
    # - type is ambiguous (generic "SN"), OR
    # - classifying source lacks verification metadata
    if tns_ternary is not None:
        source = "tns_unconfirmed"
        downgrade = None
        if has_spectra and credibility_tier == "unverified":
            source = "tns_source_unverified"
            downgrade = f"unverified_source:{source_group or 'unknown'}"
        elif has_spectra and type_is_ambiguous:
            source = "tns_ambiguous_type"
            downgrade = f"ambiguous_type:{tns_type}"
        return QualityAssignment(
            quality="weak",
            label_source=source,
            final_class_ternary=tns_ternary,
            tns_type_raw=tns_type,
            has_spectra=has_spectra,
            source_group=source_group,
            credibility_tier=credibility_tier,
            downgrade_reason=downgrade,
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
