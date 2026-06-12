"""Babamul projector — alert-time boolean flags from properties + survey_matches.

Babamul is wired as a context expert (semantic_type=binary_score,
temporal_exactness=static_safe) — the per-object REST endpoint returns
the latest alert payload's flags, not per-epoch evolution.

Verified discriminative power on the 1,920-ZTF cohort
(see ``project_babamul_verification.md``):
  babamul_star_flag             snia 0.3% / other 68% (strongly informative)
  babamul_rock_flag             snia 0% / other 26% (asteroid kill-switch)
  babamul_stationary_flag       snia 100% / other 64% (complements rock)
  babamul_near_brightstar_flag  snia 1% / other 12% (moderate)
  babamul_xmatch_lsst           degenerate on ZTF cohort (no LSST backfill)
  babamul_xmatch_ztf            degenerate (counterpart side empty)
"""
from __future__ import annotations

from typing import Any

_FEATURES = (
    "babamul_star_flag",
    "babamul_near_brightstar_flag",
    "babamul_rock_flag",
    "babamul_stationary_flag",
    "babamul_xmatch_lsst",
    "babamul_xmatch_ztf",
)


def project_events(expert_key: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        return {"prediction_type": "context_only", "reason": "missing babamul context"}
    out: dict[str, Any] = {"prediction_type": "context_only"}
    by_field = {e.get("field"): e for e in events}
    for fname in _FEATURES:
        ev = by_field.get(fname)
        if ev is None:
            out[fname] = None
            continue
        v = ev.get("canonical_projection")
        if v is None:
            v = ev.get("raw_label_or_score")
        out[fname] = float(v) if v is not None else None
    return out
