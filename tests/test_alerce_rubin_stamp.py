"""ALeRCE Rubin stamp classifier name drift (fusion_v10 fix #4).

Live LSST ALeRCE returns dated classifier variants
(``stamp_classifier_rubin_beta_20260421``) alongside the base
``stamp_classifier_rubin_beta``.  Contract:

* access layer: any classifier matching the prefix maps to the UNCHANGED
  registry expert key ``alerce/stamp_classifier_rubin_beta`` (raw classifier
  name preserved on the event);
* projector: accepts dated expert keys via prefix match and, when several
  variants are present for the same alert, uses ONLY the newest dated one
  (no probability-mass double counting).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from debass_meta.access.alerce import AlerceAdapter, canonical_alerce_expert_key
from debass_meta.projectors.alerce import (
    _select_rubin_stamp_events,
    project_events,
)
from debass_meta.projectors.base import project_expert_events


def _stamp_event(classifier: str, class_name: str, prob: float) -> dict:
    return {
        "classifier": classifier,
        "class_name": class_name,
        "canonical_projection": prob,
        "temporal_exactness": "static_safe",
    }


# ---------------------------------------------------------------------------
# access layer: expert-key normalization
# ---------------------------------------------------------------------------

def test_canonical_expert_key_prefix_match():
    assert canonical_alerce_expert_key("stamp_classifier_rubin_beta") == \
        "alerce/stamp_classifier_rubin_beta"
    assert canonical_alerce_expert_key("stamp_classifier_rubin_beta_20260421") == \
        "alerce/stamp_classifier_rubin_beta"
    # non-Rubin classifiers keep their exact key
    assert canonical_alerce_expert_key("stamp_classifier") == \
        "alerce/stamp_classifier"
    assert canonical_alerce_expert_key("stamp_classifier_2025_beta") == \
        "alerce/stamp_classifier_2025_beta"
    assert canonical_alerce_expert_key("lc_classifier_transient") == \
        "alerce/lc_classifier_transient"


def test_extract_fields_normalizes_dated_variant():
    adapter = AlerceAdapter.__new__(AlerceAdapter)  # no client needed
    raw = {"probabilities": [
        {"classifier_name": "stamp_classifier_rubin_beta_20260421",
         "class_name": "SN", "probability": 0.8},
        {"classifier_name": "stamp_classifier_rubin_beta",
         "class_name": "SN", "probability": 0.3},
        {"classifier_name": "stamp_classifier",
         "class_name": "SN", "probability": 0.5},
    ]}
    fields = adapter._extract_fields(raw)
    keys = [f["expert_key"] for f in fields]
    assert keys[0] == "alerce/stamp_classifier_rubin_beta"
    assert keys[1] == "alerce/stamp_classifier_rubin_beta"
    assert keys[2] == "alerce/stamp_classifier"        # untouched
    # raw classifier names preserved for provenance / variant selection
    assert fields[0]["classifier"] == "stamp_classifier_rubin_beta_20260421"
    assert fields[1]["classifier"] == "stamp_classifier_rubin_beta"
    # dated variant is still a stamp classifier -> static_safe
    assert fields[0]["temporal_exactness"] == "static_safe"


# ---------------------------------------------------------------------------
# silver normalization: dated keys already baked into bronze are rewritten
# ---------------------------------------------------------------------------

def test_bronze_to_silver_canonicalizes_dated_rubin_stamp_keys(tmp_path):
    """Bronze fetched under pre-canonicalization code stores dated expert
    keys verbatim; re-normalization must rewrite them, otherwise the gold
    builder's exact-key lookup silently drops the Rubin stamp expert."""
    import json

    import pandas as pd

    from debass_meta.ingest.silver import bronze_to_silver

    bronze_dir = tmp_path / "bronze"
    silver_dir = tmp_path / "silver"
    bronze_dir.mkdir()

    def _event(expert_key, classifier, score=0.8):
        return {
            "field": "SN",
            "raw_label_or_score": score,
            "canonical_projection": score,
            "semantic_type": "probability",
            "classifier": classifier,
            "class_name": "SN",
            "event_scope": "object_snapshot",
            "temporal_exactness": "static_safe",
            **({"expert_key": expert_key} if expert_key else {}),
        }

    bronze_df = pd.DataFrame([{
        "broker": "alerce",
        "object_id": "3100000001",
        "query_time": 1000.0,
        "survey": "LSST",
        "source_endpoint": "alerce",
        "request_params_json": "{}",
        "status_code": 200,
        "semantic_type": "probability",
        "availability": True,
        "fixture_used": False,
        "payload_hash": "h1",
        "events_json": json.dumps([
            # dated key baked in by pre-v10 fetch code
            _event("alerce/stamp_classifier_rubin_beta_20260421",
                   "stamp_classifier_rubin_beta_20260421", score=0.8),
            # no expert_key at all → inferred from classifier
            # (distinct score so silver dedup keeps the row)
            _event(None, "stamp_classifier_rubin_beta_20260421", score=0.7),
            # non-Rubin classifier untouched
            _event("alerce/stamp_classifier", "stamp_classifier", score=0.5),
        ]),
        "raw_payload": json.dumps({"_events": []}),
    }])
    bronze_df.to_parquet(bronze_dir / "alerce_test.parquet", index=False)

    silver_df = pd.read_parquet(
        bronze_to_silver(bronze_dir=bronze_dir, silver_dir=silver_dir))
    keys = silver_df["expert_key"].tolist()
    assert keys[0] == "alerce/stamp_classifier_rubin_beta"
    assert keys[1] == "alerce/stamp_classifier_rubin_beta"
    assert keys[2] == "alerce/stamp_classifier"
    # raw classifier names preserved for the projector's variant preference
    assert silver_df["classifier"].tolist()[0] == "stamp_classifier_rubin_beta_20260421"


# ---------------------------------------------------------------------------
# projector: variant preference + prefix dispatch
# ---------------------------------------------------------------------------

def test_projector_prefers_newest_dated_variant():
    events = [
        _stamp_event("stamp_classifier_rubin_beta", "SN", 0.2),
        _stamp_event("stamp_classifier_rubin_beta", "bogus", 0.8),
        _stamp_event("stamp_classifier_rubin_beta_20260421", "SN", 0.9),
        _stamp_event("stamp_classifier_rubin_beta_20260421", "AGN", 0.1),
    ]
    out = project_events("alerce/stamp_classifier_rubin_beta", events)
    # newest dated variant wins: SN 0.9 / AGN 0.1, NOT summed with the base
    assert out["p_nonIa_snlike"] == pytest.approx(0.9)
    assert out["p_other"] == pytest.approx(0.1)


def test_projector_prefers_newer_of_two_dated_variants():
    events = [
        _stamp_event("stamp_classifier_rubin_beta_20250101", "SN", 0.1),
        _stamp_event("stamp_classifier_rubin_beta_20250101", "bogus", 0.9),
        _stamp_event("stamp_classifier_rubin_beta_20260421", "SN", 0.7),
        _stamp_event("stamp_classifier_rubin_beta_20260421", "VS", 0.3),
    ]
    out = project_events("alerce/stamp_classifier_rubin_beta", events)
    assert out["p_nonIa_snlike"] == pytest.approx(0.7)
    assert out["p_other"] == pytest.approx(0.3)


def test_projector_single_undated_variant_unchanged():
    events = [
        _stamp_event("stamp_classifier_rubin_beta", "SN", 0.6),
        _stamp_event("stamp_classifier_rubin_beta", "bogus", 0.4),
    ]
    out = project_events("alerce/stamp_classifier_rubin_beta", events)
    assert out["p_nonIa_snlike"] == pytest.approx(0.6)
    assert out["p_other"] == pytest.approx(0.4)


def test_projector_accepts_dated_expert_key():
    """Prefix dispatch: a dated expert_key string routes to the stamp
    projector instead of falling through to 'unsupported'."""
    events = [_stamp_event("stamp_classifier_rubin_beta_20260421", "SN", 1.0)]
    out = project_events("alerce/stamp_classifier_rubin_beta_20260421", events)
    assert out["p_nonIa_snlike"] == pytest.approx(1.0)
    # and through the registry-level dispatcher too
    out2 = project_expert_events("alerce/stamp_classifier_rubin_beta", events)
    assert out2["p_nonIa_snlike"] == pytest.approx(1.0)


def test_variant_selection_passthrough_without_classifier_field():
    """Legacy events (no classifier field) behave exactly as before."""
    events = [
        {"class_name": "SN", "canonical_projection": 0.5},
        {"class_name": "AGN", "canonical_projection": 0.5},
    ]
    assert _select_rubin_stamp_events(events) == events
    out = project_events("alerce/stamp_classifier_rubin_beta", events)
    assert out["p_nonIa_snlike"] == pytest.approx(0.5)
