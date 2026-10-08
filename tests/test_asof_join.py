from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from debass_meta.ingest.gold import select_events_asof


def test_select_events_asof_uses_latest_past_event_only() -> None:
    events = pd.DataFrame(
        [
            {"event_time_jd": 10.0, "temporal_exactness": "exact_alert", "field": "a"},
            {"event_time_jd": 20.0, "temporal_exactness": "exact_alert", "field": "b"},
            {"event_time_jd": 30.0, "temporal_exactness": "exact_alert", "field": "c"},
        ]
    )

    selected = select_events_asof(events, alert_jd=24.0)

    assert len(selected) == 1
    assert selected[0]["field"] == "b"


def test_select_events_asof_ignores_unavailable_rows() -> None:
    events = pd.DataFrame(
        [
            {"event_time_jd": 20.0, "temporal_exactness": "exact_alert", "field": "stale", "availability": False},
            {"event_time_jd": 10.0, "temporal_exactness": "exact_alert", "field": "live", "availability": True},
        ]
    )

    selected = select_events_asof(events, alert_jd=24.0)

    assert len(selected) == 1
    assert selected[0]["field"] == "live"


def test_local_record_nan_alert_jd_falls_back_to_alert_mjd() -> None:
    # LSST rows appended to a silver whose alert_jd column is NaN: without the fallback
    # every epoch became untimed and the gold row averaged all of them (later ones too).
    import json

    from debass_meta.ingest.gold import _local_record_to_events

    rows = [
        {"object_id": "170000000000000001", "expert": "salt3_chi2", "n_det": n, "alert_mjd": 61000.0 + n,
         "alert_jd": float("nan"), "class_probabilities": json.dumps({"Ia": p, "II": 1 - p}), "available": True}
        for n, p in ((3, 0.9), (4, 0.2), (5, 0.7))
    ]
    events = pd.DataFrame([e for r in rows for e in _local_record_to_events(r)])
    assert events["event_time_jd"].notna().all()
    assert set(events["event_time_jd"]) == {61003.0 + 2400000.5, 61004.0 + 2400000.5, 61005.0 + 2400000.5}

    selected = select_events_asof(events, alert_jd=61004.0 + 2400000.5)
    assert {e["n_det"] for e in selected} == {4}
    assert {e["class_name"]: e["canonical_projection"] for e in selected} == {"Ia": 0.2, "II": 0.8}


def test_select_events_asof_never_returns_later_timed_reruns() -> None:
    # fusion v13k: an epoch before a local expert's first output (lc_features_bv below 4 detections) used to fall
    # through to "all rerun_exact events" and get the expert's later outputs.
    events = pd.DataFrame(
        [
            {"event_time_jd": 30.0, "temporal_exactness": "rerun_exact", "n_det": 4},
            {"event_time_jd": 40.0, "temporal_exactness": "rerun_exact", "n_det": 5},
        ]
    )
    assert select_events_asof(events, alert_jd=24.0) == []
    assert [e["n_det"] for e in select_events_asof(events, alert_jd=35.0)] == [4]


def test_select_events_asof_untimed_reruns_still_selected() -> None:
    events = pd.DataFrame(
        [
            {"event_time_jd": None, "temporal_exactness": "rerun_exact", "field": "untimed"},
            {"event_time_jd": 40.0, "temporal_exactness": "rerun_exact", "field": "later"},
        ]
    )
    assert [e["field"] for e in select_events_asof(events, alert_jd=24.0)] == ["untimed"]


def test_future_selections_counts_only_non_static_experts() -> None:
    from debass_meta.ingest.gold import future_selections

    df = pd.DataFrame(
        {
            "alert_jd": [10.0, 20.0, 30.0],
            "source_event_time_jd__lc_features_bv": [9.0, 25.0, None],
            "temporal_exactness__lc_features_bv": ["rerun_exact", "rerun_exact", None],
            "source_event_time_jd__babamul": [99.0, 99.0, 99.0],
            "temporal_exactness__babamul": ["static_safe"] * 3,
            "source_event_time_jd__alerce__lc_classifier_transient": [50.0, None, None],
            "temporal_exactness__alerce__lc_classifier_transient": ["latest_object_unsafe", None, None],
        }
    )
    assert future_selections(df) == {"lc_features_bv": 1}
    assert future_selections(df.drop(index=1)) == {}


def test_local_records_without_probabilities_never_reach_a_row() -> None:
    # An unavailable (or empty) local output must leave the epoch without that expert, never borrow a later epoch.
    import json

    from debass_meta.ingest.gold import _local_record_to_events

    rows = [{"object_id": "ZTF00test", "expert": "lc_features_bv", "n_det": n, "alert_mjd": 60000.0 + n,
             "class_probabilities": json.dumps(p), "available": bool(p)}
            for n, p in ((3, {}), (4, {"snia": 0.6, "nonIa_snlike": 0.3, "other": 0.1}))]
    events = pd.DataFrame([e for r in rows for e in _local_record_to_events(r)])
    assert select_events_asof(events, alert_jd=60003.0 + 2400000.5) == []
    assert {e["n_det"] for e in select_events_asof(events, alert_jd=60004.0 + 2400000.5)} == {4}
