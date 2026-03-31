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
