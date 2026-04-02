from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from debass_meta.access.alerce import AlerceAdapter


def test_alerce_fetch_lightcurve_uses_json_format(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeClient:
        def query_lightcurve(self, oid, format="json", survey="ztf"):
            calls.append({"format": format, "survey": survey})
            return {"detections": [], "non_detections": []}

        def query_probabilities(self, oid, format="json", survey="ztf", index=None, sort=None):
            return []

    adapter = AlerceAdapter()
    adapter._client = FakeClient()
    adapter._has_survey_param = True
    monkeypatch.setattr(AlerceAdapter, "save_fixture", lambda self, data, path: None)

    lightcurve = adapter.fetch_lightcurve("ZTF_TEST")

    assert calls[0]["format"] == "json"
    assert lightcurve == {"detections": [], "non_detections": []}
