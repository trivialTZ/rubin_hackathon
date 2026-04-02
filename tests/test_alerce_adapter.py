from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from debass_meta.access.alerce import AlerceAdapter


def test_alerce_fetch_lightcurve_uses_json_format(monkeypatch) -> None:
    calls: list[str] = []

    class FakeClient:
        def query_lightcurve(self, oid, format="json"):
            calls.append(format)
            return {"detections": [], "non_detections": []}

    adapter = AlerceAdapter()
    adapter._client = FakeClient()
    monkeypatch.setattr(AlerceAdapter, "save_fixture", lambda self, data, path: None)

    lightcurve = adapter.fetch_lightcurve("ZTF_TEST")

    assert calls == ["json"]
    assert lightcurve == {"detections": [], "non_detections": []}
