from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.discover_nightly as discover_nightly


class _DummyResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_query_lasair_recent_supports_lsst_mode(monkeypatch) -> None:
    monkeypatch.setenv("LASAIR_TOKEN", "token")
    monkeypatch.setenv("LASAIR_ENDPOINT", "https://lasair.lsst.ac.uk")

    class FakeClient:
        def query(self, selected, tables, conditions, limit=1000, offset=0):
            assert "diaObjectId" in selected
            assert tables == "objects"
            assert "lastDiaSourceMjdTai" in conditions
            return [
                {
                    "diaObjectId": 170032882292621441,
                    "ra": 62.3,
                    "decl": -46.1,
                    "nDiaSources": 1,
                    "lastDiaSourceMjdTai": 61091.03,
                },
                {
                    "diaObjectId": 170032882292621442,
                    "ra": 62.4,
                    "decl": -46.2,
                    "nDiaSources": 2,
                    "lastDiaSourceMjdTai": 61091.04,
                },
            ]

    fake_module = types.SimpleNamespace(lasair_client=lambda token, endpoint=None: FakeClient())
    monkeypatch.setitem(sys.modules, "lasair", fake_module)

    def fake_post(url: str, headers=None, data=None, timeout=None):
        if str(data.get("objectId")) == "170032882292621441":
            payload = {"lasairData": {"sherlock": {"classification": "SN"}}}
        else:
            payload = {"lasairData": {"sherlock": {}}}
        return _DummyResponse(200, payload)

    monkeypatch.setattr(discover_nightly.requests, "post", fake_post)

    df = discover_nightly.query_lasair_recent(since_mjd=61090.0, max_candidates=5)

    assert len(df) == 1
    assert df.iloc[0]["object_id"] == "170032882292621441"
    assert df.iloc[0]["discovery_broker"] == "lasair"
    assert int(df.iloc[0]["n_det"]) == 1
