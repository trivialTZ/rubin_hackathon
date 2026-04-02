from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import debass_meta.access.lasair as lasair_module
from debass_meta.access.alerce import AlerceAdapter
from debass_meta.access.fink import FinkAdapter
from debass_meta.access.lasair import LasairAdapter, extract_sherlock_label


class _DummyResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_extract_sherlock_label_reads_nested_classification() -> None:
    raw = {"lasairData": {"sherlock": {"classification": "SN"}}}
    assert extract_sherlock_label(raw) == "SN"


def test_lasair_adapter_rejects_ztf_ids_for_lsst_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("LASAIR_TOKEN", "token")
    monkeypatch.setenv("LASAIR_ENDPOINT", "https://lasair.lsst.ac.uk")

    adapter = LasairAdapter()
    out = adapter.fetch_object("ZTF21abbzjeq")

    assert out.availability is False
    assert out.fixture_used is False
    assert out.survey == "LSST"
    assert out.raw_payload["reason"] == "unsupported_identifier_for_endpoint"
    assert out.raw_payload["expected_identifier_kind"] == "lsst_dia_object_id"


def test_lasair_adapter_fetches_lsst_objects_via_api(monkeypatch) -> None:
    monkeypatch.setenv("LASAIR_TOKEN", "token")
    monkeypatch.setenv("LASAIR_ENDPOINT", "https://lasair.lsst.ac.uk")
    monkeypatch.setattr(LasairAdapter, "save_fixture", lambda self, data, path: None)

    def fake_post(url: str, headers=None, data=None, timeout=None):
        assert url == "https://lasair.lsst.ac.uk/api/object/"
        assert data == {"objectId": "170032882292621441"}
        return _DummyResponse(
            200,
            {
                "diaObjectId": "170032882292621441",
                "lasairData": {"sherlock": {"classification": "SN"}},
            },
        )

    monkeypatch.setattr(lasair_module.requests, "post", fake_post)

    adapter = LasairAdapter(timeout=5)
    out = adapter.fetch_object("170032882292621441")

    assert out.availability is True
    assert out.fixture_used is False
    assert out.survey == "LSST"
    assert out.source_endpoint == "https://lasair.lsst.ac.uk/api/object/"
    assert len(out.fields) == 1
    assert out.fields[0]["class_name"] == "SN"


def test_ztf_brokers_reject_lsst_identifiers() -> None:
    object_id = "170032882292621441"

    alerce = AlerceAdapter().fetch_object(object_id)
    fink = FinkAdapter().fetch_object(object_id)

    for out in (alerce, fink):
        assert out.availability is False
        assert out.fixture_used is False
        assert out.raw_payload["reason"] == "unsupported_identifier_for_broker"
        assert out.raw_payload["expected_identifier_kind"] == "ztf_object_id"
