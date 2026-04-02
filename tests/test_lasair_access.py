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


def test_lasair_dual_mode_routes_ztf_ids_to_ztf_endpoint(monkeypatch) -> None:
    """Dual-mode adapter should route ZTF IDs to ZTF Lasair, not reject them."""
    monkeypatch.setenv("LASAIR_TOKEN", "token")
    monkeypatch.setenv("LASAIR_ENDPOINT", "https://lasair.lsst.ac.uk")

    adapter = LasairAdapter()
    out = adapter.fetch_object("ZTF21abbzjeq")

    # Should try ZTF endpoint, not reject as unsupported
    assert out.survey == "ZTF"
    assert "unsupported_identifier_for_endpoint" not in str(out.raw_payload.get("reason", ""))


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


def test_ztf_only_brokers_reject_lsst_identifiers() -> None:
    """Fink still rejects LSST IDs; ALeRCE v2.1+ accepts them via survey param."""
    object_id = "170032882292621441"

    fink = FinkAdapter().fetch_object(object_id)
    assert fink.availability is False
    assert fink.fixture_used is False
    assert fink.raw_payload["reason"] == "unsupported_identifier_for_broker"
    assert fink.raw_payload["expected_identifier_kind"] == "ztf_object_id"

    # ALeRCE now supports LSST objects (v2.1+ with survey="lsst")
    alerce = AlerceAdapter()
    if alerce._has_survey_param:
        out = alerce.fetch_object(object_id)
        assert out.survey == "LSST"
        # Should either get data or fail gracefully (not reject)
        assert "unsupported_identifier_for_broker" not in str(out.raw_payload.get("reason", ""))
