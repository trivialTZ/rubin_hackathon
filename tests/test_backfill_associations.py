from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from debass_meta.access.base import BrokerOutput
from scripts.backfill import _fetch_with_reference


class _StubAdapter:
    name = "alerce"
    semantic_type = "probability"

    def __init__(self) -> None:
        self.queries: list[str] = []

    def fetch_object(self, object_id: str) -> BrokerOutput:
        self.queries.append(object_id)
        return BrokerOutput(
            broker=self.name,
            object_id=object_id,
            query_time=1000.0,
            raw_payload={"ok": True},
            semantic_type=self.semantic_type,
            survey="ZTF",
            fields=[],
            events=[],
            availability=True,
            fixture_used=False,
        )

    def unavailable_output(self, object_id: str, **kwargs) -> BrokerOutput:
        return BrokerOutput(
            broker=self.name,
            object_id=object_id,
            query_time=1000.0,
            raw_payload={"reason": kwargs.get("reason")},
            semantic_type=self.semantic_type,
            survey=kwargs.get("survey", "ZTF"),
            fields=[],
            events=[],
            availability=False,
            fixture_used=False,
        )


def test_fetch_with_reference_routes_lsst_objects_through_associated_ztf_id() -> None:
    adapter = _StubAdapter()

    out, reference = _fetch_with_reference(
        adapter,
        "170032882292621441",
        associations={
            "170032882292621441": {
                "ztf_object_id": "ZTF23abcegjv",
                "sep_arcsec": 0.21,
                "association_kind": "position_crossmatch",
                "association_source": "alerce_conesearch",
            }
        },
    )

    assert adapter.queries == ["ZTF23abcegjv"]
    assert reference.resolution_status == "crossmatched"
    assert out.object_id == "170032882292621441"
    assert out.requested_object_id == "ZTF23abcegjv"
    assert out.associated_object_id == "ZTF23abcegjv"


def test_fetch_with_reference_returns_unavailable_when_crossmatch_is_missing() -> None:
    adapter = _StubAdapter()

    out, reference = _fetch_with_reference(adapter, "170032882292621441", associations={})

    assert adapter.queries == []
    assert reference.resolution_status == "unresolved"
    assert out.availability is False
    assert out.raw_payload["reason"] == "missing_crossmatch_association"
