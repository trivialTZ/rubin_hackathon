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
            survey="ZTF" if object_id.startswith("ZTF") else "LSST",
            fields=[{"expert_key": "alerce/stamp"}],
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


class _StubZTFOnlyAdapter(_StubAdapter):
    """Simulates alerce_history which only supports ZTF."""
    name = "alerce_history"


def test_dual_survey_broker_queries_lsst_directly() -> None:
    """ALeRCE (dual-survey) queries LSST objects directly, not via ZTF association."""
    adapter = _StubAdapter()

    results = _fetch_with_reference(
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

    # Should query LSST ID directly, PLUS ZTF counterpart as supplement
    assert "170032882292621441" in adapter.queries
    assert "ZTF23abcegjv" in adapter.queries
    assert len(results) == 2  # LSST native + ZTF supplement

    # Primary result uses LSST ID
    out, ref = results[0]
    assert out.object_id == "170032882292621441"
    assert ref.resolution_status == "direct"

    # Supplement uses ZTF ID but stamps primary as LSST
    out2, ref2 = results[1]
    assert out2.object_id == "170032882292621441"  # apply_reference_metadata
    assert ref2.requested_object_id == "ZTF23abcegjv"


def test_ztf_only_broker_routes_through_association() -> None:
    """alerce_history (ZTF-only) routes LSST objects through association table."""
    adapter = _StubZTFOnlyAdapter()

    results = _fetch_with_reference(
        adapter,
        "170032882292621441",
        associations={
            "170032882292621441": {
                "ztf_object_id": "ZTF23abcegjv",
                "sep_arcsec": 0.21,
            }
        },
    )

    assert len(results) == 1
    assert adapter.queries == ["ZTF23abcegjv"]
    out, ref = results[0]
    assert out.object_id == "170032882292621441"
    assert ref.resolution_status == "crossmatched"


def test_ztf_only_broker_blocks_unmatched_lsst() -> None:
    adapter = _StubZTFOnlyAdapter()

    results = _fetch_with_reference(adapter, "170032882292621441", associations={})

    assert len(results) == 1
    out, ref = results[0]
    assert ref.resolution_status == "unresolved"
    assert out.availability is False


def test_dual_survey_broker_without_association_still_queries_lsst() -> None:
    """ALeRCE should still query LSST directly even without association table."""
    adapter = _StubAdapter()

    results = _fetch_with_reference(adapter, "170032882292621441", associations={})

    assert len(results) == 1  # Just the direct LSST query
    assert adapter.queries == ["170032882292621441"]
    out, ref = results[0]
    assert ref.resolution_status == "direct"
    assert out.availability is True
