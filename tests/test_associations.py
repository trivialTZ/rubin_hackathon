from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from debass_meta.access.associations import (
    ResolvedObjectReference,
    load_lsst_ztf_associations,
    resolve_object_reference,
)


def test_load_lsst_ztf_associations_filters_to_matched_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "lsst_to_ztf.csv"
    csv_path.write_text(
        "\n".join(
            [
                "lsst_object_id,match_status,ztf_object_id,sep_arcsec,association_kind,association_source",
                "17001,matched,ZTF1,0.4,position_crossmatch,alerce_conesearch",
                "17002,no_match,,,"
            ]
        )
    )

    lookup = load_lsst_ztf_associations(csv_path)

    assert lookup == {
        "17001": {
            "ztf_object_id": "ZTF1",
            "sep_arcsec": 0.4,
            "association_kind": "position_crossmatch",
            "association_source": "alerce_conesearch",
            "match_count": None,
        }
    }


def test_load_lsst_ztf_associations_ignores_pandas_na_unmatched_rows(tmp_path: Path) -> None:
    import pandas as pd

    csv_path = tmp_path / "lsst_to_ztf.csv"
    pd.DataFrame(
        [
            {
                "lsst_object_id": "17001",
                "match_status": "matched",
                "ztf_object_id": "ZTF1",
                "sep_arcsec": 0.4,
            },
            {
                "lsst_object_id": "17002",
                "match_status": "no_match",
                "ztf_object_id": pd.NA,
                "sep_arcsec": pd.NA,
            },
        ]
    ).to_csv(csv_path, index=False)

    lookup = load_lsst_ztf_associations(csv_path)

    assert sorted(lookup) == ["17001"]


def test_resolve_object_reference_keeps_direct_ztf_requests() -> None:
    resolved = resolve_object_reference("ZTF23abcegjv", broker="alerce")

    assert isinstance(resolved, ResolvedObjectReference)
    assert resolved.resolution_status == "direct"
    assert resolved.requested_object_id == "ZTF23abcegjv"
    assert resolved.primary_identifier_kind == "ztf_object_id"


def test_resolve_object_reference_crossmatches_lsst_for_ztf_brokers() -> None:
    resolved = resolve_object_reference(
        "170032882292621441",
        broker="fink",
        associations={
            "170032882292621441": {
                "ztf_object_id": "ZTF23abcegjv",
                "sep_arcsec": 0.21,
                "association_kind": "position_crossmatch",
                "association_source": "alerce_conesearch",
            }
        },
    )

    assert resolved.resolution_status == "crossmatched"
    assert resolved.requested_object_id == "ZTF23abcegjv"
    assert resolved.primary_identifier_kind == "lsst_dia_object_id"
    assert resolved.requested_identifier_kind == "ztf_object_id"
    assert resolved.association_sep_arcsec == 0.21


def test_resolve_object_reference_blocks_unmatched_lsst_for_ztf_brokers() -> None:
    resolved = resolve_object_reference("170032882292621441", broker="alerce")

    assert resolved.resolution_status == "unresolved"
    assert resolved.resolution_reason == "missing_crossmatch_association"
    assert resolved.requested_object_id is None
    assert resolved.can_query is False
