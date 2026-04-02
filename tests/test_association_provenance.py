from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from debass_meta.access.base import BrokerOutput
from debass_meta.ingest.bronze import write_bronze
from debass_meta.ingest.silver import bronze_to_silver


def test_bronze_to_silver_preserves_association_provenance(tmp_path: Path) -> None:
    bronze_dir = tmp_path / "bronze"
    silver_dir = tmp_path / "silver"

    output = BrokerOutput(
        broker="alerce",
        object_id="170032882292621441",
        query_time=1000.0,
        raw_payload={"probabilities": []},
        semantic_type="probability",
        survey="LSST",
        source_endpoint="alerce.query_probabilities",
        request_params={"oid": "ZTF23abcegjv"},
        primary_object_id="170032882292621441",
        requested_object_id="ZTF23abcegjv",
        primary_identifier_kind="lsst_dia_object_id",
        requested_identifier_kind="ztf_object_id",
        associated_object_id="ZTF23abcegjv",
        association_kind="position_crossmatch",
        association_source="alerce_conesearch",
        association_sep_arcsec=0.21,
        fields=[
            {
                "field": "lc_classifier_SN",
                "raw_label_or_score": 0.9,
                "semantic_type": "probability",
                "canonical_projection": 0.9,
                "classifier": "lc_classifier",
                "class_name": "SN",
                "expert_key": "alerce/lc_classifier",
                "event_scope": "object_snapshot",
                "temporal_exactness": "latest_object_unsafe",
            }
        ],
        events=[
            {
                "field": "lc_classifier_SN",
                "raw_label_or_score": 0.9,
                "semantic_type": "probability",
                "canonical_projection": 0.9,
                "classifier": "lc_classifier",
                "class_name": "SN",
                "expert_key": "alerce/lc_classifier",
                "event_scope": "object_snapshot",
                "temporal_exactness": "latest_object_unsafe",
            }
        ],
        availability=True,
        fixture_used=False,
    )
    write_bronze([output], bronze_dir=bronze_dir)

    out_path = bronze_to_silver(bronze_dir=bronze_dir, silver_dir=silver_dir)
    silver_df = pd.read_parquet(out_path)
    row = silver_df.iloc[0]
    provenance = json.loads(row["provenance_json"])

    assert row["object_id"] == "170032882292621441"
    assert row["requested_object_id"] == "ZTF23abcegjv"
    assert row["associated_object_id"] == "ZTF23abcegjv"
    assert row["association_kind"] == "position_crossmatch"
    assert row["association_source"] == "alerce_conesearch"
    assert row["association_sep_arcsec"] == 0.21
    assert provenance["requested_object_id"] == "ZTF23abcegjv"
    assert provenance["association_sep_arcsec"] == 0.21
