from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from debass_meta.ingest.gold import build_object_epoch_snapshots


def test_snapshot_builder_uses_associated_lightcurve_source_for_lsst_primary(tmp_path: Path) -> None:
    lc_dir = tmp_path / "lightcurves"
    silver_dir = tmp_path / "silver"
    gold_dir = tmp_path / "gold"
    truth_dir = tmp_path / "truth"
    lc_dir.mkdir()
    silver_dir.mkdir()
    truth_dir.mkdir()

    with open(lc_dir / "ZTF23abcegjv.json", "w") as fh:
        json.dump(
            [
                {"mjd": 60000.0, "magpsf": 20.0, "fid": 1, "isdiffpos": "t"},
                {"mjd": 60001.0, "magpsf": 19.8, "fid": 2, "isdiffpos": "t"},
            ],
            fh,
        )
    with open(lc_dir / "170032882292621441.json", "w") as fh:
        json.dump(
            [
                {"mjd": 60000.0, "magpsf": 99.0, "fid": 1, "isdiffpos": "t"},
            ],
            fh,
        )

    broker_events = pd.DataFrame(
        [
            {
                "object_id": "170032882292621441",
                "broker": "lasair",
                "classifier": "sherlock",
                "classifier_version": None,
                "expert_key": "lasair/sherlock",
                "field": "sherlock_classification",
                "class_name": "SN",
                "semantic_type": "crossmatch_tag",
                "event_scope": "static_context",
                "event_time_jd": None,
                "alert_id": None,
                "n_det": None,
                "raw_label_or_score": "SN",
                "canonical_projection": None,
                "ranking": 1,
                "availability": True,
                "fixture_used": False,
                "temporal_exactness": "static_safe",
                "payload_hash": None,
                "provenance_json": "{}",
                "survey": "LSST",
                "source_endpoint": None,
                "request_params_json": None,
                "status_code": None,
                "query_time_unix": None,
            }
        ]
    )
    broker_events.to_parquet(silver_dir / "broker_events.parquet", index=False)

    truth = pd.DataFrame(
        [
            {
                "object_id": "170032882292621441",
                "final_class_raw": "snia",
                "final_class_ternary": "snia",
                "follow_proxy": 1,
                "label_source": "external",
                "label_quality": "strong",
            }
        ]
    )
    truth.to_parquet(truth_dir / "object_truth.parquet", index=False)

    association_csv = tmp_path / "lsst_to_ztf.csv"
    association_csv.write_text(
        "\n".join(
            [
                "lsst_object_id,match_status,ztf_object_id,sep_arcsec,association_kind,association_source",
                "170032882292621441,matched,ZTF23abcegjv,0.21,position_crossmatch,alerce_conesearch",
            ]
        )
    )

    out_path = build_object_epoch_snapshots(
        lc_dir=lc_dir,
        silver_dir=silver_dir,
        gold_dir=gold_dir,
        truth_path=truth_dir / "object_truth.parquet",
        object_ids=["170032882292621441"],
        association_path=association_csv,
        max_n_det=2,
    )
    snapshots = pd.read_parquet(out_path).sort_values("n_det")
    first_row = snapshots.iloc[0]

    assert len(snapshots) == 2
    assert first_row["object_id"] == "170032882292621441"
    assert first_row["lightcurve_source_object_id"] == "ZTF23abcegjv"
    assert first_row["lightcurve_source_identifier_kind"] == "ztf_object_id"
    assert first_row["lightcurve_association_kind"] == "position_crossmatch"
    assert first_row["lightcurve_association_sep_arcsec"] == 0.21
    assert first_row["avail__lasair__sherlock"] == 1.0
