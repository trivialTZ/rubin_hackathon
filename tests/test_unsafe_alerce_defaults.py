from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from debass_meta.ingest.gold import build_object_epoch_snapshots



def test_snapshot_builder_excludes_unsafe_alerce_by_default(tmp_path: Path) -> None:
    lc_dir = tmp_path / "lightcurves"
    silver_dir = tmp_path / "silver"
    gold_dir = tmp_path / "gold"
    truth_dir = tmp_path / "truth"
    lc_dir.mkdir()
    silver_dir.mkdir()
    truth_dir.mkdir()

    with open(lc_dir / "ZTF1.json", "w") as fh:
        json.dump(
            [
                {"mjd": 60000.0, "magpsf": 20.0, "fid": 1, "isdiffpos": "t"},
            ],
            fh,
        )

    broker_events = pd.DataFrame(
        [
            {
                "object_id": "ZTF1",
                "broker": "alerce",
                "classifier": "lc_classifier_transient",
                "classifier_version": None,
                "expert_key": "alerce/lc_classifier_transient",
                "field": "lc_classifier_transient_SNIa",
                "class_name": "SNIa",
                "semantic_type": "probability",
                "event_scope": "object_snapshot",
                "event_time_jd": None,
                "alert_id": None,
                "n_det": None,
                "raw_label_or_score": "0.9",
                "canonical_projection": 0.9,
                "ranking": 1,
                "availability": True,
                "fixture_used": False,
                "temporal_exactness": "latest_object_unsafe",
                "payload_hash": None,
                "provenance_json": "{}",
                "survey": "ZTF",
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
                "object_id": "ZTF1",
                "final_class_raw": "snia",
                "final_class_ternary": "snia",
                "follow_proxy": 1,
                "label_source": "external",
                "label_quality": "strong",
            }
        ]
    )
    truth.to_parquet(truth_dir / "object_truth.parquet", index=False)

    out_path = build_object_epoch_snapshots(
        lc_dir=lc_dir,
        silver_dir=silver_dir,
        gold_dir=gold_dir,
        truth_path=truth_dir / "object_truth.parquet",
        max_n_det=1,
    )
    snapshots = pd.read_parquet(out_path)
    row = snapshots.iloc[0]

    assert row["avail__alerce__lc_classifier_transient"] == 0.0
    assert row["exact__alerce__lc_classifier_transient"] == 0.0
