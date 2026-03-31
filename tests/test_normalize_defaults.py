from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from debass_meta.ingest.silver import bronze_to_silver



def test_normalize_default_writes_only_silver_outputs(tmp_path: Path) -> None:
    bronze_dir = tmp_path / "bronze"
    silver_dir = tmp_path / "silver"
    gold_dir = tmp_path / "gold"
    bronze_dir.mkdir()

    bronze_df = pd.DataFrame(
        [
            {
                "broker": "fink",
                "object_id": "ZTF_TEST",
                "query_time": 1000.0,
                "survey": "ZTF",
                "source_endpoint": "fink",
                "request_params_json": "{}",
                "status_code": 200,
                "semantic_type": "mixed",
                "availability": True,
                "fixture_used": False,
                "payload_hash": "abc123",
                "events_json": json.dumps(
                    [
                        {
                            "field": "snn_snia_vs_nonia",
                            "raw_label_or_score": 0.7,
                            "canonical_projection": 0.7,
                            "semantic_type": "probability",
                            "expert_key": "fink/snn",
                            "event_scope": "alert",
                            "temporal_exactness": "exact_alert",
                            "event_time_jd": 2460001.5,
                            "n_det": 2,
                        }
                    ]
                ),
                "raw_payload": json.dumps({"_events": []}),
            }
        ]
    )
    bronze_df.to_parquet(bronze_dir / "fink_test.parquet", index=False)

    out_path = bronze_to_silver(bronze_dir=bronze_dir, silver_dir=silver_dir)

    assert out_path.name == "broker_events.parquet"
    assert (silver_dir / "broker_events.parquet").exists()
    assert not (gold_dir / "snapshots.parquet").exists()
