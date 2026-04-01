from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from debass_meta.ingest.silver import bronze_to_silver


def test_bronze_to_silver_skips_empty_live_payload_without_fake_event_rows(tmp_path: Path) -> None:
    bronze_dir = tmp_path / "bronze"
    silver_dir = tmp_path / "silver"
    bronze_dir.mkdir()

    bronze_df = pd.DataFrame(
        [
            {
                "broker": "fink",
                "object_id": "ZTF_EMPTY",
                "query_time": 1000.0,
                "survey": "ZTF",
                "source_endpoint": "fink",
                "request_params_json": "{}",
                "status_code": 200,
                "semantic_type": "mixed",
                "availability": True,
                "fixture_used": False,
                "payload_hash": "abc123",
                "events_json": json.dumps([]),
                "raw_payload": json.dumps({"alerts": []}),
            }
        ]
    )
    bronze_df.to_parquet(bronze_dir / "fink_empty.parquet", index=False)

    out_path = bronze_to_silver(bronze_dir=bronze_dir, silver_dir=silver_dir)
    silver_df = pd.read_parquet(out_path)

    assert out_path.exists()
    assert len(silver_df) == 0
