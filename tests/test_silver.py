from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from debass_meta.ingest.silver import bronze_to_silver


def test_bronze_to_silver_serializes_mixed_raw_label_or_score(tmp_path: Path) -> None:
    bronze_dir = tmp_path / "bronze"
    silver_dir = tmp_path / "silver"
    bronze_dir.mkdir()

    bronze_df = pd.DataFrame(
        [
            {
                "broker": "alerce",
                "object_id": "ZTF_TEST",
                "query_time": "2026-03-30T00:00:00Z",
                "survey": "ZTF",
                "raw_payload": json.dumps(
                    {
                        "_fields": [
                            {"field": "probability", "raw_label_or_score": 0.91},
                            {"field": "details", "raw_label_or_score": {"_fields": []}},
                            {"field": "note", "raw_label_or_score": '{"_fields": []}'},
                        ]
                    }
                ),
                "semantic_type": "score",
                "fixture_used": False,
                "payload_hash": "abc123",
            }
        ]
    )
    bronze_df.to_parquet(bronze_dir / "alerce_test.parquet", index=False)

    out_path = bronze_to_silver(bronze_dir=bronze_dir, silver_dir=silver_dir)
    silver_df = pd.read_parquet(out_path)

    assert out_path.exists()
    assert pd.api.types.is_string_dtype(silver_df["raw_label_or_score"])
    assert silver_df["raw_label_or_score"].tolist() == [
        "0.91",
        '{"_fields": []}',
        '{"_fields": []}',
    ]
