"""Silver layer: normalise bronze Parquet records into per-field rows.

Design rule: raw_label_or_score is ALWAYS preserved verbatim.
canonical_projection is stored separately and never overwrites it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_BRONZE_DIR = Path("data/bronze")
_SILVER_DIR = Path("data/silver")


def bronze_to_silver(
    bronze_dir: Path = _BRONZE_DIR,
    silver_dir: Path = _SILVER_DIR,
) -> Path:
    """Expand all bronze parquet files into a silver broker_outputs table.

    Returns the path of the written silver file.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError("pandas is required for silver transforms") from e

    bronze_files = sorted(bronze_dir.glob("*.parquet"))
    if not bronze_files:
        raise FileNotFoundError(f"No bronze parquet files found in {bronze_dir}")

    silver_rows: list[dict[str, Any]] = []

    for bf in bronze_files:
        df = pd.read_parquet(bf)
        for _, row in df.iterrows():
            raw_payload = json.loads(row["raw_payload"]) if isinstance(row["raw_payload"], str) else row["raw_payload"]
            # Re-extract fields using the per-broker field list if stored
            # Fall back to treating raw_payload itself as the single record
            fields = raw_payload.get("_fields", [])
            if not fields:
                # No pre-extracted fields — emit one row with raw payload
                silver_rows.append({
                    "broker": row["broker"],
                    "object_id": row["object_id"],
                    "query_time": row["query_time"],
                    "survey": row.get("survey", "ZTF"),
                    "field": "_raw",
                    "raw_label_or_score": json.dumps(raw_payload),
                    "semantic_type": row["semantic_type"],
                    "canonical_projection": None,
                    "classifier": None,
                    "class": None,
                    "fixture_used": bool(row["fixture_used"]),
                    "payload_hash": row.get("payload_hash"),
                })
            else:
                for f in fields:
                    silver_rows.append({
                        "broker": row["broker"],
                        "object_id": row["object_id"],
                        "query_time": row["query_time"],
                        "survey": row.get("survey", "ZTF"),
                        "field": f.get("field", "_raw"),
                        "raw_label_or_score": f.get("raw_label_or_score"),
                        "semantic_type": f.get("semantic_type", row["semantic_type"]),
                        "canonical_projection": f.get("canonical_projection"),
                        "classifier": f.get("classifier"),
                        "class": f.get("class"),
                        "fixture_used": bool(row["fixture_used"]),
                        "payload_hash": row.get("payload_hash"),
                    })

    silver_dir.mkdir(parents=True, exist_ok=True)
    out_path = silver_dir / "broker_outputs.parquet"

    silver_df = pd.DataFrame(silver_rows)
    silver_df.to_parquet(out_path, index=False)
    return out_path
