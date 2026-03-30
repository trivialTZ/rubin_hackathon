"""Bronze layer: write raw BrokerOutput records to Parquet.

Design rule: bronze files are append-only and never modified.
Each call writes one partition file named by broker + timestamp.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

_BRONZE_DIR = Path("data/bronze")


def _payload_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def write_bronze(
    outputs,  # Sequence[BrokerOutput]
    bronze_dir: Path = _BRONZE_DIR,
) -> Path:
    """Serialise a list of BrokerOutput records to a bronze Parquet file.

    Returns the path of the written file.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError("pandas is required for bronze writes") from e

    if not outputs:
        raise ValueError("No outputs to write")

    broker = outputs[0].broker
    ts = int(outputs[0].query_time)
    bronze_dir.mkdir(parents=True, exist_ok=True)
    out_path = bronze_dir / f"{broker}_{ts}.parquet"

    rows = []
    for o in outputs:
        rows.append({
            "broker": o.broker,
            "object_id": o.object_id,
            "query_time": o.query_time,
            "survey": "ZTF",
            "semantic_type": o.semantic_type,
            "availability": o.availability,
            "fixture_used": o.fixture_used,
            "payload_hash": _payload_hash(o.raw_payload),
            "raw_payload": json.dumps({**o.raw_payload, "_fields": o.fields}),
        })

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    return out_path
