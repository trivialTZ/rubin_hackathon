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
        has_association_metadata = any(
            value is not None
            for value in (
                o.primary_object_id,
                o.primary_identifier_kind,
                o.associated_object_id,
                o.association_kind,
                o.association_source,
                o.association_sep_arcsec,
            )
        )
        requested_object_id = o.requested_object_id
        if requested_object_id is None and not has_association_metadata:
            requested_object_id = o.object_id
        rows.append({
            "broker": o.broker,
            "object_id": o.object_id,
            "primary_object_id": o.primary_object_id or o.object_id,
            "requested_object_id": requested_object_id,
            "primary_identifier_kind": o.primary_identifier_kind,
            "requested_identifier_kind": o.requested_identifier_kind,
            "associated_object_id": o.associated_object_id,
            "association_kind": o.association_kind,
            "association_source": o.association_source,
            "association_sep_arcsec": o.association_sep_arcsec,
            "query_time": o.query_time,
            "survey": o.survey,
            "source_endpoint": o.source_endpoint,
            "request_params_json": json.dumps(o.request_params or {}, sort_keys=True),
            "status_code": o.status_code,
            "semantic_type": o.semantic_type,
            "availability": o.availability,
            "fixture_used": o.fixture_used,
            "payload_hash": _payload_hash(o.raw_payload),
            "events_json": json.dumps(o.events or o.fields),
            "raw_payload": json.dumps(
                {
                    **o.raw_payload,
                    "_fields": o.fields,
                    "_events": o.events or o.fields,
                }
            ),
        })

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    return out_path
