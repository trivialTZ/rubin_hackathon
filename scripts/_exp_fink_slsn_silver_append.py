"""Memory-light append of new Fink ZTF SLSN-RF events to existing silver.

The newest fink bronze includes a `slsn_score` column. The full
``normalize.py`` re-reads ALL bronze (~6 GB) and OOMs on the SCC login node.
This script reads only the NEWEST fink bronze, extracts events where
``expert_key == 'fink/slsn'``, and appends them to the existing silver
``broker_events.parquet``.

It does NOT touch existing fink/snn or fink/rf_ia rows (those come from the
prior normalize run and are already correct).

Idempotent: dropping any existing fink/slsn rows before append makes re-runs
deterministic.

Usage:
    python scripts/_exp_fink_slsn_silver_append.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pandas as pd

from debass_meta.ingest.silver import (
    _default_event_scope,
    _default_exactness,
    _infer_expert_key,
    _serialize_raw_label_or_score,
)


def _bronze_to_slsn_event_rows(bronze_file: Path) -> list[dict]:
    """Expand one fink bronze file into event rows, keeping only fink/slsn."""
    rows: list[dict] = []
    df = pd.read_parquet(bronze_file)
    for _, row in df.iterrows():
        requested_object_id = (
            row["requested_object_id"]
            if "requested_object_id" in row.index
            else row["object_id"]
        )
        primary_object_id = (
            row["primary_object_id"]
            if "primary_object_id" in row.index
            else row["object_id"]
        )
        raw_payload = (
            json.loads(row["raw_payload"])
            if isinstance(row["raw_payload"], str)
            else row["raw_payload"]
        )
        events = raw_payload.get("_events") or raw_payload.get("_fields", [])
        if isinstance(row.get("events_json"), str) and not events:
            try:
                events = json.loads(row["events_json"])
            except json.JSONDecodeError:
                events = []
        if not events:
            continue
        for ranking, event in enumerate(events, start=1):
            expert_key = event.get("expert_key") or _infer_expert_key(row["broker"], event)
            if expert_key != "fink/slsn":
                continue
            event_time_jd = event.get("event_time_jd")
            if event_time_jd is None:
                event_time_jd = event.get("alert_jd")
            rows.append({
                "object_id": row["object_id"],
                "primary_object_id": primary_object_id,
                "requested_object_id": requested_object_id,
                "primary_identifier_kind": row.get("primary_identifier_kind"),
                "requested_identifier_kind": row.get("requested_identifier_kind"),
                "associated_object_id": row.get("associated_object_id"),
                "association_kind": row.get("association_kind"),
                "association_source": row.get("association_source"),
                "association_sep_arcsec": row.get("association_sep_arcsec"),
                "broker": row["broker"],
                "classifier": event.get("classifier"),
                "classifier_version": event.get("classifier_version"),
                "expert_key": expert_key,
                "field": event.get("field", "_raw"),
                "class_name": event.get("class_name") or event.get("class"),
                "semantic_type": event.get("semantic_type", row["semantic_type"]),
                "event_scope": event.get("event_scope") or _default_event_scope(row["broker"]),
                "event_time_jd": float(event_time_jd) if event_time_jd is not None else None,
                "alert_id": event.get("alert_id"),
                "n_det": int(event["n_det"]) if event.get("n_det") is not None else None,
                "raw_label_or_score": _serialize_raw_label_or_score(
                    event.get("raw_label_or_score")
                ),
                "canonical_projection": event.get("canonical_projection"),
                "ranking": event.get("ranking", ranking),
                "availability": bool(row.get("availability", True)),
                "fixture_used": bool(row["fixture_used"]),
                "temporal_exactness": event.get("temporal_exactness")
                    or _default_exactness(row["broker"]),
                "payload_hash": row.get("payload_hash"),
                "provenance_json": json.dumps(
                    {
                        "broker": row["broker"],
                        "source_endpoint": row.get("source_endpoint"),
                        "request_params_json": row.get("request_params_json"),
                    },
                    sort_keys=True,
                ),
                "survey": row.get("survey", "ZTF"),
                "source_endpoint": row.get("source_endpoint"),
                "request_params_json": row.get("request_params_json"),
                "status_code": row.get("status_code"),
                "query_time_unix": row.get("query_time"),
            })
    return rows


def main() -> int:
    silver_path = REPO / "data" / "silver" / "broker_events.parquet"
    bronze_files = sorted((REPO / "data" / "bronze").glob("fink_*.parquet"))
    if not bronze_files:
        print("FAIL no fink bronze files found")
        return 1

    # Use only the newest bronze — older ones predate the slsn_score column
    newest = bronze_files[-1]
    print(f"using newest fink bronze: {newest.name} ({newest.stat().st_size / 1e6:.0f} MB)")

    print("expanding fink/slsn events ...")
    new_rows = _bronze_to_slsn_event_rows(newest)
    new_df = pd.DataFrame(new_rows)
    print(f"  new fink/slsn event rows: {len(new_df):,}")
    if len(new_df) == 0:
        print("WARN no slsn events extracted — check that bronze includes slsn_score column")
        return 1
    if "raw_label_or_score" in new_df.columns:
        new_df["raw_label_or_score"] = new_df["raw_label_or_score"].astype("string")
    if "classifier_version" in new_df.columns:
        new_df["classifier_version"] = new_df["classifier_version"].astype("string")

    if not silver_path.exists():
        print("no existing silver — writing fresh (this is unexpected)")
        new_df.to_parquet(silver_path, index=False)
        return 0

    print(f"reading existing silver ({silver_path}) ...")
    silver = pd.read_parquet(silver_path)
    print(f"  existing rows: {len(silver):,}")

    pre = len(silver)
    silver = silver[silver["expert_key"] != "fink/slsn"]
    if len(silver) < pre:
        print(f"  dropped {pre - len(silver):,} stale fink/slsn rows for idempotent overwrite")

    print("concatenating ...")
    out = pd.concat([silver, new_df], ignore_index=True, sort=False)

    SENTINEL = -999.0
    out["_dedup_jd"] = out["event_time_jd"].fillna(SENTINEL)
    n_pre = len(out)
    out = out.drop_duplicates(
        subset=["object_id", "expert_key", "_dedup_jd", "field", "raw_label_or_score"],
        keep="last",
    )
    out = out.drop(columns=["_dedup_jd"])
    print(f"  dedup: {n_pre:,} → {len(out):,}")
    print(f"  fink/slsn rows in final silver: {(out['expert_key']=='fink/slsn').sum():,}")

    # Coerce mixed-type columns to string for parquet compatibility
    for col in ("alert_id", "raw_label_or_score", "classifier_version", "classifier",
                "class_name", "field", "associated_object_id", "association_kind",
                "association_source", "primary_identifier_kind", "requested_identifier_kind",
                "primary_object_id", "requested_object_id", "source_endpoint",
                "request_params_json", "payload_hash", "canonical_projection",
                "provenance_json", "broker", "expert_key", "semantic_type",
                "event_scope", "temporal_exactness", "survey"):
        if col in out.columns:
            out[col] = out[col].astype("string")

    tmp = silver_path.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(silver_path)
    print(f"wrote {silver_path} ({silver_path.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
