"""Phase 4b (memory-light) — append Babamul bronze to existing silver.

The normal `normalize.py` re-reads ALL bronze files and OOMs on the SCC
login node (4.1M-row silver = 2.4 GB in arrow buffers). This script:

  1. Reads only `data/bronze/babamul_*.parquet` (the new file).
  2. Runs the same bronze→silver expansion logic.
  3. Concats the new event rows onto the existing silver parquet.
  4. De-dups so re-runs are idempotent.
  5. Writes silver back atomically (write-then-rename).

Idempotent: running twice produces the same output.

Usage:
    python scripts/_exp_babamul_silver_append.py
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


def _bronze_to_event_rows(bronze_files: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for bf in bronze_files:
        df = pd.read_parquet(bf)
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
                    "expert_key": event.get("expert_key") or _infer_expert_key(row["broker"], event),
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
    bronze_files = sorted((REPO / "data" / "bronze").glob("babamul_*.parquet"))
    if not bronze_files:
        print("FAIL no babamul bronze files found")
        return 1
    print(f"babamul bronze files: {len(bronze_files)}")

    print("expanding babamul bronze into event rows ...")
    new_rows = _bronze_to_event_rows(bronze_files)
    new_df = pd.DataFrame(new_rows)
    print(f"  new event rows: {len(new_df):,}")
    if "raw_label_or_score" in new_df.columns:
        new_df["raw_label_or_score"] = new_df["raw_label_or_score"].astype("string")
    if "classifier_version" in new_df.columns:
        new_df["classifier_version"] = new_df["classifier_version"].astype("string")

    if not silver_path.exists():
        print("no existing silver — writing fresh")
        new_df.to_parquet(silver_path, index=False)
        return 0

    print(f"reading existing silver ({silver_path}) ...")
    silver = pd.read_parquet(silver_path)
    print(f"  existing rows: {len(silver):,}; existing brokers: {sorted(silver['broker'].unique())}")

    # Drop any prior babamul rows so re-runs are idempotent
    pre = len(silver)
    silver = silver[silver["broker"] != "babamul"]
    if len(silver) < pre:
        print(f"  dropped {pre - len(silver):,} stale babamul rows for idempotent overwrite")

    print("concatenating ...")
    out = pd.concat([silver, new_df], ignore_index=True, sort=False)

    # Dedup using the same key as bronze_to_silver
    SENTINEL = -999.0
    out["_dedup_jd"] = out["event_time_jd"].fillna(SENTINEL)
    n_pre = len(out)
    out = out.drop_duplicates(
        subset=["object_id", "expert_key", "_dedup_jd", "field", "raw_label_or_score"],
        keep="last",
    )
    out = out.drop(columns=["_dedup_jd"])
    print(f"  dedup: {n_pre:,} → {len(out):,}")
    print(f"  babamul rows in final silver: {(out['broker']=='babamul').sum():,}")

    tmp = silver_path.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(silver_path)
    print(f"wrote {silver_path} ({silver_path.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
