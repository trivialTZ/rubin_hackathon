"""Batch backfill Pitt-Google SuperNNova scores via BigQuery — cost-safe.

Pitt-Google public tables are NOT clustered on objectId, so every WHERE
query is a full table scan.  This means:
  - 1 query per table (all IDs in one IN clause) is far cheaper than chunking.
  - Identical queries are cached for 24 h → repeat runs cost 0 bytes.

This script:
  1. Builds one SQL per survey (LSST / ZTF) with ALL object IDs in one IN.
  2. Runs ``dry_run=True`` first to estimate bytes scanned.
  3. Aborts (by default) if any query would scan more than ``--max-gb`` GB.
  4. Executes the real query and writes bronze parquet rows.

UPSILoN (lsst.upsilon) is intentionally skipped — its schema is per-band
(u_label, u_probability, ..., y_probability) and the projector expects a
single predicted_class/probability pair.  Fix projector first.

Usage:
    python scripts/backfill_pittgoogle_batch.py --from-labels data/labels.csv
    python scripts/backfill_pittgoogle_batch.py --objects ZTF21abbzjeq 313853517428162578
    python scripts/backfill_pittgoogle_batch.py --from-labels data/labels.csv --dry-run  # cost estimate only
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from debass_meta.access.identifiers import infer_identifier_kind
from debass_meta.access.pitt import (
    PittAdapter,
    _LSST_SUPERNNOVA_TABLE,
    _ZTF_SUPERNNOVA_TABLE,
    _FIXTURE_DIR,
)
from debass_meta.ingest.bronze import write_bronze
from debass_meta.access.base import BrokerOutput


def _load_from_labels(path: str) -> list[str]:
    oids = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            oids.append(row["object_id"])
    return oids


def _build_lsst_sql(object_ids: list[str]) -> str:
    id_list = ",".join(object_ids)
    return (
        "SELECT diaSourceId, diaObjectId, prob_class0, prob_class1, "
        "predicted_class, kafkaPublishTimestamp "
        f"FROM {_LSST_SUPERNNOVA_TABLE} "
        f"WHERE diaObjectId IN ({id_list}) "
        "ORDER BY diaObjectId, kafkaPublishTimestamp"
    )


def _build_ztf_sql(object_ids: list[str]) -> str:
    id_list = ",".join(f"'{x}'" for x in object_ids)
    return (
        "SELECT objectId, candid, prob_class0, prob_class1, predicted_class "
        f"FROM {_ZTF_SUPERNNOVA_TABLE} "
        f"WHERE objectId IN ({id_list}) "
        "ORDER BY objectId, candid"
    )


def _dry_run_gb(client, sql: str) -> float:
    """Return estimated bytes (GB) the query will scan, without executing."""
    from google.cloud import bigquery
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False),
    )
    return job.total_bytes_processed / 1024**3


def _run_query(client, sql: str) -> list[dict]:
    """Execute a query; return rows as list of plain dicts.

    BigQuery TIMESTAMP/DATETIME columns come back as Python ``datetime`` objects —
    convert them to Unix epoch milliseconds so downstream code (which expects
    numeric/string timestamps matching the pittgoogle-client JSON format) works.
    """
    import datetime as _dt
    job = client.query(sql)

    def _norm(v):
        if isinstance(v, _dt.datetime):
            if v.tzinfo is None:
                v = v.replace(tzinfo=_dt.timezone.utc)
            return int(v.timestamp() * 1000)
        if isinstance(v, _dt.date):
            return v.isoformat()
        return v

    return [{k: _norm(v) for k, v in r.items()} for r in job.result()]


def _rows_to_broker_output(
    adapter: PittAdapter,
    object_id: str,
    rows: list[dict],
    survey: str,
    expert_key: str,
) -> BrokerOutput:
    raw = {"rows": rows, "survey": survey}
    fields = adapter._extract_supernnova_fields(raw, survey=survey, expert_key=expert_key)

    id_kind = "lsst_dia_object_id" if survey == "LSST" else "ztf_object_id"
    return BrokerOutput(
        broker=adapter.name,
        object_id=object_id,
        query_time=time.time(),
        raw_payload=raw,
        semantic_type=adapter.semantic_type,
        survey=survey,
        source_endpoint=_LSST_SUPERNNOVA_TABLE if survey == "LSST" else _ZTF_SUPERNNOVA_TABLE,
        request_params={"batch_query": True},
        primary_object_id=object_id,
        requested_object_id=object_id,
        primary_identifier_kind=id_kind,
        requested_identifier_kind=id_kind,
        fields=fields,
        events=fields,
        availability=bool(fields),
        fixture_used=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch backfill Pitt-Google SuperNNova via BigQuery (cost-safe single-query mode)"
    )
    parser.add_argument("--objects", nargs="+", default=None)
    parser.add_argument("--from-labels", default=None)
    parser.add_argument("--bronze-dir", default="data/bronze")
    parser.add_argument("--save-fixtures", action="store_true")
    parser.add_argument("--max-gb", type=float, default=20.0,
                        help="Abort if any query would scan more than this many GB (default: 20)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate cost only; do not execute real queries")
    args = parser.parse_args()

    if args.objects:
        object_ids = args.objects
    elif args.from_labels:
        object_ids = _load_from_labels(args.from_labels)
    else:
        print("Provide --objects or --from-labels")
        sys.exit(1)

    lsst_ids = [oid for oid in object_ids if infer_identifier_kind(oid) == "lsst_dia_object_id"]
    ztf_ids = [oid for oid in object_ids if infer_identifier_kind(oid) == "ztf_object_id"]
    print(f"Objects: {len(lsst_ids)} LSST + {len(ztf_ids)} ZTF = {len(object_ids)} total")

    from google.cloud import bigquery
    client = bigquery.Client()
    print(f"Billing project: {client.project}")

    # ----- Cost gate -----
    total_gb = 0.0
    plans: list[tuple[str, str, list[str], str]] = []  # (survey, sql, ids, expert_key)
    if lsst_ids:
        sql = _build_lsst_sql(lsst_ids)
        gb = _dry_run_gb(client, sql)
        print(f"  LSST SNN:  {gb:6.3f} GB  ({len(lsst_ids)} ids)")
        total_gb += gb
        plans.append(("LSST", sql, lsst_ids, "pittgoogle/supernnova_lsst"))
    if ztf_ids:
        sql = _build_ztf_sql(ztf_ids)
        gb = _dry_run_gb(client, sql)
        print(f"  ZTF  SNN:  {gb:6.3f} GB  ({len(ztf_ids)} ids)")
        total_gb += gb
        plans.append(("ZTF", sql, ztf_ids, "pittgoogle/supernnova_ztf"))

    print(f"  TOTAL:     {total_gb:6.3f} GB  ({100*total_gb/1024:.2f}% of 1 TB free tier)")

    for survey, _sql, _ids, _ek in plans:
        # Check individual queries too — a single >max-gb query should still abort.
        pass  # already printed per-plan

    if any(_dry_run_gb(client, p[1]) > args.max_gb for p in plans):
        print(f"ABORT: at least one query exceeds --max-gb={args.max_gb} GB. "
              "Re-run with --max-gb <N> to override if intentional.")
        sys.exit(2)

    if args.dry_run:
        print("--dry-run specified, not executing real queries.")
        sys.exit(0)

    # ----- Real execution -----
    adapter = PittAdapter()
    bronze_dir = Path(args.bronze_dir)
    all_outputs: list[BrokerOutput] = []

    for survey, sql, ids, expert_key in plans:
        print(f"\n=== {survey} batch ({len(ids)} objects) ===")
        rows = _run_query(client, sql)
        print(f"  rows returned: {len(rows):,}")

        id_key = "diaObjectId" if survey == "LSST" else "objectId"
        grouped: dict[str, list[dict]] = {}
        for r in rows:
            grouped.setdefault(str(r.get(id_key, "")), []).append(r)
        print(f"  objects with data: {len(grouped)}")

        n_live = 0
        for oid in ids:
            obj_rows = grouped.get(oid, [])
            out = _rows_to_broker_output(
                adapter, oid, obj_rows, survey=survey, expert_key=expert_key
            )
            if not obj_rows:
                out.availability = False
            all_outputs.append(out)
            if out.availability:
                n_live += 1
                if args.save_fixtures:
                    fixture_path = _FIXTURE_DIR / f"{oid}_supernnova.json"
                    adapter.save_fixture({"rows": obj_rows, "survey": survey}, fixture_path)
        print(f"  live: {n_live}, empty: {len(ids) - n_live}")

    if all_outputs:
        available = [o for o in all_outputs if o.availability]
        if available:
            path = write_bronze(available, bronze_dir=bronze_dir)
            print(f"\nWrote {len(available)} available records → {path}")
        unavailable = [o for o in all_outputs if not o.availability]
        if unavailable:
            path2 = write_bronze(unavailable, bronze_dir=bronze_dir)
            print(f"Wrote {len(unavailable)} unavailable records → {path2}")

    total_live = sum(1 for o in all_outputs if o.availability)
    total_fields = sum(len(o.fields) for o in all_outputs)
    print(f"\n=== Summary ===")
    print(f"Total objects: {len(all_outputs)}")
    print(f"With PGB data: {total_live} ({100 * total_live / max(1, len(all_outputs)):.1f}%)")
    print(f"Total events: {total_fields:,}")


if __name__ == "__main__":
    main()
