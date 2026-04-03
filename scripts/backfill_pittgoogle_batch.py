"""Batch backfill Pitt-Google SuperNNova scores via BigQuery.

Instead of querying per-object (each query scans the full table = 0.5-8GB),
this script runs TWO queries total:
  1. One for ALL LSST objects  (~0.5 GB scan)
  2. One for ALL ZTF objects   (~8 GB scan)

The results are saved as bronze Parquet, identical to what the per-object
adapter produces, so the rest of the pipeline (normalize → gold) works unchanged.

Usage:
    python scripts/backfill_pittgoogle_batch.py --from-labels data/labels.csv
    python scripts/backfill_pittgoogle_batch.py --objects ZTF21abbzjeq 313853517428162578
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


def _batch_query_lsst(
    client,
    object_ids: list[str],
    chunk_size: int = 2000,
) -> dict[str, list[dict]]:
    """Query LSST SuperNNova for all objects in one pass.

    Returns {object_id: [row_dicts]}.
    """
    results: dict[str, list[dict]] = {}

    # Chunk to avoid SQL IN clause limits
    for i in range(0, len(object_ids), chunk_size):
        chunk = object_ids[i:i + chunk_size]
        id_list = ",".join(chunk)
        sql = (
            f"SELECT diaSourceId, diaObjectId, prob_class0, prob_class1, "
            f"predicted_class, kafkaPublishTimestamp "
            f"FROM {_LSST_SUPERNNOVA_TABLE} "
            f"WHERE diaObjectId IN ({id_list}) "
            f"ORDER BY diaObjectId, kafkaPublishTimestamp"
        )
        print(f"  LSST batch query: {len(chunk)} objects (chunk {i // chunk_size + 1})...",
              flush=True)
        df = client.query(sql)
        rows = json.loads(df.to_json(orient="records"))
        print(f"    → {len(rows):,} rows returned", flush=True)

        for row in rows:
            oid = str(row.get("diaObjectId", ""))
            results.setdefault(oid, []).append(row)

    return results


def _batch_query_ztf(
    client,
    object_ids: list[str],
    chunk_size: int = 2000,
) -> dict[str, list[dict]]:
    """Query ZTF SuperNNova for all objects in one pass."""
    results: dict[str, list[dict]] = {}

    for i in range(0, len(object_ids), chunk_size):
        chunk = object_ids[i:i + chunk_size]
        id_list = ",".join(f"'{x}'" for x in chunk)
        sql = (
            f"SELECT objectId, candid, prob_class0, prob_class1, predicted_class "
            f"FROM {_ZTF_SUPERNNOVA_TABLE} "
            f"WHERE objectId IN ({id_list}) "
            f"ORDER BY objectId, candid"
        )
        print(f"  ZTF batch query: {len(chunk)} objects (chunk {i // chunk_size + 1})...",
              flush=True)
        df = client.query(sql)
        rows = json.loads(df.to_json(orient="records"))
        print(f"    → {len(rows):,} rows returned", flush=True)

        for row in rows:
            oid = str(row.get("objectId", ""))
            results.setdefault(oid, []).append(row)

    return results


def _rows_to_broker_output(
    adapter: PittAdapter,
    object_id: str,
    rows: list[dict],
    survey: str,
    expert_key: str,
) -> BrokerOutput:
    """Convert raw BigQuery rows into a BrokerOutput."""
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
        description="Batch backfill Pitt-Google SuperNNova via BigQuery (2 queries total)"
    )
    parser.add_argument("--objects", nargs="+", default=None)
    parser.add_argument("--from-labels", default=None)
    parser.add_argument("--bronze-dir", default="data/bronze")
    parser.add_argument("--save-fixtures", action="store_true",
                        help="Also save per-object fixture JSON files")
    args = parser.parse_args()

    if args.objects:
        object_ids = args.objects
    elif args.from_labels:
        object_ids = _load_from_labels(args.from_labels)
    else:
        print("Provide --objects or --from-labels")
        sys.exit(1)

    # Split by survey
    lsst_ids = [oid for oid in object_ids if infer_identifier_kind(oid) == "lsst_dia_object_id"]
    ztf_ids = [oid for oid in object_ids if infer_identifier_kind(oid) == "ztf_object_id"]
    print(f"Objects: {len(lsst_ids)} LSST + {len(ztf_ids)} ZTF = {len(object_ids)} total")

    # Import BigQuery client
    try:
        import pittgoogle
        client = pittgoogle.bigquery.Client()
    except Exception as e:
        print(f"Cannot connect to BigQuery: {e}")
        sys.exit(1)

    adapter = PittAdapter()
    bronze_dir = Path(args.bronze_dir)
    all_outputs: list[BrokerOutput] = []

    # ---- LSST batch ----
    if lsst_ids:
        print(f"\n=== LSST batch ({len(lsst_ids)} objects) ===")
        lsst_results = _batch_query_lsst(client, lsst_ids)
        print(f"  Objects with data: {len(lsst_results)}")

        n_live = 0
        for oid in lsst_ids:
            rows = lsst_results.get(oid, [])
            out = _rows_to_broker_output(
                adapter, oid, rows, survey="LSST",
                expert_key="pittgoogle/supernnova_lsst",
            )
            if not rows:
                out.availability = False
            all_outputs.append(out)
            if out.availability:
                n_live += 1
                if args.save_fixtures:
                    fixture_path = _FIXTURE_DIR / f"{oid}_supernnova.json"
                    adapter.save_fixture({"rows": rows, "survey": "LSST"}, fixture_path)

        print(f"  Live: {n_live}, Unavailable: {len(lsst_ids) - n_live}")

    # ---- ZTF batch ----
    if ztf_ids:
        print(f"\n=== ZTF batch ({len(ztf_ids)} objects) ===")
        ztf_results = _batch_query_ztf(client, ztf_ids)
        print(f"  Objects with data: {len(ztf_results)}")

        n_live = 0
        for oid in ztf_ids:
            rows = ztf_results.get(oid, [])
            out = _rows_to_broker_output(
                adapter, oid, rows, survey="ZTF",
                expert_key="pittgoogle/supernnova_ztf",
            )
            if not rows:
                out.availability = False
            all_outputs.append(out)
            if out.availability:
                n_live += 1
                if args.save_fixtures:
                    fixture_path = _FIXTURE_DIR / f"{oid}_supernnova.json"
                    adapter.save_fixture({"rows": rows, "survey": "ZTF"}, fixture_path)

        print(f"  Live: {n_live}, Unavailable: {len(ztf_ids) - n_live}")

    # ---- Write bronze ----
    if all_outputs:
        available = [o for o in all_outputs if o.availability]
        if available:
            path = write_bronze(available, bronze_dir=bronze_dir)
            print(f"\nWrote {len(available)} available records → {path}")
        else:
            print("\nNo available records to write")

        # Also write unavailable for completeness
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
