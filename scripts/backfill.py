"""scripts/backfill.py — Fetch broker outputs and write bronze Parquet.

Usage:
    python scripts/backfill.py --broker alerce --limit 50
    python scripts/backfill.py --broker fink --objects ZTF21abbzjeq ZTF20aajnksq
    python scripts/backfill.py --broker all --limit 20
    python scripts/backfill.py --broker alerce --from-labels data/labels.csv --parallel 8
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from debass_meta.access import ALL_ADAPTERS
from debass_meta.access.associations import (
    apply_reference_metadata,
    load_lsst_ztf_associations,
    resolve_object_reference,
)
from debass_meta.access.identifiers import infer_identifier_kind
from debass_meta.ingest.bronze import write_bronze

# Default test objects (known ZTF transients)
_DEFAULT_OBJECTS = [
    "ZTF21abbzjeq",
    "ZTF20aajnksq",
    "ZTF18abukavn",
    "ZTF19aadyppr",
    "ZTF19aaiqmgl",
]


def _load_from_labels(path: str) -> list[str]:
    import csv
    oids = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            oids.append(row["object_id"])
    return oids


def _fetch_with_reference(
    adapter,
    object_id: str,
    *,
    associations: dict[str, dict[str, object]] | None = None,
):
    reference = resolve_object_reference(
        object_id,
        broker=adapter.name,
        associations=associations,
    )
    if not reference.can_query:
        out = adapter.unavailable_output(
            object_id,
            survey=reference.survey,
            identifier_kind=reference.primary_identifier_kind,
            reason=reference.resolution_reason or "unresolved_object_association",
            raw_payload_extra={
                "requested_object_id": reference.requested_object_id,
                "associated_object_id": reference.associated_object_id,
                "association_kind": reference.association_kind,
                "association_source": reference.association_source,
                "association_sep_arcsec": reference.association_sep_arcsec,
            },
        )
        return apply_reference_metadata(out, reference), reference

    out = adapter.fetch_object(reference.requested_object_id)
    return apply_reference_metadata(out, reference), reference


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill broker outputs to bronze layer")
    parser.add_argument("--broker", default="all", help="Broker name or 'all'")
    parser.add_argument("--objects", nargs="+", default=None, help="Object IDs to fetch")
    parser.add_argument("--from-labels", default=None, help="CSV file with object_id column")
    parser.add_argument("--limit", type=int, default=5, help="Max objects when using default list")
    parser.add_argument("--association-csv", default=None, help="Optional LSST→ZTF association CSV")
    parser.add_argument("--bronze-dir", default="data/bronze", help="Bronze output directory")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel threads for API calls (default: 1 = sequential)")
    args = parser.parse_args()

    if args.objects:
        object_ids = args.objects
    elif args.from_labels:
        object_ids = _load_from_labels(args.from_labels)
    else:
        object_ids = _DEFAULT_OBJECTS[: args.limit]
    bronze_dir = Path(args.bronze_dir)
    associations = load_lsst_ztf_associations(Path(args.association_csv)) if args.association_csv else {}

    adapters = [
        cls() for cls in ALL_ADAPTERS
        if args.broker == "all" or cls().name == args.broker
    ]
    if not adapters:
        print(f"No adapter found for broker '{args.broker}'")
        sys.exit(1)

    n_workers = max(1, args.parallel)

    for adapter in adapters:
        if adapter.phase > 1:
            print(f"[{adapter.name}] skipping (phase {adapter.phase} stub)")
            continue

        def _do_fetch(oid: str):
            out, reference = _fetch_with_reference(
                adapter, oid, associations=associations,
            )
            if not out.availability:
                status = out.raw_payload.get("reason", "unavailable")
            else:
                status = "fixture" if out.fixture_used else "live"
            route = (
                f"{oid} <- {reference.requested_object_id}"
                if reference.requested_object_id and reference.requested_object_id != oid
                else oid
            )
            kind = infer_identifier_kind(oid)
            return out, f"  [{adapter.name}] {route}: {len(out.fields)} fields ({kind}, {status})"

        outputs = []
        if n_workers > 1:
            print(f"  [{adapter.name}] fetching {len(object_ids)} objects with {n_workers} threads...")
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_do_fetch, oid): oid for oid in object_ids}
                for future in as_completed(futures):
                    oid = futures[future]
                    try:
                        out, msg = future.result()
                        outputs.append(out)
                        print(msg)
                    except Exception as exc:
                        print(f"  [{adapter.name}] {oid}: ERROR — {exc}")
        else:
            for oid in object_ids:
                try:
                    out, msg = _do_fetch(oid)
                    outputs.append(out)
                    print(msg)
                except Exception as exc:
                    print(f"  [{adapter.name}] {oid}: ERROR — {exc}")

        if outputs:
            path = write_bronze(outputs, bronze_dir=bronze_dir)
            print(f"  [{adapter.name}] wrote {len(outputs)} records → {path}")


if __name__ == "__main__":
    main()
