"""scripts/backfill.py — Fetch broker outputs and write bronze Parquet.

Usage:
    python scripts/backfill.py --broker alerce --limit 50
    python scripts/backfill.py --broker fink --objects ZTF21abbzjeq ZTF20aajnksq
    python scripts/backfill.py --broker all --limit 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from debass_meta.access import ALL_ADAPTERS
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill broker outputs to bronze layer")
    parser.add_argument("--broker", default="all", help="Broker name or 'all'")
    parser.add_argument("--objects", nargs="+", default=None, help="Object IDs to fetch")
    parser.add_argument("--from-labels", default=None, help="CSV file with object_id column")
    parser.add_argument("--limit", type=int, default=5, help="Max objects when using default list")
    parser.add_argument("--bronze-dir", default="data/bronze", help="Bronze output directory")
    args = parser.parse_args()

    if args.objects:
        object_ids = args.objects
    elif args.from_labels:
        object_ids = _load_from_labels(args.from_labels)
    else:
        object_ids = _DEFAULT_OBJECTS[: args.limit]
    bronze_dir = Path(args.bronze_dir)

    adapters = [
        cls() for cls in ALL_ADAPTERS
        if args.broker == "all" or cls().name == args.broker
    ]
    if not adapters:
        print(f"No adapter found for broker '{args.broker}'")
        sys.exit(1)

    for adapter in adapters:
        if adapter.phase > 1:
            print(f"[{adapter.name}] skipping (phase {adapter.phase} stub)")
            continue
        outputs = []
        for oid in object_ids:
            try:
                out = adapter.fetch_object(oid)
                outputs.append(out)
                label = "(fixture)" if out.fixture_used else "(live)"
                print(f"  [{adapter.name}] {oid}: {len(out.fields)} fields {label}")
            except Exception as exc:
                print(f"  [{adapter.name}] {oid}: ERROR — {exc}")
        if outputs:
            path = write_bronze(outputs, bronze_dir=bronze_dir)
            print(f"  [{adapter.name}] wrote {len(outputs)} records → {path}")


if __name__ == "__main__":
    main()
