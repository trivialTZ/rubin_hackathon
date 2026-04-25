"""Build exact object-epoch snapshots.

Sharding
--------
For large training-set rebuilds (e.g. v6e.2), use --shard-id/--n-shards
to split work across an SGE array job. Each task processes 1/N of the
sorted objects-csv list:

    python scripts/build_object_epoch_snapshots.py --shard-id 0 --n-shards 8 \\
        --output data/gold/object_epoch_snapshots_v6e2.shard-0-of-8.parquet
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from debass_meta.ingest.gold import build_object_epoch_snapshots


def _load_object_ids(path: Path | None) -> list[str] | None:
    if path is None or not path.exists():
        return None
    with open(path) as fh:
        return [row["object_id"] for row in csv.DictReader(fh)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--lc-dir", default="data/lightcurves")
    parser.add_argument("--silver-dir", default="data/silver")
    parser.add_argument("--gold-dir", default="data/gold")
    parser.add_argument("--truth", default="data/truth/object_truth.parquet")
    parser.add_argument("--objects-csv", default="data/labels.csv", help="Optional object filter")
    parser.add_argument(
        "--association-csv",
        default=None,
        help="Optional LSST→ZTF association CSV for lightcurve sourcing",
    )
    parser.add_argument("--max-n-det", type=int, default=20)
    parser.add_argument("--exclude-unsafe-latest-snapshot", action="store_true",
                        help="Exclude broker snapshots marked latest_object_unsafe (e.g. ALeRCE API)")
    parser.add_argument("--allow-unsafe-latest-snapshot", action="store_true",
                        help="Include latest_object_unsafe events for scoring/debugging")
    parser.add_argument("--output", default=None, help="Optional explicit parquet output path")
    parser.add_argument("--shard-id", type=int, default=None,
                        help="0-indexed shard. Selects objects where idx %% n-shards == shard-id "
                             "after sorting the objects-csv list. Use with --n-shards.")
    parser.add_argument("--n-shards", type=int, default=None,
                        help="Total shard count. Required with --shard-id.")
    args = parser.parse_args()

    include_unsafe = (
        args.allow_unsafe_latest_snapshot and not args.exclude_unsafe_latest_snapshot
    )

    object_ids = _load_object_ids(Path(args.objects_csv)) if args.objects_csv else None

    if (args.shard_id is None) ^ (args.n_shards is None):
        parser.error("--shard-id and --n-shards must be used together")
    if args.shard_id is not None:
        if not (0 <= args.shard_id < args.n_shards):
            parser.error(f"--shard-id must be in [0, {args.n_shards}), got {args.shard_id}")
        if object_ids is None:
            parser.error("--shard-id requires --objects-csv (need a fixed object list)")
        sorted_ids = sorted(object_ids)
        shard_ids = [o for i, o in enumerate(sorted_ids) if i % args.n_shards == args.shard_id]
        print(f"Shard {args.shard_id}/{args.n_shards}: {len(shard_ids):,} objects "
              f"(of {len(sorted_ids):,} total)")
        object_ids = shard_ids

    path = build_object_epoch_snapshots(
        lc_dir=Path(args.lc_dir),
        silver_dir=Path(args.silver_dir),
        gold_dir=Path(args.gold_dir),
        truth_path=Path(args.truth),
        max_n_det=args.max_n_det,
        object_ids=object_ids,
        association_path=Path(args.association_csv) if args.association_csv else None,
        allow_unsafe_latest_snapshot=include_unsafe,
        output_path=Path(args.output) if args.output else None,
    )
    print(f"Wrote object-epoch snapshots → {path}")


if __name__ == "__main__":
    main()
