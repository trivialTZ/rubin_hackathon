"""Build exact object-epoch snapshots."""
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
    parser = argparse.ArgumentParser(description="Build object_epoch_snapshots.parquet")
    parser.add_argument("--lc-dir", default="data/lightcurves")
    parser.add_argument("--silver-dir", default="data/silver")
    parser.add_argument("--gold-dir", default="data/gold")
    parser.add_argument("--truth", default="data/truth/object_truth.parquet")
    parser.add_argument("--objects-csv", default="data/labels.csv", help="Optional object filter")
    parser.add_argument("--max-n-det", type=int, default=20)
    parser.add_argument("--exclude-unsafe-latest-snapshot", action="store_true",
                        help="Exclude broker snapshots marked latest_object_unsafe (e.g. ALeRCE API)")
    parser.add_argument("--allow-unsafe-latest-snapshot", action="store_true",
                        help="Include latest_object_unsafe events for scoring/debugging")
    parser.add_argument("--output", default=None, help="Optional explicit parquet output path")
    args = parser.parse_args()

    include_unsafe = (
        args.allow_unsafe_latest_snapshot and not args.exclude_unsafe_latest_snapshot
    )

    path = build_object_epoch_snapshots(
        lc_dir=Path(args.lc_dir),
        silver_dir=Path(args.silver_dir),
        gold_dir=Path(args.gold_dir),
        truth_path=Path(args.truth),
        max_n_det=args.max_n_det,
        object_ids=_load_object_ids(Path(args.objects_csv)) if args.objects_csv else None,
        allow_unsafe_latest_snapshot=include_unsafe,
        output_path=Path(args.output) if args.output else None,
    )
    print(f"Wrote object-epoch snapshots → {path}")


if __name__ == "__main__":
    main()
