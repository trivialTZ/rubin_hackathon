"""Build exact object-epoch snapshots."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from debass.ingest.gold import build_object_epoch_snapshots


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
    parser.add_argument("--disallow-unsafe-latest-snapshot", action="store_true")
    args = parser.parse_args()

    path = build_object_epoch_snapshots(
        lc_dir=Path(args.lc_dir),
        silver_dir=Path(args.silver_dir),
        gold_dir=Path(args.gold_dir),
        truth_path=Path(args.truth),
        max_n_det=args.max_n_det,
        object_ids=_load_object_ids(Path(args.objects_csv)) if args.objects_csv else None,
        allow_unsafe_latest_snapshot=not args.disallow_unsafe_latest_snapshot,
    )
    print(f"Wrote object-epoch snapshots → {path}")


if __name__ == "__main__":
    main()
