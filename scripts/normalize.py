"""scripts/normalize.py — Bronze → Silver transforms.

By default this command only writes event-level silver outputs.
Legacy baseline gold generation is opt-in via ``--emit-legacy-gold``.

Usage:
    python scripts/normalize.py
    python scripts/normalize.py --emit-legacy-gold --labels labels.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from debass_meta_meta.ingest.silver import bronze_to_silver
from debass_meta_meta.ingest.gold import silver_to_gold


def _load_labels(path: Path) -> dict[str, str]:
    """Load a CSV with columns object_id,label into a dict."""
    import csv
    labels = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            labels[row["object_id"]] = row["label"]
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalise bronze data to event-level silver")
    parser.add_argument("--bronze-dir", default="data/bronze")
    parser.add_argument("--silver-dir", default="data/silver")
    parser.add_argument("--gold-dir", default="data/gold")
    parser.add_argument("--emit-legacy-gold", action="store_true",
                        help="Also build legacy baseline snapshots.parquet")
    parser.add_argument("--labels", default=None, help="CSV file with object_id,label columns")
    args = parser.parse_args()

    print("Bronze → Silver...")
    silver_path = bronze_to_silver(
        bronze_dir=Path(args.bronze_dir),
        silver_dir=Path(args.silver_dir),
    )
    print(f"  wrote {silver_path}")

    if args.emit_legacy_gold:
        label_map = _load_labels(Path(args.labels)) if args.labels else {}
        print("Silver → Legacy baseline gold...")
        gold_path = silver_to_gold(
            silver_dir=Path(args.silver_dir),
            gold_dir=Path(args.gold_dir),
            label_map=label_map,
        )
        print(f"  wrote {gold_path}")


if __name__ == "__main__":
    main()
