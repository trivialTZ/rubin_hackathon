"""scripts/normalize.py — Bronze → Silver → Gold transforms.

Usage:
    python scripts/normalize.py
    python scripts/normalize.py --skip-gold
    python scripts/normalize.py --labels labels.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from debass.ingest.silver import bronze_to_silver
from debass.ingest.gold import silver_to_gold


def _load_labels(path: Path) -> dict[str, str]:
    """Load a CSV with columns object_id,label into a dict."""
    import csv
    labels = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            labels[row["object_id"]] = row["label"]
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalise bronze data to silver and gold")
    parser.add_argument("--bronze-dir", default="data/bronze")
    parser.add_argument("--silver-dir", default="data/silver")
    parser.add_argument("--gold-dir", default="data/gold")
    parser.add_argument("--skip-gold", action="store_true")
    parser.add_argument("--labels", default=None, help="CSV file with object_id,label columns")
    args = parser.parse_args()

    print("Bronze → Silver...")
    silver_path = bronze_to_silver(
        bronze_dir=Path(args.bronze_dir),
        silver_dir=Path(args.silver_dir),
    )
    print(f"  wrote {silver_path}")

    if not args.skip_gold:
        label_map = _load_labels(Path(args.labels)) if args.labels else {}
        print("Silver → Gold...")
        gold_path = silver_to_gold(
            silver_dir=Path(args.silver_dir),
            gold_dir=Path(args.gold_dir),
            label_map=label_map,
        )
        print(f"  wrote {gold_path}")


if __name__ == "__main__":
    main()
