"""scripts/build_epoch_table.py — Build (object_id, n_det) epoch feature table.

Usage:
    python scripts/build_epoch_table.py
    python scripts/build_epoch_table.py --max-n-det 10 --labels data/labels.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from debass_meta.ingest.epochs import build_epoch_table


def _load_labels(path: Path) -> dict[str, str]:
    labels = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            labels[row["object_id"]] = row["label"]
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Build epoch feature table")
    parser.add_argument("--lc-dir",     default="data/lightcurves")
    parser.add_argument("--silver-dir", default="data/silver")
    parser.add_argument("--epochs-dir", default="data/epochs")
    parser.add_argument("--labels",     default="data/labels.csv")
    parser.add_argument("--max-n-det",  type=int, default=20)
    args = parser.parse_args()

    label_map = {}
    if Path(args.labels).exists():
        label_map = _load_labels(Path(args.labels))
        print(f"Loaded {len(label_map)} labels from {args.labels}")

    print("Building epoch feature table...")
    out = build_epoch_table(
        lc_dir=Path(args.lc_dir),
        silver_dir=Path(args.silver_dir),
        epochs_dir=Path(args.epochs_dir),
        label_map=label_map,
        max_n_det=args.max_n_det,
    )
    print(f"Wrote {out}")

    # Quick summary
    try:
        import pandas as pd
        df = pd.read_parquet(out)
        print(f"  Rows: {len(df)}")
        print(f"  Objects: {df['object_id'].nunique()}")
        print(f"  n_det range: {df['n_det'].min()} – {df['n_det'].max()}")
        print(f"  Label counts: {df[df['target_label'].notna()]['target_label'].value_counts().to_dict()}")
        labelled_early = df[df['n_det'] <= 5]
        print(f"  Rows with n_det ≤ 5: {len(labelled_early)}")
    except Exception as e:
        print(f"  (summary error: {e})")


if __name__ == "__main__":
    main()
