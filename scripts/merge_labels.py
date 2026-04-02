"""Merge LSST candidates into labels.csv with deduplication.

Safe to run multiple times — deduplicates by object_id, never loses
existing labels, and reports what changed.

Usage:
    python3 scripts/merge_labels.py --new data/lsst_candidates.csv
    python3 scripts/merge_labels.py --new data/lsst_candidates.csv --labels data/labels.csv
    python3 scripts/merge_labels.py --new data/lsst_candidates.csv --dry-run
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LSST candidates into labels.csv")
    parser.add_argument("--labels", default="data/labels.csv", help="Existing labels CSV")
    parser.add_argument("--new", required=True, help="New candidates CSV (from discover_lsst_training.py)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    new_path = Path(args.new)

    if not new_path.exists():
        print(f"ERROR: {new_path} not found")
        sys.exit(1)

    # Read existing labels
    existing: dict[str, dict] = {}
    existing_fieldnames: list[str] = ["object_id", "label"]
    if labels_path.exists():
        with open(labels_path) as fh:
            reader = csv.DictReader(fh)
            existing_fieldnames = list(reader.fieldnames or ["object_id", "label"])
            for row in reader:
                existing[row["object_id"]] = dict(row)

    # Read new candidates
    with open(new_path) as fh:
        new_rows = list(csv.DictReader(fh))

    # Merge: add new objects, don't overwrite existing labels
    n_new = 0
    n_skipped = 0
    for row in new_rows:
        oid = row["object_id"]
        if oid in existing:
            n_skipped += 1
            continue
        # Map LSST candidate to labels.csv format
        merged_row = {"object_id": oid}
        # No label yet — will come from TNS crossmatch or broker consensus
        merged_row["label"] = ""
        # Carry over ra/dec if labels.csv has those columns
        for col in ("ra", "dec", "survey", "n_det", "discovery_broker"):
            if col in existing_fieldnames or col in ("ra", "dec"):
                merged_row[col] = row.get(col, "")
        existing[oid] = merged_row
        n_new += 1

    # Ensure fieldnames include any new columns we added
    for col in ("ra", "dec", "survey", "n_det", "discovery_broker"):
        if col not in existing_fieldnames:
            existing_fieldnames.append(col)

    print(f"Labels:     {labels_path} ({len(existing) - n_new} existing)")
    print(f"New source: {new_path} ({len(new_rows)} candidates)")
    print(f"Result:     {n_new} new objects added, {n_skipped} already existed")
    print(f"Total:      {len(existing)} objects")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    # Backup existing labels
    if labels_path.exists():
        backup = labels_path.with_suffix(".csv.bak")
        shutil.copy2(labels_path, backup)
        print(f"\nBackup:     {backup}")

    # Write merged labels
    merged_rows = list(existing.values())
    with open(labels_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=existing_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged_rows)

    print(f"Wrote:      {labels_path}")

    # Count by survey
    ztf_count = sum(1 for r in merged_rows if not r.get("object_id", "").isdigit())
    lsst_count = sum(1 for r in merged_rows if r.get("object_id", "").isdigit())
    print(f"\n  ZTF:  {ztf_count}")
    print(f"  LSST: {lsst_count}")


if __name__ == "__main__":
    main()
