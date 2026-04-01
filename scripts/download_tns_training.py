"""Build a large training set by starting from TNS spectroscopic classifications.

TNS has ~160K transients. Many have ZTF internal names, meaning we can fetch
their lightcurves from ALeRCE. This gives us thousands of objects with
spectroscopic ground truth — far better than ALeRCE's weak self-labels.

Workflow:
  1. Load the TNS bulk CSV (download first with download_tns_bulk.py)
  2. Filter for objects with ZTF IDs + spectroscopic type
  3. Map TNS types to ternary labels (snia / nonIa_snlike / other)
  4. Fetch lightcurves from ALeRCE for each object
  5. Write data/labels.csv and data/lightcurves/<oid>.json

Usage:
  # First download the TNS bulk CSV (requires TNS credentials in .env):
  python3 scripts/download_tns_bulk.py

  # Then build the training set:
  python3 scripts/download_tns_training.py --limit 5000
  python3 scripts/download_tns_training.py --limit 10000 --resume
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from debass_meta.access.tns import map_tns_type_to_ternary


def _extract_ztf_ids(internal_names: str) -> list[str]:
    """Extract ZTF object IDs from TNS internal_names field."""
    if not isinstance(internal_names, str):
        return []
    return re.findall(r"ZTF\w+", internal_names)


def _fetch_lightcurve(client, oid: str, lc_dir: Path) -> bool:
    """Fetch and cache a lightcurve from ALeRCE. Returns True on success."""
    cache = lc_dir / f"{oid}.json"
    if cache.exists():
        return True
    try:
        df = client.query_lightcurve(oid=oid, format="pandas")
        if "detections" in df.columns:
            records = df["detections"].iloc[0]
            if not isinstance(records, list):
                return False
        else:
            records = df.to_dict(orient="records")
        if not records:
            return False
        clean = []
        for r in records:
            clean.append({k: (float(v) if hasattr(v, "item") else v)
                          for k, v in r.items()})
        with open(cache, "w") as fh:
            json.dump(clean, fh)
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build training set from TNS spectroscopic classifications"
    )
    parser.add_argument("--tns-csv", default="data/tns_public_objects.csv",
                        help="Path to TNS bulk CSV (download with download_tns_bulk.py)")
    parser.add_argument("--limit", type=int, default=5000,
                        help="Max objects to include in training set")
    parser.add_argument("--lc-dir", default="data/lightcurves")
    parser.add_argument("--labels-out", default="data/labels.csv")
    parser.add_argument("--resume", action="store_true",
                        help="Skip lightcurves already cached")
    parser.add_argument("--rate-limit", type=float, default=0.1,
                        help="Seconds between ALeRCE API calls")
    parser.add_argument("--min-per-class", type=int, default=100,
                        help="Minimum objects per ternary class")
    args = parser.parse_args()

    tns_csv = Path(args.tns_csv)
    if not tns_csv.exists():
        print(f"TNS bulk CSV not found at {tns_csv}")
        print("Download it first:")
        print("  python3 scripts/download_tns_bulk.py")
        sys.exit(1)

    lc_dir = Path(args.lc_dir)
    lc_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load and filter TNS catalog ──
    print(f"Loading TNS bulk CSV from {tns_csv} ...")
    # TNS bulk CSV has two header rows: first is a timestamp, second is columns.
    # Columns are quoted: "objid","name_prefix","name","ra",...,"internal_names",...
    # We skip the timestamp row with skiprows=0 if it looks like a data row,
    # or detect it automatically.
    tns_df = pd.read_csv(tns_csv, low_memory=False, quotechar='"')
    # If the first row was a timestamp/comment, the column names will be wrong.
    # Detect by checking for expected columns.
    expected_cols = {"internal_names", "type", "name_prefix"}
    if not expected_cols.issubset(set(tns_df.columns)):
        # First row was likely the timestamp header — re-read skipping it
        tns_df = pd.read_csv(tns_csv, low_memory=False, quotechar='"', skiprows=1)
    if not expected_cols.issubset(set(tns_df.columns)):
        print(f"  ERROR: TNS CSV columns don't match expected format.")
        print(f"  Found columns: {list(tns_df.columns)[:10]}...")
        print(f"  Expected at least: {expected_cols}")
        sys.exit(1)
    print(f"  Total TNS objects: {len(tns_df)}")
    print(f"  Columns: {list(tns_df.columns)}")

    # Keep only objects with ZTF internal names and a spectroscopic type
    tns_df["internal_names"] = tns_df["internal_names"].fillna("")
    tns_df["type"] = tns_df["type"].fillna("").astype(str).str.strip()
    tns_df["name_prefix"] = tns_df["name_prefix"].fillna("").astype(str).str.strip()

    # Extract ZTF IDs
    tns_df["ztf_ids"] = tns_df["internal_names"].apply(_extract_ztf_ids)
    has_ztf = tns_df["ztf_ids"].apply(len) > 0
    has_type = tns_df["type"] != ""
    has_spectra = (tns_df["name_prefix"] == "SN") & has_type

    candidates = tns_df[has_ztf & has_type].copy()
    spectroscopic = tns_df[has_ztf & has_spectra].copy()

    print(f"  With ZTF ID + type: {len(candidates)}")
    print(f"  With ZTF ID + spectroscopic confirmation: {len(spectroscopic)}")

    # Map to ternary labels
    candidates["ternary"] = candidates["type"].apply(map_tns_type_to_ternary)
    candidates = candidates[candidates["ternary"].notna()].copy()
    print(f"  Mappable to ternary labels: {len(candidates)}")

    # ── Step 2: Balance classes and select objects ──
    class_counts = candidates["ternary"].value_counts()
    print(f"\n  Available by class:")
    for cls, count in class_counts.items():
        print(f"    {cls}: {count}")

    # Allocate per-class quotas (balanced sampling)
    n_classes = len(class_counts)
    per_class = max(args.limit // n_classes, args.min_per_class)

    selected = []
    for cls in ["snia", "nonIa_snlike", "other"]:
        pool = candidates[candidates["ternary"] == cls]
        # Prefer spectroscopically confirmed
        spec_pool = pool[pool["name_prefix"] == "SN"]
        rest_pool = pool[pool["name_prefix"] != "SN"]
        ordered = pd.concat([spec_pool, rest_pool])

        n_take = min(per_class, len(ordered))
        selected.append(ordered.head(n_take))
        print(f"  Selected {n_take} {cls} objects")

    selected_df = pd.concat(selected, ignore_index=True)
    print(f"\nTotal selected: {len(selected_df)} objects")

    # Explode to (ztf_id, ternary) pairs — pick first ZTF ID per TNS object
    rows = []
    seen_ztf = set()
    for _, row in selected_df.iterrows():
        for ztf_id in row["ztf_ids"]:
            if ztf_id not in seen_ztf and len(rows) < args.limit:
                seen_ztf.add(ztf_id)
                rows.append({
                    "ztf_id": ztf_id,
                    "ternary": row["ternary"],
                    "tns_type": row["type"],
                    "tns_name": row.get("name", row.get("objname", "")),
                })
                break  # one ZTF ID per TNS object

    print(f"Unique ZTF objects: {len(rows)}")
    dist = Counter(r["ternary"] for r in rows)
    print(f"Class distribution: {dict(dist)}")

    # ── Step 3: Fetch lightcurves from ALeRCE ──
    print(f"\n=== Fetching {len(rows)} lightcurves from ALeRCE ===")
    try:
        from alerce.core import Alerce
        client = Alerce()
    except ImportError:
        print("alerce package not installed. Run: pip install alerce")
        sys.exit(1)

    ok, failed = 0, 0
    for i, row in enumerate(rows):
        oid = row["ztf_id"]
        cache = lc_dir / f"{oid}.json"
        if args.resume and cache.exists():
            ok += 1
        else:
            if _fetch_lightcurve(client, oid, lc_dir):
                ok += 1
            else:
                failed += 1
            time.sleep(args.rate_limit)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(rows)}] {ok} ok, {failed} failed")

    print(f"\nLightcurves: {ok} ok, {failed} failed")

    # ── Step 4: Write labels.csv ──
    labels_out = Path(args.labels_out)
    labels_out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(labels_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["object_id", "label"])
        writer.writeheader()
        for row in rows:
            if (lc_dir / f"{row['ztf_id']}.json").exists():
                writer.writerow({"object_id": row["ztf_id"], "label": row["ternary"]})
                written += 1

    final_dist = Counter(
        row["ternary"] for row in rows
        if (lc_dir / f"{row['ztf_id']}.json").exists()
    )
    print(f"\nWrote {written} labels → {labels_out}")
    print(f"Final class distribution: {dict(final_dist)}")
    print(f"Label quality: spectroscopic (TNS ground truth)")
    print("\nNext steps:")
    print("  python3 scripts/backfill.py --broker all --from-labels data/labels.csv")
    print("  python3 scripts/normalize.py")
    print("  python3 scripts/build_truth_table.py --labels data/labels.csv")


if __name__ == "__main__":
    main()
