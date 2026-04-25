#!/usr/bin/env python3
"""Merge sharded silver outputs from local_infer.py --shard-id runs.

Each shard's `local_infer.py` writes to its own silver directory (to
avoid races on the JSON cache). After all shards complete, this script
concatenates the per-shard partition parquets, dedups on
(expert, object_id, n_det), merges with any existing rows in the
target silver partition, and updates local_expert_outputs.json.

Usage
-----
    python scripts/_merge_silver_v6e2.py \\
        --shard-glob 'data/silver_v6e2_salt3_shard_*/local_expert_outputs/salt3_chi2/part-latest.parquet' \\
        --target-silver data/silver \\
        --expert salt3_chi2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--shard-glob", required=True,
                    help="Glob matching shard partition parquets")
    ap.add_argument("--target-silver", type=Path, default=Path("data/silver"),
                    help="Target silver dir (default data/silver)")
    ap.add_argument("--expert", required=True,
                    help="Expert name (e.g. salt3_chi2). Determines which target "
                         "partition to merge into.")
    args = ap.parse_args()

    shard_paths = sorted(Path().glob(args.shard_glob))
    if not shard_paths:
        sys.exit(f"no shards matched {args.shard_glob}")

    # Concatenate shard records
    print(f"merging {len(shard_paths)} shards for expert={args.expert}:")
    frames = []
    for p in shard_paths:
        df = pd.read_parquet(p)
        oids = df["object_id"].astype(str)
        z = int(oids.str.startswith("ZTF").sum())
        l = int(oids.str.match(r"^[0-9]{10,}$").sum())
        print(f"  {p}: {len(df):,} ({z:,} ZTF, {l:,} LSST)")
        frames.append(df)
    new_df = pd.concat(frames, ignore_index=True)
    print(f"  combined shards: {len(new_df):,} rows")

    # Merge with existing target partition
    target_partition = (
        args.target_silver / "local_expert_outputs" / args.expert / "part-latest.parquet"
    )
    if target_partition.exists():
        existing = pd.read_parquet(target_partition)
        existing_oids = existing["object_id"].astype(str)
        z = int(existing_oids.str.startswith("ZTF").sum())
        l = int(existing_oids.str.match(r"^[0-9]{10,}$").sum())
        print(f"  existing partition: {len(existing):,} ({z:,} ZTF, {l:,} LSST)")
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged = merged.drop_duplicates(
            subset=["expert", "object_id", "n_det"], keep="last"
        )
    else:
        merged = new_df.drop_duplicates(
            subset=["expert", "object_id", "n_det"], keep="last"
        )

    merged_oids = merged["object_id"].astype(str)
    z = int(merged_oids.str.startswith("ZTF").sum())
    l = int(merged_oids.str.match(r"^[0-9]{10,}$").sum())
    print(f"  merged total: {len(merged):,} ({z:,} ZTF, {l:,} LSST)")

    target_partition.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(target_partition, index=False)
    print(f"  → wrote {target_partition}")

    # Rebuild local_expert_outputs.json from ALL partition parquets
    print("\nrebuilding local_expert_outputs.json from all expert partitions:")
    all_records: list[dict] = []
    base = args.target_silver / "local_expert_outputs"
    if base.exists():
        for ek_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            part = ek_dir / "part-latest.parquet"
            if not part.exists():
                continue
            df = pd.read_parquet(part).copy()
            if "class_probabilities" in df.columns:
                df["class_probabilities"] = df["class_probabilities"].apply(
                    lambda v: json.loads(v) if isinstance(v, str) else v
                )
            recs = df.to_dict(orient="records")
            all_records.extend(recs)
            print(f"  {ek_dir.name}: {len(recs):,}")
    json_path = args.target_silver / "local_expert_outputs.json"
    with open(json_path, "w") as fh:
        json.dump(all_records, fh)
    print(f"  → wrote {json_path} ({len(all_records):,} total)")


if __name__ == "__main__":
    main()
