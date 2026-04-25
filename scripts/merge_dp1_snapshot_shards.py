#!/usr/bin/env python3
"""Merge shard outputs from build_dp1_snapshots.py or build_object_epoch_snapshots.py.

Concatenates N parquet shards into a single output, deduplicates on the
detected primary key (diaObjectId for DP1; object_id for training-set
gold), and prints per-expert avail counts.

You can override the dedup key with `--dedup-keys oid_col,n_det_col`
if neither auto-detected pair is present.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--shard-glob", required=True,
                    help="Glob like 'data/gold/dp1_snapshots_v6e.shard-*-of-8.parquet' "
                         "or 'data/gold/object_epoch_snapshots_v6e2.shard-*-of-8.parquet'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dedup-keys", default=None,
                    help="Comma-separated columns to dedup on. If omitted, auto-detects "
                         "(diaObjectId, n_det) or (object_id, n_det).")
    args = ap.parse_args()

    shard_paths = sorted(Path().glob(args.shard_glob))
    if not shard_paths:
        sys.exit(f"no shards matched {args.shard_glob}")
    print(f"merging {len(shard_paths)} shards:")
    frames = []
    for p in shard_paths:
        df = pd.read_parquet(p)
        print(f"  {p}: {len(df):,} rows")
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)

    if args.dedup_keys:
        dedup_keys = [k.strip() for k in args.dedup_keys.split(",") if k.strip()]
    else:
        dedup_keys = None
        for candidate in (["diaObjectId", "n_det"], ["object_id", "n_det"]):
            if set(candidate).issubset(merged.columns):
                dedup_keys = candidate
                break

    if dedup_keys:
        before = len(merged)
        merged = merged.drop_duplicates(subset=dedup_keys, keep="last")
        if before != len(merged):
            print(f"  dropped {before - len(merged):,} duplicate {tuple(dedup_keys)} rows")
    else:
        print("  no dedup key detected; skipping dedup")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    print(f"\nwrote {len(merged):,} rows × {len(merged.columns)} cols → {out_path}")

    print("\nper-expert avail counts:")
    for c in sorted(merged.columns):
        if not c.startswith("avail__"):
            continue
        n = int(merged[c].fillna(0).astype(float).gt(0.5).sum())
        print(f"  {c}: {n:,} / {len(merged):,} ({100*n/len(merged):.1f}%)")


if __name__ == "__main__":
    main()
