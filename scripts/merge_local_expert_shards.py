#!/usr/bin/env python3
"""Merge shard outputs for a local expert into part-latest.parquet."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--expert", required=True, help="expert key (e.g. salt3_chi2, alerce_lc)")
    p.add_argument("--shards-root", default=None,
                   help="defaults to data/silver/local_expert_outputs/<expert>_shards")
    p.add_argument("--out", default=None,
                   help="defaults to data/silver/local_expert_outputs/<expert>/part-latest.parquet")
    args = p.parse_args()

    root = Path(args.shards_root or f"data/silver/local_expert_outputs/{args.expert}_shards")
    parts = sorted(root.glob(f"shard_*/{args.expert}/part-latest.parquet"))
    if not parts:
        raise SystemExit(f"no shards found under {root}")
    frames = [pd.read_parquet(p_path) for p_path in parts]
    for p_path, f in zip(parts, frames):
        print(f"  {p_path}: {len(f)} rows")
    merged = pd.concat(frames, ignore_index=True)
    out = Path(args.out or f"data/silver/local_expert_outputs/{args.expert}/part-latest.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    print(f"wrote {out}: {len(merged)} rows / {merged['object_id'].nunique()} objects from {len(parts)} shards")


if __name__ == "__main__":
    main()
