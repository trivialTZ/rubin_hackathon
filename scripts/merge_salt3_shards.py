#!/usr/bin/env python3
"""Merge all SALT3 χ² shard outputs into the canonical part-latest.parquet."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shards-root", default="data/silver/local_expert_outputs/salt3_chi2_shards")
    p.add_argument("--out", default="data/silver/local_expert_outputs/salt3_chi2/part-latest.parquet")
    args = p.parse_args()

    root = Path(args.shards_root)
    parts = sorted(root.glob("shard_*/salt3_chi2/part-latest.parquet"))
    if not parts:
        raise SystemExit(f"no shards found under {root}")

    frames = []
    for p_path in parts:
        frames.append(pd.read_parquet(p_path))
        print(f"  {p_path}: {len(frames[-1])} rows")

    merged = pd.concat(frames, ignore_index=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    print(f"wrote {out_path}: {len(merged)} rows from {len(parts)} shards "
          f"({merged['object_id'].nunique()} unique objects)")


if __name__ == "__main__":
    main()
