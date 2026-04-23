#!/usr/bin/env python3
"""Shardable SALT3 χ² runner for SGE array jobs.

Slices `data/labels.csv` by (shard-id, n-shards) stride, writes a tmp labels file,
invokes `scripts/collect_epoch_history.py` with `--experts salt3_chi2`, and
routes output to `data/silver/local_expert_outputs/salt3_chi2_shards/shard_<id>/`.
Run `scripts/merge_salt3_shards.py` after all shards complete.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Run SALT3 χ² on a shard of labels.csv")
    p.add_argument("--from-labels", default="data/labels.csv")
    p.add_argument("--lc-dir", default="data/lightcurves")
    p.add_argument("--shard-id", type=int, required=True, help="0-indexed shard id")
    p.add_argument("--n-shards", type=int, required=True)
    p.add_argument("--max-n-det", type=int, default=20)
    p.add_argument("--output-root", default="data/silver/local_expert_outputs/salt3_chi2_shards")
    args = p.parse_args()

    src = Path(args.from_labels)
    with open(src) as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or ["object_id"]
        all_rows = list(reader)

    shard_rows = [r for i, r in enumerate(all_rows) if i % args.n_shards == args.shard_id]
    print(f"shard {args.shard_id}/{args.n_shards}: {len(shard_rows)}/{len(all_rows)} objects")

    shard_dir = Path(args.output_root) / f"shard_{args.shard_id:03d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    tmp_labels = shard_dir / "labels.csv"
    with open(tmp_labels, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(shard_rows)

    cmd = [
        sys.executable, "-u", "scripts/collect_epoch_history.py",
        "--from-labels", str(tmp_labels),
        "--lc-dir", args.lc_dir,
        "--experts", "salt3_chi2",
        "--max-n-det", str(args.max_n_det),
        "--output-dir", str(shard_dir),
    ]
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=Path.cwd())
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
