#!/usr/bin/env python3
"""CLI driver for debass_meta.ingest.rsp_dp1.build_dp1_snapshots.

Reads data/lightcurves/dp1/*.parquet + data/truth/dp1_truth.parquet,
runs features + 4 local experts per epoch, emits
data/gold/dp1_snapshots.parquet in v5d schema.

Usage
-----
    # Smoke test: just the 15 published SNe
    python scripts/build_dp1_snapshots.py --only-published

    # Full pool, single job
    python scripts/build_dp1_snapshots.py

    # Sharded (one shard per SGE array task)
    python scripts/build_dp1_snapshots.py --shard-id 0 --n-shards 8 \\
        --out data/gold/dp1_snapshots_v6e.shard-0-of-8.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from debass_meta.ingest.rsp_dp1 import build_dp1_snapshots

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--lc-dir", default=str(REPO / "data/lightcurves/dp1"))
    ap.add_argument("--truth", default=str(REPO / "data/truth/dp1_truth.parquet"))
    ap.add_argument("--out", default=str(REPO / "data/gold/dp1_snapshots.parquet"))
    ap.add_argument("--max-n-det", type=int, default=20)
    ap.add_argument("--only-published", action="store_true",
                    help="Restrict to the 15 published DP1 SN diaObjectIds.")
    ap.add_argument("--only-ids", default=None,
                    help="Comma-separated diaObjectIds.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard-id", type=int, default=None,
                    help="0-indexed shard. Selects LCs where idx %% n-shards == shard-id "
                         "after sorting the LC directory. Use with --n-shards.")
    ap.add_argument("--n-shards", type=int, default=None,
                    help="Total shard count. Required with --shard-id.")
    args = ap.parse_args()

    only_ids = None
    if args.only_published:
        truth = pd.read_parquet(args.truth)
        pub = truth.loc[truth["is_published_sn"] == True, "diaObjectId"]
        only_ids = [int(x) for x in pub]
        print(f"Restricting to {len(only_ids)} published SN diaObjectIds")
    elif args.only_ids:
        only_ids = [int(s) for s in args.only_ids.split(",") if s.strip()]

    if (args.shard_id is None) ^ (args.n_shards is None):
        ap.error("--shard-id and --n-shards must be used together")
    if args.shard_id is not None:
        if not (0 <= args.shard_id < args.n_shards):
            ap.error(f"--shard-id must be in [0, {args.n_shards}), got {args.shard_id}")
        all_oids: list[int] = []
        for p in sorted(Path(args.lc_dir).glob("*.parquet")):
            try:
                all_oids.append(int(p.stem))
            except ValueError:
                continue
        shard_oids = [o for i, o in enumerate(all_oids) if i % args.n_shards == args.shard_id]
        if only_ids is not None:
            keep = set(only_ids)
            shard_oids = [o for o in shard_oids if o in keep]
        only_ids = shard_oids
        print(f"Shard {args.shard_id}/{args.n_shards}: {len(only_ids):,} LCs "
              f"(of {len(all_oids):,} total)")

    build_dp1_snapshots(
        lc_dir=Path(args.lc_dir),
        truth_path=Path(args.truth),
        out_path=Path(args.out),
        max_n_det=args.max_n_det,
        only_object_ids=only_ids,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
