#!/usr/bin/env python3
"""Bulk-pull DP1 DiaSource lightcurves for a list of diaObjectIds.

Reads diaObjectIds from data/truth/dp1_truth.parquet (built by
scripts/build_dp1_truth.py), pulls DiaSource rows via batched TAP IN(...)
queries, applies a reliability filter, and caches one parquet per object
under data/lightcurves/dp1/.

Per v6 plan §5 Stage 2:
  - Batch size 500 is verified (6.4K rows returned in one query).
  - Reliability >= 0.1 drops 85% of DiaSource rows (commissioning noise).
  - File-per-object caching keeps ingest re-entrant.

Usage
-----
    # 15 published SN smoke test
    python scripts/fetch_dp1_lightcurves.py \\
        --truth data/truth/dp1_truth.parquet --only-published

    # Full truth table (re-entrant; skips already-cached)
    python scripts/fetch_dp1_lightcurves.py --truth data/truth/dp1_truth.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from debass_meta.access.rubin_rsp import RSPClient

REPO = Path(__file__).resolve().parents[1]
_LC_DIR = REPO / "data/lightcurves/dp1"


def _object_path(diaObjectId: int, lc_dir: Path) -> Path:
    return lc_dir / f"{int(diaObjectId)}.parquet"


def _already_cached(ids: list[int], lc_dir: Path, overwrite: bool) -> list[int]:
    if overwrite:
        return ids
    pending = []
    for i in ids:
        if not _object_path(i, lc_dir).exists():
            pending.append(i)
    return pending


def fetch_many(
    truth_path: Path,
    lc_dir: Path,
    *,
    only_published: bool,
    only_ids: list[int] | None,
    min_reliability: float,
    batch_size: int,
    limit: int | None,
    overwrite: bool,
) -> None:
    """Per-batch shard + empty-sentinel → resumable at 50K+ scale.

    For each 500-ID TAP chunk:
      1. Query DiaSource rows.
      2. Group by diaObjectId and write a parquet per object.
      3. For every diaObjectId in the chunk that returned zero rows (all
         pruned by the reliability filter), write a 0-row sentinel parquet
         so reruns skip it.
    A mid-run kill loses at most one batch of progress.
    """
    truth = pd.read_parquet(truth_path)
    ids_all = truth["diaObjectId"].astype("int64").tolist()

    if only_ids:
        ids_all = [i for i in ids_all if i in set(only_ids)]
    if only_published:
        ids_all = truth.loc[truth["is_published_sn"] == True, "diaObjectId"].astype("int64").tolist()
    if limit is not None:
        ids_all = ids_all[:limit]

    print(f"Requested: {len(ids_all):,} diaObjectIds")
    todo = _already_cached(ids_all, lc_dir, overwrite)
    print(f"To fetch (after cache): {len(todo):,}")
    if not todo:
        print("Nothing to do — all cached. Use --overwrite to re-pull.")
        return

    lc_dir.mkdir(parents=True, exist_ok=True)
    client = RSPClient()
    n_batches = (len(todo) + batch_size - 1) // batch_size
    t0 = time.time()
    total_rows = 0
    wrote_nonempty = 0
    wrote_empty = 0

    for bi, start in enumerate(range(0, len(todo), batch_size)):
        chunk = todo[start:start + batch_size]
        try:
            df = client.fetch_diasources(
                chunk,
                batch_size=batch_size,
                min_reliability=min_reliability,
                progress_every=0,
            )
        except Exception as exc:
            print(f"  batch {bi+1}/{n_batches} FAILED: "
                  f"{type(exc).__name__}: {str(exc)[:160]}")
            # Continue — next batch may work; failed IDs stay uncached.
            continue

        df["diaObjectId"] = df["diaObjectId"].astype("int64") if len(df) else df.get("diaObjectId")
        seen_in_chunk = set()
        if len(df):
            for oid, grp in df.groupby("diaObjectId"):
                out = _object_path(int(oid), lc_dir)
                grp.sort_values("midpointMjdTai").to_parquet(out, index=False)
                wrote_nonempty += 1
                seen_in_chunk.add(int(oid))
                total_rows += len(grp)
        empty_schema = pd.DataFrame({
            "diaObjectId": pd.Series([], dtype="int64"),
            "diaSourceId": pd.Series([], dtype="int64"),
            "band": pd.Series([], dtype="object"),
            "midpointMjdTai": pd.Series([], dtype="float64"),
            "psfFlux": pd.Series([], dtype="float64"),
            "psfFluxErr": pd.Series([], dtype="float64"),
            "reliability": pd.Series([], dtype="float64"),
        })
        for oid in chunk:
            if int(oid) in seen_in_chunk:
                continue
            out = _object_path(int(oid), lc_dir)
            if out.exists():
                continue
            empty_schema.to_parquet(out, index=False)
            wrote_empty += 1

        elapsed = time.time() - t0
        print(f"  batch {bi+1:>3}/{n_batches}: "
              f"{total_rows:,} rows total, "
              f"{wrote_nonempty:,} objects with LCs, "
              f"{wrote_empty:,} empty-sentinels ({elapsed:.1f}s elapsed)",
              flush=True)

    dt = time.time() - t0
    print(f"\nDone: {total_rows:,} DiaSource rows, "
          f"{wrote_nonempty:,} non-empty objects, "
          f"{wrote_empty:,} empty sentinels, "
          f"{dt:.1f}s "
          f"(mean {total_rows/max(len(todo), 1):.1f} det/obj, "
          f"reliability>={min_reliability})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--truth", default=str(REPO / "data/truth/dp1_truth.parquet"))
    ap.add_argument("--lc-dir", default=str(_LC_DIR))
    ap.add_argument("--only-published", action="store_true",
                    help="Only pull lightcurves for the 15 published DP1 SNe.")
    ap.add_argument("--only-ids", default=None,
                    help="Comma-separated list of diaObjectIds to fetch.")
    ap.add_argument("--min-reliability", type=float, default=0.1,
                    help="Apply `reliability >= X` cut (default 0.1; plan §5 Stage 2).")
    ap.add_argument("--batch-size", type=int, default=500,
                    help="diaObjectId IN(...) batch size (default 500; verified).")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-pull even when a cached parquet exists.")
    args = ap.parse_args()

    only_ids = None
    if args.only_ids:
        only_ids = [int(s) for s in args.only_ids.split(",") if s.strip()]

    fetch_many(
        truth_path=Path(args.truth),
        lc_dir=Path(args.lc_dir),
        only_published=args.only_published,
        only_ids=only_ids,
        min_reliability=args.min_reliability,
        batch_size=args.batch_size,
        limit=args.limit,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
