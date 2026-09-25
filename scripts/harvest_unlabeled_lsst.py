#!/usr/bin/env python3
"""Harvest UNLABELED LSST alert lightcurves for self-supervised pretraining.

LSST publishes millions of alert lightcurves with no spectroscopic label — a
huge unsupervised corpus the sequence encoder (scripts/pretrain_seq_ssl.py) can
learn LSST cadence/depth from before it ever sees a label.  This script:

  1. Pages the Lasair LSST ``objects`` table (ORDER BY diaObjectId so offset
     pagination is deterministic; AS aliases because /query/ flattens qualified
     names; one JSON page per file on disk so the harvest is resumable).
  2. Keeps objects with >= ``--min-ndet`` detections (``nDiaSources`` — the
     objects-table detection count; the actual column is probed at runtime and
     falls back through a candidate list).
  3. EXCLUDES every id in the frozen benchmark manifest
     (``data/gold/lsst_live_locked_test.json`` — test_ids + excluded_ids): the
     locked test set must never be pretrained on, or headline metrics measure
     memorization.
  4. Fetches + caches each lightcurve via the scripts/fetch_lightcurves.py
     mechanics (ALeRCE LSST + Fink LSST batch endpoints, chunks of 50,
     skip-existing) so a re-run resumes where it left off.

``--shard-id/--n-shards`` partition the id space (sha1(diaObjectId) % n_shards)
so N array tasks harvest disjoint slices in parallel; the paged Lasair cache is
shared across shards (same query → same page files).

The pure helpers (``select_ids``, ``load_benchmark_exclusions``,
``pick_detcount_column``, ``shard_of``) take no network and are unit-tested;
``harvest_object_ids`` accepts an injected ``adapter`` so pagination/caching is
testable with a fake HTTP layer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass

# Candidate detection-count columns on the LSST ``objects`` table, richest/most
# canonical first.  ``nDiaSources`` is the one harvest_ztf_lsst_associations.py
# already selects successfully; the rest are defensive fallbacks.
DETCOUNT_CANDIDATES = ("nDiaSources", "nDetections", "nSources", "ndethist")
DEFAULT_DETCOUNT_COLUMN = "nDiaSources"
DEFAULT_BENCHMARK_MANIFEST = "data/gold/lsst_live_locked_test.json"


# ------------------------------------------------------------------ #
# Pure helpers (unit-tested, no network)                              #
# ------------------------------------------------------------------ #


def load_benchmark_exclusions(path: Path | str) -> set[str]:
    """Frozen-benchmark ids that must NEVER enter the pretrain corpus.

    Reads ``test_ids`` plus the keys of ``excluded_ids`` (locked-test
    association counterparts) from the manifest.  A missing file returns an
    empty set (with a warning at the call site) rather than raising, so an
    id-only dry run on a machine without the manifest still works — but the
    default path is committed, so in practice the set is always populated.
    """
    path = Path(path)
    if not path.exists():
        return set()
    payload = json.loads(path.read_text())
    excl: set[str] = {str(o) for o in (payload.get("test_ids") or [])}
    excl |= {str(o) for o in (payload.get("excluded_ids") or {})}
    return excl


def pick_detcount_column(columns: Iterable[str]) -> str | None:
    """First :data:`DETCOUNT_CANDIDATES` present in ``columns`` (or None)."""
    cols = {str(c) for c in columns}
    for cand in DETCOUNT_CANDIDATES:
        if cand in cols:
            return cand
    return None


def record_object_id(record: dict[str, Any]) -> str | None:
    for key in ("diaObjectId", "objects.diaObjectId", "objectId", "id"):
        val = record.get(key)
        if val is not None:
            text = str(val).strip()
            if text:
                return text
    return None


def record_ndet(record: dict[str, Any], detcol: str) -> int | None:
    """Detection count for one record: aliased ``n_det`` first, then the raw
    detection-count column, then any candidate present."""
    for key in ("n_det", detcol, f"objects.{detcol}", *DETCOUNT_CANDIDATES):
        val = record.get(key)
        if val is None:
            continue
        try:
            return int(float(val))
        except (TypeError, ValueError):
            continue
    return None


def shard_of(object_id: str, n_shards: int) -> int:
    """Deterministic shard index for an id (sha1 so it is uniform across the
    numeric-id space and stable across runs/machines)."""
    if n_shards <= 1:
        return 0
    h = hashlib.sha1(str(object_id).encode()).hexdigest()
    return int(h, 16) % n_shards


def select_ids(
    records: Iterable[dict[str, Any]],
    *,
    detcol: str,
    min_ndet: int,
    exclude: set[str],
    shard_id: int = 0,
    n_shards: int = 1,
    seen: set[str] | None = None,
) -> list[str]:
    """Filter raw ``objects`` records → the diaObjectIds to harvest.

    Drops: unparseable ids, benchmark ids in ``exclude``, objects below
    ``min_ndet``, ids not belonging to this shard, and (via ``seen``, mutated
    in place) duplicates carried across pages.  Order-preserving.
    """
    if seen is None:
        seen = set()
    out: list[str] = []
    for record in records:
        oid = record_object_id(record)
        if oid is None or oid in seen or oid in exclude:
            if oid is not None:
                seen.add(oid)
            continue
        n = record_ndet(record, detcol)
        if n is not None and n < min_ndet:
            seen.add(oid)
            continue
        if shard_of(oid, n_shards) != shard_id:
            seen.add(oid)
            continue
        seen.add(oid)
        out.append(oid)
    return out


def _selected_clause(detcol: str) -> str:
    # Single-table query, but alias anyway: /query/ flattens qualified names.
    return f"objects.diaObjectId AS diaObjectId, objects.{detcol} AS n_det"


# ------------------------------------------------------------------ #
# Live harvest (network; cached + resumable)                          #
# ------------------------------------------------------------------ #


def probe_detcount_column(adapter: Any) -> str:
    """Confirm the objects-table detection-count column via a limit=1 query.

    Tries each candidate; returns the first whose limit=1 query succeeds and
    returns the aliased ``n_det`` key.  Falls back to
    :data:`DEFAULT_DETCOUNT_COLUMN` if every probe errors (offline / schema
    drift) — the client-side ``record_ndet`` guard tolerates a mismatch.
    """
    for cand in DETCOUNT_CANDIDATES:
        try:
            rows = adapter._query_api(
                selected=_selected_clause(cand),
                tables="objects",
                conditions="1=1 ORDER BY objects.diaObjectId",
                limit=1,
                offset=0,
                survey="lsst",
            )
        except Exception:
            continue
        if rows and ("n_det" in rows[0] or "diaObjectId" in rows[0]):
            return cand
    return DEFAULT_DETCOUNT_COLUMN


def harvest_object_ids(
    *,
    target: int,
    cache_dir: Path,
    exclude: set[str],
    min_ndet: int = 3,
    shard_id: int = 0,
    n_shards: int = 1,
    page_size: int = 1000,
    max_pages: int = 20000,
    detcol: str | None = None,
    adapter: Any = None,
) -> list[str]:
    """Paged Lasair ``objects`` harvest → this shard's kept diaObjectIds.

    Resumable: each page is cached as one JSON file keyed by a hash of the
    exact query, so a changed query never reuses a stale page.  Stops when the
    shard's slice of ``target`` is filled or a short/empty page ends the table.
    ``adapter`` is injected in tests; production builds a ``LasairAdapter``.
    """
    if adapter is None:
        from debass_meta.access.lasair import LasairAdapter

        adapter = LasairAdapter()
    if detcol is None:
        detcol = probe_detcount_column(adapter)

    cache_dir.mkdir(parents=True, exist_ok=True)
    # Server-side pre-filter reduces pages, but record_ndet re-checks so a
    # missing/renamed column can never silently pass shallow objects through.
    conditions = f"{detcol} >= {int(min_ndet)} ORDER BY objects.diaObjectId"
    selected = _selected_clause(detcol)
    per_shard_target = -(-target // max(n_shards, 1))  # ceil division

    sel_tag = hashlib.sha1(f"{selected}|{conditions}|{page_size}".encode()).hexdigest()[:8]
    seen: set[str] = set()
    kept: list[str] = []
    for page in range(max_pages):
        if len(kept) >= per_shard_target:
            break
        page_path = cache_dir / f"objects_{sel_tag}_{page:05d}.json"
        if page_path.exists():
            rows = json.loads(page_path.read_text())
        else:
            rows = adapter._query_api(
                selected=selected,
                tables="objects",
                conditions=conditions,
                limit=page_size,
                offset=page * page_size,
                survey="lsst",
            )
            page_path.write_text(json.dumps(rows))
        picked = select_ids(
            rows, detcol=detcol, min_ndet=min_ndet, exclude=exclude,
            shard_id=shard_id, n_shards=n_shards, seen=seen,
        )
        remaining = per_shard_target - len(kept)
        kept.extend(picked[:remaining])
        print(f"  page {page}: {len(rows)} rows, +{min(len(picked), remaining)} kept "
              f"(total {len(kept)}/{per_shard_target})", flush=True)
        if len(rows) < page_size:
            break
    return kept


def fetch_lightcurves(ids: list[str], lc_dir: Path, *, chunk: int = 50) -> tuple[int, int]:
    """Fetch + cache each id's LSST lightcurve in chunks of ``chunk`` (skip
    -existing), reusing the scripts/fetch_lightcurves.py ALeRCE+Fink mechanics.
    Returns (n_ok, n_skipped)."""
    from scripts.fetch_lightcurves import (
        AlerceHistoryFetcher,
        _batch_fetch_alerce_lsst,
        _batch_fetch_fink_lsst_fp,
        fetch_lightcurves_for_objects,
    )

    lc_dir.mkdir(parents=True, exist_ok=True)
    fetcher = AlerceHistoryFetcher(lc_dir=lc_dir)
    total_ok = total_skip = 0
    for i in range(0, len(ids), chunk):
        batch = ids[i : i + chunk]
        todo = [o for o in batch if not (lc_dir / f"{o}.json").exists()]
        alerce_map = _batch_fetch_alerce_lsst(todo) if todo else {}
        fink_map = _batch_fetch_fink_lsst_fp(todo) if todo else {}
        ok, skip = fetch_lightcurves_for_objects(
            object_ids=batch, fetcher=fetcher, lc_dir=lc_dir,
            associations={}, fixtures_dir=None,
            alerce_lsst_map=alerce_map, fink_lsst_map=fink_map,
        )
        total_ok += ok
        total_skip += skip
        print(f"  fetched chunk {i // chunk + 1}: {ok} ok, {skip} skipped "
              f"(cumulative {total_ok} ok / {total_skip} skipped)", flush=True)
    return total_ok, total_skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=int, default=20000,
                    help="Total unlabeled objects to harvest across all shards")
    ap.add_argument("--min-ndet", type=int, default=3,
                    help="Keep objects with at least this many detections (nDiaSources)")
    ap.add_argument("--out-dir", default="data/pretrain_lsst",
                    help="Writes ids.csv + lightcurves/*.json here")
    ap.add_argument("--benchmark-manifest", default=DEFAULT_BENCHMARK_MANIFEST,
                    help="Frozen locked-test manifest; its test_ids + excluded_ids "
                         "are EXCLUDED from the corpus (never pretrain on the benchmark)")
    ap.add_argument("--cache-dir", default=None,
                    help="Lasair page cache dir (default <out-dir>/lasair_cache)")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--page-size", type=int, default=1000)
    ap.add_argument("--max-pages", type=int, default=20000)
    ap.add_argument("--no-fetch", action="store_true",
                    help="Only harvest + write ids.csv (skip lightcurve download)")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "lasair_cache"

    exclude = load_benchmark_exclusions(args.benchmark_manifest)
    if not exclude:
        print(f"WARNING: no benchmark exclusions loaded from {args.benchmark_manifest} "
              f"— proceeding, but verify the locked-test manifest path!", flush=True)
    else:
        print(f"Excluding {len(exclude):,} frozen-benchmark ids from the corpus", flush=True)

    print(f"Harvesting shard {args.shard_id}/{args.n_shards} "
          f"(target {args.target:,}, min_ndet {args.min_ndet})", flush=True)
    ids = harvest_object_ids(
        target=args.target, cache_dir=cache_dir, exclude=exclude,
        min_ndet=args.min_ndet, shard_id=args.shard_id, n_shards=args.n_shards,
        page_size=args.page_size, max_pages=args.max_pages,
    )
    print(f"Harvested {len(ids):,} unlabeled LSST ids", flush=True)

    ids_csv = out_dir / (f"ids_shard{args.shard_id}.csv" if args.n_shards > 1 else "ids.csv")
    with open(ids_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["object_id"])
        writer.writerows([[o] for o in ids])
    print(f"Wrote {len(ids):,} ids -> {ids_csv}", flush=True)

    if args.no_fetch:
        print(f"--no-fetch: skipping lightcurve download ({time.time() - t0:.0f}s)", flush=True)
        return

    lc_dir = out_dir / "lightcurves"
    ok, skip = fetch_lightcurves(ids, lc_dir)
    n_cached = len(list(lc_dir.glob("*.json")))
    print(f"Done: {ok} fetched-or-cached, {skip} skipped; {n_cached:,} lightcurves "
          f"in {lc_dir} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
