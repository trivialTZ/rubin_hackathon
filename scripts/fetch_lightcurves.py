"""scripts/fetch_lightcurves.py — Fetch and cache raw lightcurves for all training objects.

Lightcurves are stored in data/lightcurves/{oid}.json and are used by:
  - build_epoch_table.py  (to key broker scores by n_det)
  - local_infer_gpu.sh    (SCC: truncate to N detections, re-run SNN/ParSNIP)

Supports both ZTF lightcurves (via ALeRCE) and LSST lightcurves.
For LSST we union three sources (richest wins on conflict):
  1. ALeRCE LSST  /lightcurve_api/detections    (has reliability + snr)
  2. Fink LSST    /api/v1/fp (forced photometry)
  3. Lasair LSST  fixtures already saved by Phase-1 backfill

Usage:
    python scripts/fetch_lightcurves.py
    python scripts/fetch_lightcurves.py --objects ZTF21abywdxt ZTF22aalpfln
    python scripts/fetch_lightcurves.py --from-labels data/labels.csv
    python scripts/fetch_lightcurves.py --only-kind lsst_dia_object_id
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from debass_meta.access.alerce_history import AlerceHistoryFetcher
from debass_meta.access.associations import (
    load_lsst_ztf_associations,
    resolve_object_reference,
)
from debass_meta.access.identifiers import infer_identifier_kind
from debass_meta.features.detection import normalize_lightcurve

_ALERCE_LSST_DETECTIONS = "https://api-lsst.alerce.online/lightcurve_api/detections"
_FINK_LSST_FP = "https://api.lsst.fink-portal.org/api/v1/fp"

_RETRY_STATUS = {500, 502, 503, 504}
_RETRY_DELAYS = (2.0, 5.0, 12.0, 30.0)


def _request_with_retry(method: str, url: str, *, timeout: int = 60, **kwargs):
    """HTTP request with exponential backoff on transient failures.

    Retries on 500/502/503/504 and common timeout/connection errors.
    Returns the final Response (which may still be non-2xx) or raises
    after the last attempt.
    """
    last_exc = None
    for attempt, delay in enumerate([0.0, *_RETRY_DELAYS]):
        if delay:
            time.sleep(delay)
        try:
            resp = requests.request(method, url, timeout=timeout, **kwargs)
            if resp.status_code not in _RETRY_STATUS:
                return resp
            if attempt < len(_RETRY_DELAYS):
                print(
                    f"  [retry] {method} {url.split('?')[0]} -> HTTP "
                    f"{resp.status_code}, retrying in {_RETRY_DELAYS[attempt]:.0f}s "
                    f"(attempt {attempt + 1}/{len(_RETRY_DELAYS)})"
                )
                last_exc = None
                continue
            return resp
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            if attempt < len(_RETRY_DELAYS):
                print(
                    f"  [retry] {method} {url.split('?')[0]} -> {type(exc).__name__}, "
                    f"retrying in {_RETRY_DELAYS[attempt]:.0f}s"
                )
                continue
            raise
    if last_exc is not None:
        raise last_exc
    return resp  # type: ignore[name-defined]


def _load_oids_from_labels(path: Path) -> list[str]:
    oids = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            oids.append(row["object_id"])
    return oids


def _fetch_lsst_lightcurve_from_lasair(object_id: str) -> list[dict] | None:
    """Fetch LSST lightcurve from Lasair LSST API (diaSourcesList).

    Returns normalized detection list, or None if unavailable.
    """
    import requests

    endpoint = os.environ.get("LASAIR_ENDPOINT", "").rstrip("/")
    if not endpoint:
        return None

    token = os.environ.get("LASAIR_TOKEN", "")
    url = f"{endpoint}/api/object/{object_id}/"
    headers = {"Authorization": f"Token {token}"} if token else {}

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception:
        return None

    # Try multiple paths for diaSourcesList
    sources = data.get("diaSourcesList") or []
    if not sources and "lasairData" in data:
        sources = data["lasairData"].get("diaSourcesList") or []

    if not sources:
        return None

    # Normalize to common schema
    return normalize_lightcurve(sources, survey="LSST")


def _fetch_lsst_from_fixture(object_id: str, fixtures_dir: Path) -> list[dict] | None:
    """Load LSST lightcurve from local Lasair fixture file."""
    fixture_path = fixtures_dir / f"{object_id}_object.json"
    if not fixture_path.exists():
        return None
    try:
        with open(fixture_path) as fh:
            data = json.load(fh)
        sources = data.get("diaSourcesList") or []
        if not sources and "lasairData" in data:
            sources = data["lasairData"].get("diaSourcesList") or []
        if not sources:
            return None
        return normalize_lightcurve(sources, survey="LSST")
    except Exception:
        return None


def _batch_fetch_alerce_lsst(
    object_ids: list[str], *, batch_size: int = 50, timeout: int = 60
) -> dict[str, list[dict]]:
    """Fetch detections from ALeRCE LSST /lightcurve_api/detections.

    Returns {oid: [normalized_detections]}. Resolves numeric band -> letter
    via the per-row band_map before normalization. Empty dict if the API is
    unreachable; caller falls back to Fink/fixture.
    """
    results: dict[str, list[dict]] = {}
    n_batches = (len(object_ids) + batch_size - 1) // batch_size
    for bi, i in enumerate(range(0, len(object_ids), batch_size), start=1):
        chunk = object_ids[i : i + batch_size]
        qs = "survey_id=lsst&" + "&".join(f"oid={o}" for o in chunk)
        url = f"{_ALERCE_LSST_DETECTIONS}?{qs}"
        t0 = time.time()
        try:
            resp = _request_with_retry("GET", url, timeout=timeout)
            if resp.status_code != 200:
                print(f"  [alerce] batch {bi}/{n_batches}: HTTP {resp.status_code} (after retries)")
                continue
            rows = resp.json()
        except Exception as exc:
            print(f"  [alerce] batch {bi}/{n_batches}: {type(exc).__name__} {exc}")
            continue
        if not isinstance(rows, list):
            continue
        raw_per_oid: dict[str, list[dict]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            oid = str(row.get("oid") or row.get("diaObjectId") or "")
            if not oid:
                continue
            bmap = row.get("band_map") or {}
            b = row.get("band")
            if isinstance(b, (int, float)) and bmap:
                letter = bmap.get(str(int(b))) or bmap.get(int(b))
                if letter:
                    row = {**row, "band": letter}
            raw_per_oid.setdefault(oid, []).append(row)
        for oid, raw in raw_per_oid.items():
            dets = normalize_lightcurve(raw, survey="LSST")
            for d in dets:
                d["_source"] = "alerce_lsst"
            results.setdefault(oid, []).extend(dets)
        dt = time.time() - t0
        print(
            f"  [alerce] batch {bi}/{n_batches}: {len(chunk)} ids -> "
            f"{len(rows)} rows / {len(raw_per_oid)} objs ({dt:.1f}s)"
        )
    return results


def _batch_fetch_fink_lsst_fp(
    object_ids: list[str], *, batch_size: int = 50, timeout: int = 60
) -> dict[str, list[dict]]:
    """Fetch forced-photometry from Fink LSST /api/v1/fp.

    Returns {oid: [normalized_detections]}. Strips the ``r:`` prefix from
    response columns so _normalize_lsst finds psfFlux/midpointMjdTai/band.
    """
    results: dict[str, list[dict]] = {}
    n_batches = (len(object_ids) + batch_size - 1) // batch_size
    for bi, i in enumerate(range(0, len(object_ids), batch_size), start=1):
        chunk = object_ids[i : i + batch_size]
        payload = {"diaObjectId": ",".join(str(o) for o in chunk), "output-format": "json"}
        t0 = time.time()
        try:
            resp = _request_with_retry("POST", _FINK_LSST_FP, timeout=timeout, json=payload)
            if resp.status_code != 200:
                print(f"  [fink]   batch {bi}/{n_batches}: HTTP {resp.status_code} (after retries)")
                continue
            rows = resp.json()
        except Exception as exc:
            print(f"  [fink]   batch {bi}/{n_batches}: {type(exc).__name__} {exc}")
            continue
        if not isinstance(rows, list):
            continue
        raw_per_oid: dict[str, list[dict]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            clean = {(k[2:] if k.startswith("r:") else k): v for k, v in row.items()}
            oid = str(clean.get("diaObjectId") or "")
            if not oid:
                continue
            raw_per_oid.setdefault(oid, []).append(clean)
        for oid, raw in raw_per_oid.items():
            dets = normalize_lightcurve(raw, survey="LSST")
            for d in dets:
                d["_source"] = "fink_lsst_fp"
            results.setdefault(oid, []).extend(dets)
        dt = time.time() - t0
        print(
            f"  [fink]   batch {bi}/{n_batches}: {len(chunk)} ids -> "
            f"{len(rows)} rows / {len(raw_per_oid)} objs ({dt:.1f}s)"
        )
    return results


def _merge_lsst_dets(*sources: list[dict]) -> list[dict]:
    """Union detections from multiple sources, dedup by (band, round(mjd,5)).

    Earlier sources win on conflict — pass the richest source first
    (ALeRCE has reliability+snr → prefer it over Fink → prefer over Lasair).
    Output is sorted ascending by mjd.
    """
    seen: dict[tuple, dict] = {}
    for src in sources:
        if not src:
            continue
        for det in src:
            mjd = det.get("mjd")
            try:
                mjd_val = float(mjd)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(mjd_val):
                continue
            key = (det.get("band", "?"), round(mjd_val, 5))
            if key not in seen:
                seen[key] = det
    merged = sorted(seen.values(), key=lambda d: float(d.get("mjd") or 0.0))
    return merged


def fetch_lightcurves_for_objects(
    *,
    object_ids: list[str],
    fetcher: AlerceHistoryFetcher,
    lc_dir: Path,
    max_epochs: int = 20,
    associations: dict[str, dict[str, object]] | None = None,
    fixtures_dir: Path | None = None,
    alerce_lsst_map: dict[str, list[dict]] | None = None,
    fink_lsst_map: dict[str, list[dict]] | None = None,
    lsst_max_epochs: int | None = None,
) -> tuple[int, int]:
    ok = 0
    skipped = 0
    alerce_lsst_map = alerce_lsst_map or {}
    fink_lsst_map = fink_lsst_map or {}
    lsst_cap = lsst_max_epochs if lsst_max_epochs is not None else max(max_epochs, 50)

    for oid in object_ids:
        cache = lc_dir / f"{oid}.json"
        if cache.exists():
            print(f"  {oid}: cached")
            ok += 1
            continue

        id_kind = infer_identifier_kind(oid)

        # --- LSST objects: union of ALeRCE + Fink + Lasair ---
        if id_kind == "lsst_dia_object_id":
            alerce_dets = alerce_lsst_map.get(str(oid), []) or []
            fink_dets = fink_lsst_map.get(str(oid), []) or []
            fixture_dets = _fetch_lsst_from_fixture(oid, fixtures_dir) if fixtures_dir else None
            fixture_dets = fixture_dets or []
            if fixture_dets:
                for d in fixture_dets:
                    d.setdefault("_source", "lasair_fixture")

            lsst_dets = _merge_lsst_dets(alerce_dets, fink_dets, fixture_dets)

            if not lsst_dets:
                # Last-ditch: live Lasair call (usually redundant with fixtures)
                live = _fetch_lsst_lightcurve_from_lasair(oid)
                if live:
                    for d in live:
                        d.setdefault("_source", "lasair_live")
                    lsst_dets = live

            if lsst_dets:
                if lsst_cap and len(lsst_dets) > lsst_cap:
                    lsst_dets = lsst_dets[:lsst_cap]
                lc_dir.mkdir(parents=True, exist_ok=True)
                with open(cache, "w") as fh:
                    json.dump(lsst_dets, fh, indent=2, default=str)
                src_counts = {}
                for d in lsst_dets:
                    src_counts[d.get("_source", "?")] = src_counts.get(d.get("_source", "?"), 0) + 1
                src_str = ", ".join(f"{v} {k}" for k, v in src_counts.items())
                print(f"  {oid}: {len(lsst_dets)} LSST dets ({src_str})")
                ok += 1
            else:
                # Fall back to ZTF via association
                reference = resolve_object_reference(
                    oid, broker="alerce_history", associations=associations,
                )
                if reference.can_query:
                    epochs = fetcher.fetch_epoch_scores(
                        reference.requested_object_id, max_epochs=max_epochs,
                    )
                    if epochs:
                        source_cache = lc_dir / f"{reference.requested_object_id}.json"
                        if source_cache.exists():
                            shutil.copyfile(source_cache, cache)
                            print(f"  {oid}: {len(epochs)} epochs (via ZTF {reference.requested_object_id})")
                            ok += 1
                            continue
                print(f"  {oid}: no LSST or ZTF lightcurve source")
                skipped += 1
            continue

        # --- ZTF objects: existing ALeRCE fetch path ---
        reference = resolve_object_reference(
            oid,
            broker="alerce_history",
            associations=associations,
        )
        if not reference.can_query:
            print(f"  {oid}: {reference.resolution_reason or 'no lightcurve source'}")
            skipped += 1
            continue

        epochs = fetcher.fetch_epoch_scores(reference.requested_object_id, max_epochs=max_epochs)
        if not epochs:
            print(f"  {oid}: no data")
            skipped += 1
            continue

        if reference.requested_object_id != oid:
            source_cache = lc_dir / f"{reference.requested_object_id}.json"
            if source_cache.exists():
                shutil.copyfile(source_cache, cache)
                print(
                    f"  {oid}: {len(epochs)} detection epochs "
                    f"(via {reference.requested_object_id})"
                )
            else:
                print(f"  {oid}: no cached lightcurve for associated object {reference.requested_object_id}")
                skipped += 1
                continue
        else:
            print(f"  {oid}: {len(epochs)} detection epochs")
        ok += 1

    return ok, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and cache lightcurves")
    parser.add_argument("--objects", nargs="+", default=None)
    parser.add_argument("--from-labels", default="data/labels.csv",
                        help="CSV with object_id column")
    parser.add_argument("--association-csv", default=None,
                        help="Optional LSST→ZTF association CSV for LSST primary objects")
    parser.add_argument("--lc-dir", default="data/lightcurves")
    parser.add_argument("--max-epochs", type=int, default=20,
                        help="ZTF cap; LSST uses --lsst-max-epochs or max(this,50)")
    parser.add_argument("--lsst-max-epochs", type=int, default=None,
                        help="Per-LSST-object detection cap after union (default: max(50, --max-epochs))")
    parser.add_argument("--only-kind", choices=["ztf_object_id", "lsst_dia_object_id"],
                        default=None, help="Filter to one identifier kind")
    parser.add_argument("--fixtures-dir", default="fixtures/raw/lasair",
                        help="Lasair LSST fixture dir (100%% coverage from Phase-1 backfill)")
    parser.add_argument("--alerce-batch", type=int, default=50)
    parser.add_argument("--fink-batch", type=int, default=50)
    parser.add_argument("--no-alerce-lsst", action="store_true",
                        help="Skip ALeRCE LSST (use Fink + fixtures only)")
    parser.add_argument("--no-fink-lsst", action="store_true",
                        help="Skip Fink LSST /fp (use ALeRCE + fixtures only)")
    args = parser.parse_args()

    lc_dir = Path(args.lc_dir)
    fixtures_dir = Path(args.fixtures_dir) if args.fixtures_dir else None

    if args.objects:
        oids = args.objects
    elif Path(args.from_labels).exists():
        oids = _load_oids_from_labels(Path(args.from_labels))
    else:
        print(f"No objects specified and {args.from_labels} not found.")
        sys.exit(1)

    if args.only_kind:
        before = len(oids)
        oids = [o for o in oids if infer_identifier_kind(o) == args.only_kind]
        print(f"Filter --only-kind={args.only_kind}: {before} -> {len(oids)}")

    fetcher = AlerceHistoryFetcher(lc_dir=lc_dir)
    associations = load_lsst_ztf_associations(Path(args.association_csv)) if args.association_csv else {}
    print(f"Fetching lightcurves for {len(oids)} objects → {lc_dir}")

    # Pre-batch-fetch LSST data from ALeRCE + Fink for uncached ids
    lsst_todo = [
        o for o in oids
        if infer_identifier_kind(o) == "lsst_dia_object_id"
        and not (lc_dir / f"{o}.json").exists()
    ]
    alerce_map: dict[str, list[dict]] = {}
    fink_map: dict[str, list[dict]] = {}
    if lsst_todo:
        print(f"\nLSST pre-fetch: {len(lsst_todo)} uncached objects")
        if not args.no_alerce_lsst:
            print("[alerce] batch-fetch /lightcurve_api/detections ...")
            alerce_map = _batch_fetch_alerce_lsst(lsst_todo, batch_size=args.alerce_batch)
            print(f"[alerce] got data for {len(alerce_map)}/{len(lsst_todo)} objects")
        if not args.no_fink_lsst:
            print("[fink] batch-fetch /api/v1/fp ...")
            fink_map = _batch_fetch_fink_lsst_fp(lsst_todo, batch_size=args.fink_batch)
            print(f"[fink] got data for {len(fink_map)}/{len(lsst_todo)} objects")
        print()

    ok, skipped = fetch_lightcurves_for_objects(
        object_ids=oids,
        fetcher=fetcher,
        lc_dir=lc_dir,
        max_epochs=args.max_epochs,
        associations=associations,
        fixtures_dir=fixtures_dir,
        alerce_lsst_map=alerce_map,
        fink_lsst_map=fink_map,
        lsst_max_epochs=args.lsst_max_epochs,
    )

    print(f"\nDone: {ok} fetched, {skipped} skipped")


if __name__ == "__main__":
    main()
