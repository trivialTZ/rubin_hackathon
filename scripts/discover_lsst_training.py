"""Discover LSST transient candidates for mixed-survey training.

Queries Lasair LSST (fast SQL API) for multi-detection transients with
Sherlock SN/NT classification, then enriches with ALeRCE stamp scores.
Outputs a candidates CSV that can be appended to labels.csv for training.

Usage:
    python3 scripts/discover_lsst_training.py --max-candidates 200
    python3 scripts/discover_lsst_training.py --min-detections 3 --output data/lsst_candidates.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import requests

from debass_meta.access.identifiers import infer_identifier_kind


def _lasair_query(
    *,
    token: str,
    endpoint: str = "https://lasair.lsst.ac.uk",
    selected: str,
    tables: str,
    conditions: str,
    limit: int = 1000,
) -> list[dict]:
    """Run a SQL query against Lasair LSST API."""
    api_base = endpoint.rstrip("/")
    if not api_base.endswith("/api"):
        api_base += "/api"
    resp = requests.post(
        f"{api_base}/query/",
        headers={"Authorization": f"Token {token}", "Accept": "application/json"},
        data={
            "selected": selected,
            "tables": tables,
            "conditions": conditions,
            "limit": str(limit),
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else []


def _alerce_stamp_scores(oids: list[str]) -> dict[str, dict]:
    """Query ALeRCE stamp_classifier_rubin_beta for a list of LSST oids."""
    try:
        from alerce.core import Alerce
        import inspect
        a = Alerce()
        sig = inspect.signature(a.query_probabilities)
        has_survey = "survey" in sig.parameters
    except ImportError:
        return {}

    from concurrent.futures import ThreadPoolExecutor

    def _fetch_one(oid: str) -> tuple[str, dict | None]:
        try:
            kwargs = {"oid": oid, "format": "pandas"}
            if has_survey:
                kwargs["survey"] = "lsst"
            probs = a.query_probabilities(**kwargs)
            if len(probs) == 0:
                return oid, None
            scores = {}
            for _, row in probs.iterrows():
                cn = row.get("classifier_name", "")
                cls = row.get("class_name", "")
                p = row.get("probability")
                scores[f"{cn}__{cls}"] = p
            # Extract SN probability from stamp_classifier_rubin_beta
            p_sn = scores.get("stamp_classifier_rubin_beta__SN", 0.0)
            return oid, {"p_sn_stamp": p_sn, "n_classifiers": len(probs)}
        except Exception:
            return oid, None

    results = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for oid, data in pool.map(lambda o: _fetch_one(o), oids):
            if data is not None:
                results[oid] = data
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover LSST transient candidates")
    parser.add_argument("--min-detections", type=int, default=2,
                        help="Minimum number of detections")
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument("--output", default="data/lsst_candidates.csv")
    parser.add_argument("--no-alerce", action="store_true",
                        help="Skip ALeRCE stamp enrichment")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    token = os.environ.get("LASAIR_LSST_TOKEN") or os.environ.get("LASAIR_TOKEN", "")
    endpoint = os.environ.get("LASAIR_LSST_ENDPOINT",
               os.environ.get("LASAIR_ENDPOINT", "https://lasair.lsst.ac.uk"))

    if not token:
        print("ERROR: Set LASAIR_TOKEN or LASAIR_LSST_TOKEN for Lasair LSST API access")
        sys.exit(1)

    print(f"Querying Lasair LSST ({endpoint}) for transients with >= {args.min_detections} detections...")

    # Query Lasair LSST for multi-detection objects
    # nDiaSources >= min_detections, sort by nDiaSources descending
    try:
        rows = _lasair_query(
            token=token,
            endpoint=endpoint,
            selected="objects.diaObjectId, objects.ra, objects.decl, "
                     "objects.nDiaSources, objects.firstDiaSourceMjdTai, objects.lastDiaSourceMjdTai, "
                     "sherlock_classifications.classification AS sherlock_class",
            tables="objects, sherlock_classifications",
            conditions=f"objects.nDiaSources >= {args.min_detections} "
                       "AND sherlock_classifications.classification IN ('SN', 'NT') "
                       "AND objects.diaObjectId = sherlock_classifications.diaObjectId",
            limit=args.max_candidates * 2,  # fetch extra, filter later
        )
    except Exception as e:
        print(f"Lasair query failed: {e}")
        print("Trying simpler query without Sherlock join...")
        try:
            rows = _lasair_query(
                token=token,
                endpoint=endpoint,
                selected="diaObjectId, ra, decl, nDiaSources, "
                         "firstDiaSourceMjdTai, lastDiaSourceMjdTai",
                tables="objects",
                conditions=f"nDiaSources >= {args.min_detections}",
                limit=args.max_candidates,
            )
        except Exception as e2:
            print(f"Simple query also failed: {e2}")
            sys.exit(1)

    print(f"  Lasair returned {len(rows)} candidates")

    if not rows:
        print("No LSST candidates found.")
        sys.exit(0)

    # Build candidates list
    candidates = []
    for row in rows:
        oid = str(row.get("diaObjectId", ""))
        if not oid:
            continue
        candidates.append({
            "object_id": oid,
            "ra": float(row.get("ra") or 0),
            "dec": float(row.get("decl") or row.get("dec") or 0),
            "n_det": int(row.get("nDiaSources") or 0),
            "first_mjd": float(row.get("firstDiaSourceMjdTai") or 0),
            "last_mjd": float(row.get("lastDiaSourceMjdTai") or 0),
            "sherlock_class": row.get("sherlock_class") or row.get("classification") or "",
            "discovery_broker": "lasair_lsst",
            "survey": "LSST",
        })

    # Sort by n_det descending
    candidates.sort(key=lambda c: c["n_det"], reverse=True)
    candidates = candidates[:args.max_candidates]

    print(f"  After filtering: {len(candidates)} candidates (min n_det={args.min_detections})")
    if candidates:
        print(f"  n_det range: {candidates[-1]['n_det']} — {candidates[0]['n_det']}")

    # Enrich with ALeRCE stamp scores
    if not args.no_alerce and candidates:
        oids = [c["object_id"] for c in candidates]
        print(f"  Querying ALeRCE for {len(oids)} LSST objects (8 threads)...")
        alerce_scores = _alerce_stamp_scores(oids)
        for c in candidates:
            scores = alerce_scores.get(c["object_id"], {})
            c["alerce_stamp_p_sn"] = scores.get("p_sn_stamp", "")
        n_with_alerce = sum(1 for c in candidates if c.get("alerce_stamp_p_sn", "") != "")
        print(f"  {n_with_alerce}/{len(candidates)} have ALeRCE stamp scores")

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["object_id", "ra", "dec", "n_det", "first_mjd", "last_mjd",
                  "sherlock_class", "discovery_broker", "survey", "alerce_stamp_p_sn"]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(candidates)

    print(f"\nWrote {len(candidates)} LSST candidates → {out_path}")
    print(f"\nTo add to training data:")
    print(f"  # 1. Append to labels.csv")
    print(f"  python3 scripts/merge_labels.py --existing data/labels.csv --new {out_path}")
    print(f"  # 2. Fetch lightcurves + broker scores")
    print(f"  python3 scripts/fetch_lightcurves.py --from-labels data/labels.csv")
    print(f"  python3 scripts/backfill.py --broker all --from-labels data/labels.csv --parallel 8")


if __name__ == "__main__":
    main()
