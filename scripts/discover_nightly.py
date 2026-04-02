"""scripts/discover_nightly.py — Discover new transient candidates from broker APIs.

Queries ALeRCE, Fink, and Lasair for recently detected SN-like transients,
deduplicates across brokers by sky position, and writes a candidates CSV
compatible with the existing --from-labels pipeline.

Usage:
    python scripts/discover_nightly.py --lookback-days 2 --output data/nightly/2026-04-02/candidates.csv
    python scripts/discover_nightly.py --lookback-days 7 --max-candidates 10  # smoke test
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from debass_meta.access.identifiers import build_lasair_api_endpoint, is_lsst_lasair_endpoint
from debass_meta.access.lasair import extract_sherlock_label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mjd_now() -> float:
    """Current Modified Julian Date (UTC)."""
    # MJD = JD - 2400000.5; J2000 epoch: 2000-01-01T12:00:00 = MJD 51544.5
    unix_now = time.time()
    return unix_now / 86400.0 + 40587.0


def _date_to_mjd(date_str: str) -> float:
    """Convert YYYY-MM-DD string to MJD."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    unix_ts = dt.timestamp()
    return unix_ts / 86400.0 + 40587.0


def _mjd_to_jd(mjd: float) -> float:
    return mjd + 2400000.5


def _haversine_deg(ra1, dec1, ra2, dec2):
    """Angular separation in arcseconds between two sky positions (degrees)."""
    ra1, dec1, ra2, dec2 = map(np.radians, [ra1, dec1, ra2, dec2])
    dlat = dec2 - dec1
    dlon = ra2 - ra1
    a = np.sin(dlat / 2) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin(dlon / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(a))) * 3600  # arcsec


# ---------------------------------------------------------------------------
# Broker queries
# ---------------------------------------------------------------------------

def query_alerce_recent(since_mjd: float, max_candidates: int = 500,
                        max_pages: int = 10, min_prob: float = 0.3) -> pd.DataFrame:
    """Query ALeRCE for recently detected SN-like objects.

    Queries stamp_classifier (SN) and lc_classifier_transient (SNIa, SNIbc,
    SNII, SLSN) classes, filters by lastmjd >= since_mjd.
    """
    try:
        from alerce.core import Alerce
    except ImportError:
        print("[alerce] alerce package not installed — skipping", file=sys.stderr)
        return pd.DataFrame()

    client = Alerce()
    rows = []

    queries = [
        ("stamp_classifier", "SN"),
        ("lc_classifier_transient", "SNIa"),
        ("lc_classifier_transient", "SNIbc"),
        ("lc_classifier_transient", "SNII"),
        ("lc_classifier_transient", "SLSN"),
    ]

    for classifier, class_name in queries:
        collected = 0
        for page in range(1, max_pages + 1):
            try:
                result = client.query_objects(
                    classifier=classifier,
                    class_name=class_name,
                    format="pandas",
                    page_size=100,
                    page=page,
                )
            except Exception as exc:
                print(f"  [alerce] {classifier}/{class_name} page {page}: {exc}",
                      file=sys.stderr)
                break

            if result is None or len(result) == 0:
                break

            # Filter by recency
            if "lastmjd" in result.columns:
                result = result[result["lastmjd"] >= since_mjd]

            for _, obj in result.iterrows():
                oid = obj.get("oid") or obj.name
                rows.append({
                    "object_id": str(oid),
                    "ra": float(obj.get("meanra", np.nan)),
                    "dec": float(obj.get("meandec", np.nan)),
                    "n_det": int(obj.get("ndet", 0)),
                    "last_mjd": float(obj.get("lastmjd", np.nan)),
                    "discovery_broker": "alerce",
                })
                collected += 1

            if collected >= max_candidates:
                break
            time.sleep(0.15)  # courtesy rate limit

        print(f"  [alerce] {classifier}/{class_name}: {collected} objects")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).drop_duplicates(subset="object_id", keep="first")
    return df.head(max_candidates)


def query_fink_recent(since_date: str, until_date: str,
                      max_candidates: int = 500) -> pd.DataFrame:
    """Query Fink /api/v1/latests for recent SN Ia candidates."""
    url = "https://api.fink-portal.org/api/v1/latests"
    rows = []

    fink_classes = [
        "Early SN Ia candidate",
        "(TNS) SN Ia",
        "(TNS) SN II",
        "(TNS) SN Ibc",
    ]

    for fink_class in fink_classes:
        payload = {
            "class": fink_class,
            "n": str(max_candidates),
            "startdate": since_date,
            "stopdate": until_date,
            "columns": "i:objectId,i:ra,i:dec,i:ndethist,i:jd,d:snn_snia_vs_nonia",
        }
        try:
            r = requests.post(url, json=payload, timeout=30, verify=True)
            if r.status_code == 200:
                data = r.json()
            else:
                # SSL fallback (matches fink.py pattern)
                warnings.warn(f"Fink returned {r.status_code}, retrying without SSL verify")
                r = requests.post(url, json=payload, timeout=30, verify=False)
                data = r.json()
        except requests.exceptions.SSLError:
            warnings.warn("Fink SSL error — retrying without verify")
            try:
                r = requests.post(url, json=payload, timeout=30, verify=False)
                data = r.json()
            except Exception as exc:
                print(f"  [fink] {fink_class}: request failed — {exc}", file=sys.stderr)
                continue
        except Exception as exc:
            print(f"  [fink] {fink_class}: request failed — {exc}", file=sys.stderr)
            continue

        if not data:
            print(f"  [fink] {fink_class}: 0 objects")
            continue

        for item in data:
            jd = float(item.get("i:jd", 0))
            rows.append({
                "object_id": str(item.get("i:objectId", "")),
                "ra": float(item.get("i:ra", np.nan)),
                "dec": float(item.get("i:dec", np.nan)),
                "n_det": int(item.get("i:ndethist", 0)),
                "last_mjd": jd - 2400000.5 if jd > 0 else np.nan,
                "discovery_broker": "fink",
            })

        print(f"  [fink] {fink_class}: {len(data)} objects")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).drop_duplicates(subset="object_id", keep="first")
    return df.head(max_candidates)


def query_lasair_recent(since_mjd: float, max_candidates: int = 500) -> pd.DataFrame:
    """Query Lasair for recent SN/NT-classified objects."""
    token = os.environ.get("LASAIR_TOKEN")
    if not token:
        print("  [lasair] LASAIR_TOKEN not set — skipping", file=sys.stderr)
        return pd.DataFrame()

    site_base = os.environ.get("LASAIR_ENDPOINT", "https://lasair-ztf.lsst.ac.uk").rstrip("/")
    api_base = build_lasair_api_endpoint(site_base)
    lsst_mode = is_lsst_lasair_endpoint(site_base)

    try:
        from lasair import lasair_client as lasair_mod
    except ImportError:
        print("  [lasair] lasair package not installed — skipping", file=sys.stderr)
        return pd.DataFrame()

    try:
        try:
            client = lasair_mod(token, endpoint=api_base)
        except TypeError:
            if lsst_mode:
                raise RuntimeError("installed lasair package does not support LASAIR_ENDPOINT override")
            client = lasair_mod(token)
    except Exception as exc:
        print(f"  [lasair] client init failed — {exc}", file=sys.stderr)
        return pd.DataFrame()

    if not lsst_mode:
        try:
            results = client.query(
                "objectId,ramean,decmean,ncand,jdmax,classification",
                "objects",
                (
                    f"classification IN ('SN', 'NT') "
                    f"AND jdmax > {_mjd_to_jd(since_mjd)} "
                    f"ORDER BY jdmax DESC"
                ),
                limit=max_candidates,
            )
        except Exception as exc:
            print(f"  [lasair] query failed — {exc}", file=sys.stderr)
            return pd.DataFrame()

        rows = []
        for obj in results:
            jdmax = float(obj.get("jdmax", 0))
            rows.append({
                "object_id": str(obj.get("objectId", "")),
                "ra": float(obj.get("ramean", np.nan)),
                "dec": float(obj.get("decmean", np.nan)),
                "n_det": int(obj.get("ncand", 0)),
                "last_mjd": jdmax - 2400000.5 if jdmax > 0 else np.nan,
                "discovery_broker": "lasair",
            })

        print(f"  [lasair] ZTF SN/NT: {len(rows)} objects")
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).head(max_candidates)

    query_limit = min(max(max_candidates * 5, 50), 500)
    try:
        results = client.query(
            "diaObjectId,ra,decl,nDiaSources,lastDiaSourceMjdTai",
            "objects",
            f"lastDiaSourceMjdTai > {since_mjd} ORDER BY lastDiaSourceMjdTai DESC",
            limit=query_limit,
        )
    except Exception as exc:
        print(f"  [lasair] LSST query failed — {exc}", file=sys.stderr)
        return pd.DataFrame()

    headers = {"Accept": "application/json", "Authorization": f"Token {token}"}
    rows = []
    for obj in results:
        object_id = str(obj.get("diaObjectId", "")).strip()
        if not object_id:
            continue
        try:
            response = requests.post(
                f"{api_base}/object/",
                headers=headers,
                data={"objectId": object_id},
                timeout=30,
            )
            if response.status_code != 200:
                continue
            payload = response.json()
        except Exception:
            continue

        sherlock_label = extract_sherlock_label(payload)
        if sherlock_label not in {"SN", "NT"}:
            continue

        rows.append({
            "object_id": object_id,
            "ra": float(obj.get("ra", np.nan)),
            "dec": float(obj.get("decl", np.nan)),
            "n_det": int(obj.get("nDiaSources", 0)),
            "last_mjd": float(obj.get("lastDiaSourceMjdTai", np.nan)),
            "discovery_broker": "lasair",
        })
        if len(rows) >= max_candidates:
            break

    print(f"  [lasair] LSST Sherlock SN/NT: {len(rows)} objects")
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_candidates(dfs: list[pd.DataFrame],
                           match_radius_arcsec: float = 2.0) -> pd.DataFrame:
    """Merge candidate lists from multiple brokers, dedup by sky position.

    When two objects from different brokers are within match_radius_arcsec,
    keep the one with more detections and note the other broker.
    """
    combined = pd.concat([df for df in dfs if len(df) > 0], ignore_index=True)
    if len(combined) == 0:
        return combined

    # Remove rows with missing coordinates
    combined = combined.dropna(subset=["ra", "dec"])
    if len(combined) == 0:
        return combined

    # Sort by n_det descending so we keep the highest-detection entry
    combined = combined.sort_values("n_det", ascending=False).reset_index(drop=True)

    # Simple O(n^2) pairwise matching — fine for <1000 objects
    keep = np.ones(len(combined), dtype=bool)
    also_in = [""] * len(combined)

    for i in range(len(combined)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(combined)):
            if not keep[j]:
                continue
            sep = _haversine_deg(
                combined.iloc[i]["ra"], combined.iloc[i]["dec"],
                combined.iloc[j]["ra"], combined.iloc[j]["dec"],
            )
            if sep <= match_radius_arcsec:
                # Mark j as duplicate (i has more detections due to sorting)
                keep[j] = False
                broker_j = combined.iloc[j]["discovery_broker"]
                if also_in[i]:
                    also_in[i] += f",{broker_j}"
                else:
                    also_in[i] = broker_j

    result = combined[keep].copy()
    result["also_in_brokers"] = [also_in[i] for i in range(len(combined)) if keep[i]]
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover new transient candidates from broker APIs"
    )
    parser.add_argument("--lookback-days", type=int, default=2,
                        help="Query brokers for objects detected in last N days (default: 2)")
    parser.add_argument("--output", required=True,
                        help="Output CSV path (e.g. data/nightly/2026-04-02/candidates.csv)")
    parser.add_argument("--min-detections", type=int, default=2,
                        help="Minimum number of detections (default: 2)")
    parser.add_argument("--max-candidates", type=int, default=500,
                        help="Max candidates per broker (default: 500)")
    parser.add_argument("--match-radius", type=float, default=2.0,
                        help="Cross-match radius in arcsec (default: 2.0)")
    args = parser.parse_args()

    now_mjd = _mjd_now()
    since_mjd = now_mjd - args.lookback_days

    # Date strings for Fink
    since_date = (datetime.now(timezone.utc) - timedelta(days=args.lookback_days)).strftime("%Y-%m-%d")
    until_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print(f"=== DEBASS Nightly Discovery ===")
    print(f"Date: {until_date}")
    print(f"Lookback: {args.lookback_days} days (since MJD {since_mjd:.1f})")
    print()

    # Query each broker
    dfs = []

    print("Querying ALeRCE...")
    df_alerce = query_alerce_recent(since_mjd, max_candidates=args.max_candidates)
    dfs.append(df_alerce)

    print("Querying Fink...")
    df_fink = query_fink_recent(since_date, until_date, max_candidates=args.max_candidates)
    dfs.append(df_fink)

    print("Querying Lasair...")
    df_lasair = query_lasair_recent(since_mjd, max_candidates=args.max_candidates)
    dfs.append(df_lasair)

    # Combine and deduplicate
    print("\nDeduplicating...")
    candidates = deduplicate_candidates(dfs, match_radius_arcsec=args.match_radius)

    # Filter by minimum detections
    if len(candidates) > 0 and args.min_detections > 0:
        candidates = candidates[candidates["n_det"] >= args.min_detections]

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    candidates.to_csv(out_path, index=False)

    print(f"\n=== Summary ===")
    print(f"ALeRCE: {len(df_alerce)} candidates")
    print(f"Fink:   {len(df_fink)} candidates")
    print(f"Lasair: {len(df_lasair)} candidates")
    print(f"After dedup + filter: {len(candidates)} unique candidates")
    print(f"Output: {out_path}")

    if len(candidates) == 0:
        print("\nNo candidates found. This may be expected during intermittent early operations.")
        sys.exit(0)


if __name__ == "__main__":
    main()
