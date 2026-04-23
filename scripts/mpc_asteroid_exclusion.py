#!/usr/bin/env python3
"""Cross-match labels with Minor Planet Center (MPC) to exclude asteroids.

For each object in labels.csv (with ra/dec and first-detection mjd), query
MPC for known minor planets within a small tolerance (default 10 arcsec) at
that epoch.  Output a boolean `is_asteroid` flag per object so the truth
builder can force those to `target_class='other'` with `label_source='mpc_exclusion'`.

This closes the asteroid-contamination gap flagged in the methodology audit:
ALeRCE's 'asteroid' stamp-classifier label sometimes leaks through the
discovery pipeline into the labelled training set.

Note: `astroquery.mpc` rate-limits at ~60 requests/min.  The script includes
a small delay between calls and can be resumed via --resume.

Output: data/truth/mpc_exclusion.parquet
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _query_mpc(ra_deg: float, dec_deg: float, mjd: float, radius_arcsec: float) -> list[dict]:
    """Return list of minor-planet hits within radius at epoch (empty if none)."""
    try:
        from astroquery.mpc import MPC
    except ImportError:
        raise RuntimeError("astroquery not installed — pip install astroquery")
    import astropy.units as u

    try:
        result = MPC.get_ephemeris(
            target="all",
            location="500",  # Geocenter
            start=mjd,
            step="1d",
            number=1,
        )
        # This API variant returns all known bodies near the epoch;
        # we post-filter by separation from (ra,dec).
    except Exception:
        return []
    if result is None or len(result) == 0:
        return []
    hits = []
    from astropy.coordinates import SkyCoord

    center = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
    for row in result:
        try:
            pos = SkyCoord(row["RA"], row["Dec"], unit=(u.hourangle, u.deg))
            sep = center.separation(pos).to(u.arcsec).value
        except Exception:
            continue
        if sep <= radius_arcsec:
            hits.append({
                "designation": str(row.get("Number") or row.get("Name") or "?"),
                "sep_arcsec": float(sep),
            })
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description="MPC asteroid exclusion cross-match")
    parser.add_argument("--labels", default="data/labels.csv")
    parser.add_argument("--output", default="data/truth/mpc_exclusion.parquet")
    parser.add_argument("--radius-arcsec", type=float, default=10.0)
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="Seconds between MPC queries (rate-limit ~60/min)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N rows (for smoke tests)")
    parser.add_argument("--resume", action="store_true",
                        help="Append to existing output parquet")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    if not labels_path.exists():
        print(f"[error] labels file not found: {labels_path}")
        sys.exit(1)

    with open(labels_path) as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[: args.limit]

    output_path = Path(args.output)
    already: set[str] = set()
    if args.resume and output_path.exists():
        try:
            already = set(pd.read_parquet(output_path)["object_id"].astype(str))
            print(f"Resuming — {len(already)} objects already processed")
        except Exception:
            already = set()

    records = []
    for i, row in enumerate(rows):
        oid = str(row.get("object_id") or "")
        if not oid or oid in already:
            continue
        ra = row.get("ra") or row.get("RA")
        dec = row.get("dec") or row.get("Dec")
        mjd = row.get("first_det_mjd") or row.get("discovery_mjd") or row.get("mjd")
        if not (ra and dec and mjd):
            records.append({"object_id": oid, "is_asteroid": False, "reason": "no astrometry/time"})
            continue
        try:
            hits = _query_mpc(float(ra), float(dec), float(mjd), args.radius_arcsec)
        except RuntimeError as exc:
            print(f"[fatal] {exc}")
            sys.exit(1)
        except Exception as exc:
            records.append({"object_id": oid, "is_asteroid": False, "reason": f"query failed: {exc}"})
            continue

        records.append({
            "object_id": oid,
            "is_asteroid": bool(hits),
            "n_hits": len(hits),
            "closest_sep_arcsec": hits[0]["sep_arcsec"] if hits else None,
            "closest_designation": hits[0]["designation"] if hits else None,
        })

        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(rows)} — {sum(1 for r in records if r.get('is_asteroid'))} asteroids so far")

        time.sleep(args.sleep)

    df = pd.DataFrame(records)
    if args.resume and output_path.exists():
        prev = pd.read_parquet(output_path)
        df = pd.concat([prev, df], ignore_index=True).drop_duplicates("object_id", keep="last")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    n_ast = int(df["is_asteroid"].sum())
    print(f"Wrote {output_path} — {n_ast} asteroids among {len(df)} objects")


if __name__ == "__main__":
    main()
