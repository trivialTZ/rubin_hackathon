#!/usr/bin/env python3
"""DP1 sky regions + time span (third pass — diverse sampling)."""
from __future__ import annotations
import os, sys
from pathlib import Path
import pyvo, requests


def load_token(repo_root):
    tok = os.environ.get("RSP_TOKEN")
    if tok: return tok
    for line in (repo_root / ".env").read_text().splitlines():
        s = line.strip()
        if s.startswith("RSP_TOKEN="):
            return s.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit("no token")


def tap(url, token):
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    return pyvo.dal.TAPService(url, session=s)


def run(svc, adql, label):
    print(f"\n--- {label} ---")
    try:
        print(svc.search(adql).to_table())
    except Exception as e:
        print(f"ERR: {type(e).__name__}: {str(e)[:200]}")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    svc = tap("https://data.lsst.cloud/api/tap", load_token(repo_root))

    # MJD span from Visit (smaller table)
    run(svc,
        "SELECT MIN(obsStartMJD) AS mjd_min, MAX(obsStartMJD) AS mjd_max, "
        "COUNT(DISTINCT band) AS n_bands FROM dp1.Visit",
        "DP1 Visit MJD span + bands")

    # Visits per band
    run(svc,
        "SELECT band, COUNT(*) AS n_visits FROM dp1.Visit "
        "GROUP BY band ORDER BY band",
        "Visits per band (telescope exposures)")

    # Sky footprint via Visit (ra, dec) — smaller table so full scan OK
    run(svc,
        "SELECT TOP 20 ra, dec, COUNT(*) AS n_visits FROM dp1.Visit "
        "GROUP BY ra, dec ORDER BY n_visits DESC",
        "Top 20 most-visited pointings")

    # Well-sampled DiaObject — spread across the sky by selecting from different RA/Dec ranges
    for ra_lo, ra_hi, label in [(0, 15, "SMC/early-RA"), (15, 50, "RA 15-50"),
                                  (50, 80, "RA 50-80"), (80, 107, "ECDFS/late-RA")]:
        run(svc,
            f"SELECT TOP 5 diaObjectId, ra, dec, nDiaSources "
            f"FROM dp1.DiaObject "
            f"WHERE nDiaSources >= 10 AND ra >= {ra_lo} AND ra < {ra_hi} "
            f"ORDER BY nDiaSources DESC",
            f"Well-sampled obj in {label}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
