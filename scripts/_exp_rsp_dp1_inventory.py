#!/usr/bin/env python3
"""DP1 inventory: counts, MJD range, nDiaSources bucket, 10-obj sample."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pyvo
import requests


def load_token(repo_root: Path) -> str:
    tok = os.environ.get("RSP_TOKEN")
    if tok:
        return tok
    for line in (repo_root / ".env").read_text().splitlines():
        s = line.strip()
        if s.startswith("RSP_TOKEN="):
            return s.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit("RSP_TOKEN not found")


def tap(url: str, token: str) -> pyvo.dal.TAPService:
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"
    return pyvo.dal.TAPService(url, session=session)


def run(svc, adql: str, label: str):
    print(f"\n--- {label} ---")
    try:
        tbl = svc.search(adql).to_table()
        print(tbl)
        return tbl
    except Exception as e:
        print(f"ERR: {type(e).__name__}: {str(e)[:250]}")
        return None


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    token = load_token(repo_root)
    svc = tap("https://data.lsst.cloud/api/tap", token)

    # MJD range from DiaSource (has per-detection timestamp)
    run(svc,
        "SELECT MIN(midpointMjdTai) AS mjd_min, "
        "MAX(midpointMjdTai) AS mjd_max, "
        "COUNT(DISTINCT band) AS n_bands "
        "FROM dp1.DiaSource",
        "DP1 DiaSource MJD + band count")

    # Band distribution from DiaSource
    run(svc,
        "SELECT band, COUNT(*) AS n_det FROM dp1.DiaSource "
        "GROUP BY band ORDER BY band",
        "DP1 detections per band")

    # nDiaSources distribution (buckets)
    for lo, hi, label in [(1, 4, "1-3"), (5, 9, "5-9"), (10, 29, "10-29"), (30, 10000, "30+")]:
        run(svc,
            f"SELECT COUNT(*) AS n FROM dp1.DiaObject "
            f"WHERE nDiaSources >= {lo} AND nDiaSources < {hi + 1}",
            f"DP1 DiaObject with nDiaSources {label}")

    # Sample 10 objects with decent cadence
    tbl = run(svc,
        "SELECT TOP 10 diaObjectId, ra, dec, nDiaSources, radecMjdTai "
        "FROM dp1.DiaObject WHERE nDiaSources >= 5 ORDER BY diaObjectId",
        "DP1 sample diaObjectIds (nDiaSources >= 5)")

    if tbl is not None and len(tbl) > 0:
        out = repo_root / "reports" / "rsp_probe"
        out.mkdir(parents=True, exist_ok=True)
        tbl.write(str(out / "dp1_sample_10.ecsv"), overwrite=True, format="ascii.ecsv")
        print(f"\nSaved sample to {out / 'dp1_sample_10.ecsv'}")
        print("\ndiaObjectIds for downstream probes:")
        for row in tbl:
            print(f"  {row['diaObjectId']}  ra={row['ra']:.4f}  dec={row['dec']:.4f}  n={row['nDiaSources']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
