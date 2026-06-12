#!/usr/bin/env python3
"""Check reliability distribution across DP1 DiaSource — is 0.03 normal or a red flag?"""
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

    # Reliability distribution across all of DP1 DiaSource
    run(svc,
        "SELECT MIN(reliability) AS rel_min, MAX(reliability) AS rel_max, "
        "AVG(reliability) AS rel_mean, "
        "COUNT(*) AS n "
        "FROM dp1.DiaSource WHERE reliability IS NOT NULL",
        "DP1 DiaSource reliability summary")

    # Buckets
    for lo, hi, label in [(0, 0.1, "0.0-0.1"), (0.1, 0.5, "0.1-0.5"),
                          (0.5, 0.9, "0.5-0.9"), (0.9, 1.01, "0.9-1.0")]:
        run(svc,
            f"SELECT COUNT(*) AS n FROM dp1.DiaSource "
            f"WHERE reliability >= {lo} AND reliability < {hi}",
            f"reliability in [{label}]")

    # High-reliability SN-candidate objects
    run(svc,
        "SELECT TOP 10 d.diaObjectId, d.ra, d.dec, d.nDiaSources, "
        "AVG(s.reliability) AS mean_rel "
        "FROM dp1.DiaObject d "
        "JOIN dp1.DiaSource s ON d.diaObjectId = s.diaObjectId "
        "WHERE d.nDiaSources BETWEEN 10 AND 40 "
        "AND d.ra BETWEEN 52 AND 54 AND d.dec BETWEEN -28.5 AND -27.5 "
        "GROUP BY d.diaObjectId, d.ra, d.dec, d.nDiaSources "
        "HAVING AVG(s.reliability) >= 0.5 "
        "ORDER BY AVG(s.reliability) DESC",
        "ECDFS SN-candidates with mean reliability >= 0.5")

    return 0


if __name__ == "__main__":
    sys.exit(main())
