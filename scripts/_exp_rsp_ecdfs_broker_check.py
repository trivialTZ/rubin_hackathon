#!/usr/bin/env python3
"""Sample ECDFS-region DiaObjects + test broker coverage (Fink LSST) + TNS overlap.

This is the pivotal check: does ECDFS DP1 data have any SNe with broker scores
and/or TNS spec types? If yes, metaDEBASS can actually benefit from v6a ingest.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import pyvo
import requests


def load_token(repo_root):
    tok = os.environ.get("RSP_TOKEN")
    if tok: return tok
    for line in (repo_root / ".env").read_text().splitlines():
        s = line.strip()
        if s.startswith("RSP_TOKEN="):
            return s.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit("RSP_TOKEN not found")


def tap(url, token):
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    return pyvo.dal.TAPService(url, session=s)


def query_ecdfs_sample(svc, n=20):
    """Get n well-sampled DiaObjects in ECDFS (RA 52-54, Dec -28.5 to -27.5)."""
    adql = (
        f"SELECT TOP {n} diaObjectId, ra, dec, nDiaSources "
        "FROM dp1.DiaObject "
        "WHERE nDiaSources >= 20 "
        "AND ra BETWEEN 52 AND 54 "
        "AND dec BETWEEN -28.5 AND -27.5 "
        "ORDER BY nDiaSources DESC"
    )
    return svc.search(adql).to_table()


def check_fink_lsst(diaObjectId: int):
    """Query Fink LSST API for this diaObjectId (POST /api/v1/sources)."""
    try:
        r = requests.post(
            "https://api.lsst.fink-portal.org/api/v1/sources",
            json={"diaObjectId": str(diaObjectId), "output-format": "json"},
            timeout=20,
        )
        if r.status_code != 200:
            return {"ok": False, "status": r.status_code, "body": r.text[:120]}
        data = r.json()
        if not data:
            return {"ok": True, "found": False, "n_alerts": 0}
        rec0 = data[0] if isinstance(data, list) else data
        keys = sorted(list(rec0.keys()))[:20] if isinstance(rec0, dict) else []
        n = len(data) if isinstance(data, list) else 1
        # Check for classifier scores presence
        clf_fields = [k for k in keys if k.startswith("f:clf_")]
        return {
            "ok": True, "found": True, "n_alerts": n,
            "has_classifiers": bool(clf_fields),
            "clf_fields": clf_fields[:5],
        }
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:120]}"}


def check_tns_by_coord(ra, dec, radius_arcsec=2.0):
    """Free TNS crossmatch via cone search (no API key needed for basic search)."""
    # TNS public search is rate-limited. For now, just note we'd query.
    # Proper check uses TNS_API_KEY from .env.
    return {"note": f"cone ({ra:.4f}, {dec:.4f}, r={radius_arcsec}\") — deferred"}


def main():
    repo_root = Path(__file__).resolve().parents[1]
    svc = tap("https://data.lsst.cloud/api/tap", load_token(repo_root))

    print("=== ECDFS DP1 sample (ra 52-54, dec -28.5 to -27.5, n≥20 det) ===")
    sample = query_ecdfs_sample(svc, n=20)
    print(sample)

    results = []
    print("\n=== Fink LSST coverage ===")
    for row in sample:
        oid = int(row["diaObjectId"])
        ra = float(row["ra"])
        dec = float(row["dec"])
        n = int(row["nDiaSources"])
        fink = check_fink_lsst(oid)
        print(f"  {oid}  ra={ra:.3f} dec={dec:.3f} n={n:<4}  fink={fink}")
        results.append({
            "diaObjectId": oid, "ra": ra, "dec": dec, "nDiaSources": n,
            "fink_lsst": fink,
        })

    out = repo_root / "reports" / "rsp_probe" / "ecdfs_broker_coverage.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out}")

    # Summary
    n_ok = sum(1 for r in results if r["fink_lsst"].get("found") is True)
    n_empty = sum(1 for r in results if r["fink_lsst"].get("found") is False)
    n_err = sum(1 for r in results if not r["fink_lsst"].get("ok"))
    print(f"\nFink LSST summary: {n_ok} found, {n_empty} not-found, {n_err} error (of {len(results)})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
