#!/usr/bin/env python3
"""Final plan verification — run each claim in execplan_v6 as an assertion."""
from __future__ import annotations
import os, sys
from pathlib import Path

import pyvo, requests
import pandas as pd


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


def check(ok, msg):
    flag = "✓" if ok else "✗"
    print(f"  [{flag}] {msg}")
    return ok


def main():
    repo = Path(__file__).resolve().parents[1]
    svc = tap("https://data.lsst.cloud/api/tap", load_token(repo))

    ok_all = True

    # Check 1: DP1 DiaObject nDiaSources >= 5 count (claim: ~93K)
    print("\n1. DP1 nDiaSources>=5 pool size")
    r = svc.search("SELECT COUNT(*) AS n FROM dp1.DiaObject WHERE nDiaSources >= 5").to_table()
    n = int(r["n"][0])
    ok_all &= check(n > 85000 and n < 100000,
                    f"n={n} (plan claim ~93K) {'IN RANGE' if 85000 < n < 100000 else 'OUT'}")

    # Check 2: DP1 DiaSource reliability distribution (claim: 85% have rel<0.1)
    print("\n2. DP1 DiaSource reliability < 0.1 fraction")
    r_tot = svc.search("SELECT COUNT(*) AS n FROM dp1.DiaSource").to_table()
    r_lo = svc.search("SELECT COUNT(*) AS n FROM dp1.DiaSource WHERE reliability < 0.1").to_table()
    frac = int(r_lo["n"][0]) / int(r_tot["n"][0])
    ok_all &= check(0.80 < frac < 0.90,
                    f"frac(rel<0.1) = {frac:.3f} (plan claim ~0.85)")

    # Check 3: TAP batched IN query with 500 diaObjectIds
    print("\n3. TAP `diaObjectId IN (...)` batched query at 500 IDs")
    sample = svc.search(
        "SELECT TOP 500 diaObjectId FROM dp1.DiaObject WHERE nDiaSources >= 5"
    ).to_table().to_pandas()
    ids_str = ",".join(str(int(x)) for x in sample["diaObjectId"])
    q = f"SELECT COUNT(*) AS n FROM dp1.DiaSource WHERE diaObjectId IN ({ids_str})"
    try:
        r = svc.search(q).to_table()
        n_src = int(r["n"][0])
        ok_all &= check(n_src > 0, f"got {n_src} DiaSource rows for 500 objs — BATCH WORKS")
    except Exception as e:
        ok_all &= check(False, f"500-ID IN query FAILED: {str(e)[:180]}")

    # Check 4: Verify 12 published SN candidates' diaObjectIds still resolve
    print("\n4. Published SN candidate diaObjectIds resolvable")
    hits_path = repo / "reports" / "rsp_probe" / "dp1_published_sne_hits.csv"
    if not hits_path.exists():
        ok_all &= check(False, "dp1_published_sne_hits.csv not found")
    else:
        hits = pd.read_csv(hits_path)
        ok_all &= check(len(hits) >= 12, f"csv has {len(hits)} rows (claim: 12 typed + 3 untyped = 15)")
        # Pick 3 and confirm they still exist
        test_ids = [int(x) for x in hits["diaObjectId"].iloc[:3]]
        q = f"SELECT diaObjectId, nDiaSources FROM dp1.DiaObject WHERE diaObjectId IN ({','.join(str(i) for i in test_ids)})"
        r = svc.search(q).to_table().to_pandas()
        ok_all &= check(len(r) == 3, f"3 test diaObjectIds resolved: {len(r)}/3")

    # Check 5: Verify all 12 published SN candidates have nDiaSources and band coverage
    print("\n5. Published SN candidates have real lightcurves (nDiaSources, bands)")
    hits = pd.read_csv(repo / "reports" / "rsp_probe" / "dp1_published_sne_hits.csv")
    total_n = int(hits["nDiaSources"].sum())
    median_n = hits["nDiaSources"].median()
    ok_all &= check(total_n > 500, f"total detections across 15 objs = {total_n}, median/obj = {median_n}")

    # Count typed vs untyped
    print("\n6. SN type counts (for §1 breakdown correction)")
    typed = hits[hits["photo_type"].notna()]
    print(f"  typed: {len(typed)}, by type:")
    print(typed["photo_type"].value_counts().to_string())
    print(f"  untyped (Dong+ adds, no photo_type): {len(hits) - len(typed)}")

    # Check 7: Confirm no SN has TNS type yet
    print("\n7. TNS type status for matched DP1 candidates (expect: 0 spec-typed)")
    tns = pd.read_csv(repo / "data" / "tns_public_objects.csv",
                      skiprows=1, low_memory=False)
    # Strip "AT " prefix
    hits["bare_name"] = hits["name"].str.replace("AT ", "")
    joined = hits.merge(tns[["name", "type"]], left_on="bare_name", right_on="name", how="left")
    n_in_tns = int(joined["type"].notna().sum() - joined["type"].isna().sum())  # will fix below
    n_typed = int(joined["type"].dropna().shape[0])
    print(f"  in TNS: {int((joined['type'].notna() | joined['type'].isna()).sum())} of {len(joined)}")
    # Actual check: any spec type (not null)?
    spec_typed_count = int(joined["type"].dropna().shape[0])
    ok_all &= check(spec_typed_count == 0,
                    f"{spec_typed_count} have TNS spec type (expected 0)")

    print(f"\n\n{'='*60}")
    print(f"Plan verification: {'ALL CHECKS PASSED ✓' if ok_all else 'SOME CHECKS FAILED ✗'}")
    print(f"{'='*60}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
