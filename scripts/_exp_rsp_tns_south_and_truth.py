#!/usr/bin/env python3
"""Diagnose TNS coverage + check if DP1 has truth-like catalogs."""
from __future__ import annotations
import os, sys
from pathlib import Path

import pandas as pd
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


def main():
    repo_root = Path(__file__).resolve().parents[1]

    # 1. TNS declination distribution
    print("=== TNS declination distribution ===")
    tns = pd.read_csv(repo_root / "data" / "tns_public_objects.csv",
                      skiprows=1, low_memory=False)
    tns = tns.dropna(subset=["declination", "type"])
    tns["dec_bin"] = pd.cut(tns["declination"].astype(float),
                             bins=[-90, -70, -50, -30, -10, 10, 30, 50, 70, 90])
    print("All TNS typed objects by declination bin:")
    print(tns.groupby("dec_bin").size())

    print("\nSN Ia only by declination bin:")
    sn_ia = tns[tns["type"] == "SN Ia"]
    print(sn_ia.groupby("dec_bin").size())

    # DP1 Dec range was -72 to -25 for our sample
    print(f"\nDP1 footprint: Dec -72 to -25")
    print(f"TNS objects (any type) with Dec ∈ [-75, -25]:")
    south_tns = tns[(tns["declination"] >= -75) & (tns["declination"] <= -25)]
    print(f"  {len(south_tns)} total, type breakdown:")
    print(south_tns["type"].value_counts().head(15))

    # 2. Check DP1 for truth-like tables
    print("\n=== DP1 schema tables with 'truth' / 'match' / 'sim' ===")
    svc = tap("https://data.lsst.cloud/api/tap", load_token(repo_root))
    tables = svc.search(
        "SELECT table_name FROM tap_schema.tables "
        "WHERE schema_name = 'dp1' OR schema_name = 'dp02_dc2_catalogs'"
    ).to_table()
    for row in tables:
        nm = row["table_name"].lower()
        if any(x in nm for x in ["truth", "match", "sim", "spec"]):
            print(f"  {row['table_name']}")

    # 3. ForcedSourceOnDiaObject — does this give us host context?
    print("\n=== Check dp1.Object for host-galaxy spec-z / type columns ===")
    result = svc.search(
        "SELECT column_name, description FROM tap_schema.columns "
        "WHERE table_name = 'dp1.Object' "
        "AND (column_name LIKE '%redshift%' OR column_name LIKE '%spec%' "
        "OR column_name LIKE '%type%' OR column_name LIKE '%class%')"
    ).to_table()
    print(result)

    # 4. Southward TNS — match to DP1 more carefully at wider radius
    print("\n=== DP1 ECDFS sample at 5\" TNS radius ===")
    ecdfs = svc.search(
        "SELECT TOP 5000 diaObjectId, ra, dec, nDiaSources FROM dp1.DiaObject "
        "WHERE nDiaSources >= 5 AND ra BETWEEN 52 AND 54 AND dec BETWEEN -29 AND -27"
    ).to_table().to_pandas()
    print(f"ECDFS sample: {len(ecdfs)} objects")

    from astropy.coordinates import SkyCoord
    from astropy import units as u
    if len(ecdfs) > 0 and len(south_tns) > 0:
        dp1c = SkyCoord(ra=ecdfs["ra"].values, dec=ecdfs["dec"].values, unit="deg")
        tnsc = SkyCoord(ra=south_tns["ra"].values, dec=south_tns["declination"].values, unit="deg")
        idx, sep, _ = dp1c.match_to_catalog_sky(tnsc)
        sep_as = sep.to(u.arcsec).value
        for radius in [2.0, 5.0, 10.0]:
            mask = sep_as <= radius
            n_match = mask.sum()
            matched = south_tns.iloc[idx[mask]]
            n_typed = matched["type"].notna().sum()
            print(f"  ECDFS match at {radius}\": {n_match} total, {n_typed} spec-typed")
            if n_match > 0 and n_match <= 20:
                print(matched[["name", "type", "redshift", "discoverymag", "source_group"]].head())

    return 0


if __name__ == "__main__":
    sys.exit(main())
