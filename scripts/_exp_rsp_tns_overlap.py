#!/usr/bin/env python3
"""Probe: crossmatch a large DP1 DiaObject sample against the local TNS CSV.

This answers: how many spec-typed objects are in DP1? Across which sky fields?
By which SN types?
"""
from __future__ import annotations
import os, sys
from pathlib import Path

import pandas as pd
import pyvo, requests
from astropy.coordinates import SkyCoord
from astropy import units as u


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


def pull_dp1_positions(svc, n=20000):
    """Pull n DiaObject positions with nDiaSources >= 5, spread across sky."""
    # Use ORDER BY RAND()-equivalent or just pull a lot and sample
    print(f"Pulling up to {n} DP1 DiaObject positions (nDiaSources >= 5)...")
    tbl = svc.search(
        f"SELECT TOP {n} diaObjectId, ra, dec, nDiaSources "
        f"FROM dp1.DiaObject WHERE nDiaSources >= 5"
    ).to_table()
    df = tbl.to_pandas()
    print(f"Got {len(df)} positions. RA range: {df['ra'].min():.2f} - {df['ra'].max():.2f}, "
          f"Dec range: {df['dec'].min():.2f} - {df['dec'].max():.2f}")
    return df


def load_tns(repo_root):
    path = repo_root / "data" / "tns_public_objects.csv"
    # First line is timestamp, real header on line 2
    print(f"Loading TNS from {path}...")
    tns = pd.read_csv(path, skiprows=1, low_memory=False)
    tns = tns.dropna(subset=["ra", "declination"])
    print(f"TNS: {len(tns)} rows with ra/dec. Type counts:")
    tc = tns["type"].fillna("(no type)").value_counts().head(12)
    print(tc)
    return tns


def crossmatch(dp1_df, tns_df, radius_arcsec=2.0):
    """Find nearest TNS neighbor for each DP1 object. Return hits."""
    print(f"\nCrossmatching {len(dp1_df)} DP1 positions → {len(tns_df)} TNS positions at {radius_arcsec}\"...")
    dp1_coord = SkyCoord(ra=dp1_df["ra"].values, dec=dp1_df["dec"].values, unit="deg")
    tns_coord = SkyCoord(ra=tns_df["ra"].values, dec=tns_df["declination"].values, unit="deg")
    idx, sep, _ = dp1_coord.match_to_catalog_sky(tns_coord)
    sep_arcsec = sep.to(u.arcsec).value
    mask = sep_arcsec <= radius_arcsec

    hits = dp1_df[mask].copy().reset_index(drop=True)
    matched_tns = tns_df.iloc[idx[mask]].reset_index(drop=True)
    hits["tns_name"] = matched_tns["name"].values
    hits["tns_prefix"] = matched_tns["name_prefix"].values
    hits["tns_type"] = matched_tns["type"].fillna("(untyped)").values
    hits["tns_redshift"] = matched_tns["redshift"].values
    hits["sep_arcsec"] = sep_arcsec[mask]
    return hits


def main():
    repo_root = Path(__file__).resolve().parents[1]
    svc = tap("https://data.lsst.cloud/api/tap", load_token(repo_root))

    dp1 = pull_dp1_positions(svc, n=20000)
    tns = load_tns(repo_root)
    hits = crossmatch(dp1, tns, radius_arcsec=2.0)

    print(f"\n=== Crossmatch result ===")
    print(f"Total DP1 positions queried: {len(dp1)}")
    print(f"Matched at 2\": {len(hits)}")
    if len(hits) > 0:
        print(f"\nBy type:")
        print(hits["tns_type"].value_counts())
        print(f"\nBy prefix (SN/AT/etc):")
        print(hits["tns_prefix"].value_counts())
        print(f"\nFirst 20 matches:")
        cols = ["diaObjectId", "ra", "dec", "nDiaSources", "tns_name", "tns_prefix", "tns_type", "tns_redshift", "sep_arcsec"]
        print(hits[cols].head(20).to_string())

        # Bin by rough sky field
        def field_of(ra, dec):
            if -28.5 <= dec <= -27.5 and 52 <= ra <= 54: return "ECDFS"
            if -73 <= dec <= -72 and 4 <= ra <= 8:       return "SMC"
            if 6 <= dec <= 8 and 37 <= ra <= 39:         return "RA38_Dec7"
            if -25.5 <= dec <= -24.5 and 94 <= ra <= 96: return "RA95_Dec-25"
            if -11 <= dec <= -10 and 105 <= ra <= 107:   return "RA106_Dec-11"
            if -49.5 <= dec <= -48.5 and 58 <= ra <= 60: return "RA59_Dec-49"
            return "other"
        hits["field"] = [field_of(r, d) for r, d in zip(hits["ra"], hits["dec"])]
        print(f"\nBy field:")
        print(hits.groupby("field")["tns_type"].value_counts())

    out = repo_root / "reports" / "rsp_probe" / "dp1_tns_overlap.csv"
    hits.to_csv(out, index=False)
    print(f"\nSaved matches to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
