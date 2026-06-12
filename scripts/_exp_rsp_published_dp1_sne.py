#!/usr/bin/env python3
"""Crossmatch published DP1 SN candidates against DP1 DiaObject to get diaObjectIds.

Published DP1 SN sources:
  1. arXiv:2507.22864 — 11 photometrically-typed extragalactic transients
  2. arXiv:2603.00262 — AT 2024ahzi, SN IIP (photometric), host-z 0.211
  3. arXiv:2507.22156 — DECam difference imaging (same DP1 region, adds context)

All classifications are photometric. No spec-confirmed SN yet.
"""
from __future__ import annotations
import os, sys
from pathlib import Path

import pandas as pd
import pyvo, requests
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u


def load_token(repo_root):
    tok = os.environ.get("RSP_TOKEN")
    if tok: return tok
    for line in (repo_root / ".env").read_text().splitlines():
        s = line.strip()
        if s.startswith("RSP_TOKEN="):
            return s.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit("no token")


def tap_rsp(token):
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    return pyvo.dal.TAPService("https://data.lsst.cloud/api/tap", session=s)


# DP1 transient candidates with positions in literature (2026-04-24)
# Types and confidence from Superphot+ (Freeburn+) unless otherwise noted.
# TNS spec-type column is "(untyped)" for all: no DP1 SN is spec-confirmed.
PUBLISHED = [
    # (name, ra_str, dec_str, photo_type, photo_conf, prev_known, field, source_paper)
    # -- Freeburn+ 2507.22864 Table 4 (photometric via Superphot+) --
    ("AT 2024aigs", "03:56:53.2",  "-49:06:18.1",  "IIn",   0.132, "No",  "EDFS",  "2507.22864"),
    ("AT 2024aigl", "03:59:24.2",  "-48:46:50.5",  "Ia",    0.814, "No",  "EDFS",  "2507.22864"),
    ("AT 2024aigh", "03:57:17.8",  "-48:22:08.3",  "SLSN-I",0.056, "No",  "EDFS",  "2507.22864"),
    ("AT 2024aigv", "03:32:13.8",  "-28:28:14.4",  "Ia",    0.984, "No",  "ECDFS", "2507.22864"),  # >95%
    ("AT 2024ahyy", "03:31:34.2",  "-28:24:45.5",  "IIn",   0.133, "YSE", "ECDFS", "2507.22864"),
    ("AT 2024ahzc", "03:31:21.2",  "-28:16:47.7",  "IIn",   0.059, "YSE", "ECDFS", "2507.22864"),
    ("AT 2024aigt", "03:33:41.4",  "-28:13:24.8",  "II",    0.726, "No",  "ECDFS", "2507.22864"),
    ("AT 2024aigw", "03:30:55.6",  "-27:51:58.9",  "II",    0.954, "No",  "ECDFS", "2507.22864"),  # >95%
    ("AT 2024aigg", "03:32:29.9",  "-27:44:23.3",  "II",    0.647, "No",  "ECDFS", "2507.22864"),  # host z=0.076 in TNS
    ("AT 2024aigj", "03:32:51.0",  "-27:40:52.6",  "Ibc",   0.172, "No",  "ECDFS", "2507.22864"),
    ("AT 2024aaux", "02:34:22.8",  "+07:12:52.7",  "IIn",   0.058, "ZTF", "LELF",  "2507.22864"),
    # -- de Soto+ 2603.00262 --
    ("AT 2024ahzi", "03:53:20.41", "-48:45:01.09", "IIP",   None,  "No",  "EDFS",  "2603.00262"),  # paper claims "high confidence"
    # -- Dong+ 2507.22156 additions (TNS-reported but NOT in Freeburn+; no photometric type) --
    ("AT 2024ahsx", "03:33:28.1",  "-28:12:53.0",  None,    None,  "YSE", "ECDFS", "2507.22156"),
    ("AT 2024ahwk", "03:29:50.9",  "-28:13:05.0",  None,    None,  "YSE", "ECDFS", "2507.22156"),
    ("AT 2024ahyq", "03:31:37.7",  "-28:20:03.0",  None,    None,  "YSE", "ECDFS", "2507.22156"),
    ("AT 2024aigk", "03:55:31.6",  "-48:27:40.0",  None,    None,  "YSE", "EDFS",  "2507.22156"),
]


def parse_to_deg(row):
    c = SkyCoord(row["ra_str"], row["dec_str"], unit=(u.hourangle, u.deg))
    return c.ra.deg, c.dec.deg


def main():
    repo = Path(__file__).resolve().parents[1]
    svc = tap_rsp(load_token(repo))

    # Build the truth table
    tbl = pd.DataFrame(
        PUBLISHED,
        columns=["name", "ra_str", "dec_str", "photo_type", "photo_conf",
                 "prev_known", "field", "source_paper"],
    )
    tbl[["ra_deg", "dec_deg"]] = tbl.apply(lambda r: pd.Series(parse_to_deg(r)), axis=1)

    print("=== DP1 transient candidates (positions in degrees) ===")
    print(tbl[["name", "ra_deg", "dec_deg", "photo_type", "photo_conf",
               "prev_known", "field", "source_paper"]].to_string())

    # For each, cone-search DP1 DiaObject at 3" radius
    print("\n=== Crossmatch vs dp1.DiaObject ===")
    hits = []
    for _, row in tbl.iterrows():
        ra = row["ra_deg"]
        dec = row["dec_deg"]
        adql = (
            f"SELECT diaObjectId, ra, dec, nDiaSources "
            f"FROM dp1.DiaObject "
            f"WHERE CONTAINS(POINT('ICRS', ra, dec), "
            f"CIRCLE('ICRS', {ra}, {dec}, 0.000833)) = 1"  # 3" = 3/3600 deg ≈ 0.000833
        )
        try:
            r = svc.search(adql).to_table().to_pandas()
            if len(r) > 0:
                r_sorted = r.copy()
                # Rank by separation
                c = SkyCoord(ra=r_sorted["ra"].values, dec=r_sorted["dec"].values, unit="deg")
                tgt = SkyCoord(ra=ra, dec=dec, unit="deg")
                r_sorted["sep_arcsec"] = c.separation(tgt).to(u.arcsec).value
                r_sorted = r_sorted.sort_values("sep_arcsec").iloc[0:1]
                row_out = {
                    "name": row["name"], "photo_type": row["photo_type"], "photo_conf": row["photo_conf"],
                    "field": row["field"], "source": row["source_paper"],
                    "ra_paper": ra, "dec_paper": dec,
                    "diaObjectId": int(r_sorted["diaObjectId"].iloc[0]),
                    "ra_dp1": float(r_sorted["ra"].iloc[0]),
                    "dec_dp1": float(r_sorted["dec"].iloc[0]),
                    "nDiaSources": int(r_sorted["nDiaSources"].iloc[0]),
                    "sep_arcsec": float(r_sorted["sep_arcsec"].iloc[0]),
                }
                hits.append(row_out)
                print(f"  ✓ {row['name']:<14}  dia={row_out['diaObjectId']:<20} "
                      f"n={row_out['nDiaSources']:<4} sep={row_out['sep_arcsec']:.2f}\"")
            else:
                print(f"  ✗ {row['name']:<14}  NO MATCH at 3\" in dp1.DiaObject "
                      f"(ra={ra:.4f}, dec={dec:.4f})")
        except Exception as e:
            print(f"  ! {row['name']:<14}  ERR: {type(e).__name__}: {str(e)[:120]}")

    out = repo / "reports" / "rsp_probe" / "dp1_published_sne_hits.csv"
    hits_df = pd.DataFrame(hits)
    hits_df.to_csv(out, index=False)
    print(f"\n\nSaved {len(hits_df)} matches to {out}")

    # Summary
    if len(hits_df) > 0:
        print(f"\n=== Summary ===")
        print(f"  Matched: {len(hits_df)}/{len(tbl)}")
        print(f"\n  By photo_type (Freeburn+/de Soto+):")
        print(hits_df["photo_type"].value_counts(dropna=False))
        # Confidence tiers
        hi = hits_df[hits_df["photo_conf"] >= 0.95]
        mid = hits_df[(hits_df["photo_conf"] >= 0.60) & (hits_df["photo_conf"] < 0.95)]
        lo = hits_df[(hits_df["photo_conf"].notna()) & (hits_df["photo_conf"] < 0.60)]
        nope = hits_df[hits_df["photo_conf"].isna()]
        print(f"\n  Confidence tiers:")
        print(f"    >95% (strong):  {len(hi)}  {list(hi['name'])}")
        print(f"    60-95% (medium): {len(mid)}  {list(mid['name'])}")
        print(f"    <60% (weak):    {len(lo)}  {list(lo['name'])}")
        print(f"    no type (trans only): {len(nope)}  {list(nope['name'])}")
        print(f"\n  By DP1 field:")
        print(hits_df["field"].value_counts())
        print(f"\n  nDiaSources distribution:")
        print(hits_df["nDiaSources"].describe())
        print(f"\n  Sep: min={hits_df['sep_arcsec'].min():.2f}, "
              f"max={hits_df['sep_arcsec'].max():.2f}, "
              f"mean={hits_df['sep_arcsec'].mean():.2f}\"")

    return 0


if __name__ == "__main__":
    sys.exit(main())
