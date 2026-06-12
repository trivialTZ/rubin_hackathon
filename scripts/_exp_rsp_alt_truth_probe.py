#!/usr/bin/env python3
"""Probe alternative truth sources for DP1 via CDS X-Match bulk crossmatch.

Catalogs tested (in order of science leverage):
  - SIMBAD (otype/main_id): astrophysical type (SN, AGN, variable, star, galaxy)
  - VizieR I/355/gaiadr3: Gaia DR3 main table (distance, G mag)
  - VizieR I/358/vclassre: Gaia DR3 variability classifier
  - VizieR II/379/smcogle4: OGLE-IV SMC variables (if in DP1-SMC field)
  - VizieR J/A+A/658/A78 etc.: DES-SN Y5 (searchable)

For each DP1 field we pull ~500 positions, upload to X-Match, get hits.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path

import pandas as pd
import pyvo, requests
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table
from astroquery.xmatch import XMatch


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


def pull_dp1_sample_per_field(svc, per_field_n=500):
    fields = [
        ("ECDFS",        52, 54,   -28.5, -27.5),
        ("SMC",          4,  8,    -73,   -71),
        ("RA59_Dec-49",  58, 60,   -49.5, -48.5),
        ("RA95_Dec-25",  94, 96,   -25.5, -24.5),
        ("RA106_Dec-11", 105, 107, -11,   -10),
        ("RA38_Dec+7",   37, 39,   6,     8),
    ]
    frames = []
    for name, ra0, ra1, d0, d1 in fields:
        q = (f"SELECT TOP {per_field_n} diaObjectId, ra, dec, nDiaSources "
             f"FROM dp1.DiaObject WHERE nDiaSources >= 5 "
             f"AND ra BETWEEN {ra0} AND {ra1} AND dec BETWEEN {d0} AND {d1}")
        try:
            df = svc.search(q).to_table().to_pandas()
            df["dp1_field"] = name
            frames.append(df)
            print(f"  {name}: {len(df)} objects")
        except Exception as e:
            print(f"  {name}: ERR {e}")
    return pd.concat(frames, ignore_index=True)


def xmatch_against(dp1_df, vizier_cat, radius_arcsec=2.0, label=None):
    """Bulk crossmatch DP1 df against a VizieR catalog via CDS X-Match."""
    label = label or vizier_cat
    print(f"\n=== X-Match DP1 × {label} at {radius_arcsec}\" ===")
    tab = Table.from_pandas(dp1_df[["diaObjectId", "ra", "dec", "nDiaSources", "dp1_field"]])
    try:
        t0 = time.time()
        hits = XMatch.query(
            cat1=tab,
            cat2=vizier_cat,
            max_distance=radius_arcsec * u.arcsec,
            colRA1="ra", colDec1="dec",
        )
        dt = time.time() - t0
        print(f"  got {len(hits)} hits in {dt:.1f}s")
        return hits.to_pandas() if len(hits) > 0 else None
    except Exception as e:
        print(f"  ERR: {type(e).__name__}: {str(e)[:250]}")
        return None


def main():
    repo = Path(__file__).resolve().parents[1]
    svc = tap_rsp(load_token(repo))

    print("=== Pull DP1 sample across 6 fields ===")
    dp1 = pull_dp1_sample_per_field(svc, per_field_n=500)
    print(f"Total DP1 sample: {len(dp1)}")

    out_dir = repo / "reports" / "rsp_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    dp1.to_csv(out_dir / "dp1_sample_by_field.csv", index=False)

    # SIMBAD (via CDS) — full typed catalog
    simbad = xmatch_against(dp1, "simbad", radius_arcsec=2.0, label="SIMBAD")
    if simbad is not None:
        print(f"\n  SIMBAD type distribution:")
        print(simbad["main_type"].value_counts().head(20))
        print(f"\n  SIMBAD hits by DP1 field:")
        print(simbad.groupby("dp1_field").size())
        simbad.to_csv(out_dir / "dp1_simbad_hits.csv", index=False)

    # Gaia DR3 main table — for stellar context / variable star flags
    gaia = xmatch_against(dp1, "vizier:I/355/gaiadr3", radius_arcsec=1.0,
                           label="Gaia DR3")
    if gaia is not None:
        print(f"  Gaia hits: {len(gaia)}, columns sample:",
              list(gaia.columns)[:15])
        gaia.to_csv(out_dir / "dp1_gaia_dr3_hits.csv", index=False)

    # OGLE-IV SMC variables — stellar var negative class in SMC field
    ogle = xmatch_against(dp1, "vizier:II/379/smcvar", radius_arcsec=2.0,
                           label="OGLE-IV SMC")
    if ogle is None:
        ogle = xmatch_against(dp1, "vizier:J/AcA/65/233/table3", radius_arcsec=2.0,
                               label="OGLE SMC alt")
    if ogle is not None:
        print(f"  OGLE SMC hits: {len(ogle)}")
        ogle.to_csv(out_dir / "dp1_ogle_smc_hits.csv", index=False)

    # Gaia DR3 variable classifier — variability type for stars
    gaia_var = xmatch_against(dp1, "vizier:I/358/vclassre", radius_arcsec=1.0,
                                label="Gaia DR3 variables")
    if gaia_var is not None:
        print(f"  Gaia variable hits: {len(gaia_var)}")
        if "BestClass" in gaia_var.columns:
            print("  Variability class:")
            print(gaia_var["BestClass"].value_counts().head(15))
        gaia_var.to_csv(out_dir / "dp1_gaia_var_hits.csv", index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
