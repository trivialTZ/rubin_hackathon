#!/usr/bin/env python3
"""Extend alt-truth probe: hunt for SN-specific catalogs + OGLE variables.

Tests VizieR tables that publish SN catalogs and stellar variable catalogs
with known coverage of DP1's southern footprint.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path

import pandas as pd
import pyvo, requests
from astropy import units as u
from astropy.table import Table
from astroquery.xmatch import XMatch
from astroquery.vizier import Vizier


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


def xmatch_cat(dp1_df, cat, radius_arcsec=2.0, label=None):
    label = label or cat
    print(f"\n=== X-Match DP1 × {label} at {radius_arcsec}\" ===")
    tab = Table.from_pandas(dp1_df[["diaObjectId", "ra", "dec", "nDiaSources", "dp1_field"]])
    try:
        t0 = time.time()
        hits = XMatch.query(cat1=tab, cat2=cat, max_distance=radius_arcsec * u.arcsec,
                            colRA1="ra", colDec1="dec")
        dt = time.time() - t0
        print(f"  got {len(hits)} hits in {dt:.1f}s")
        return hits.to_pandas() if len(hits) > 0 else None
    except Exception as e:
        print(f"  ERR: {type(e).__name__}: {str(e)[:220]}")
        return None


def main():
    repo = Path(__file__).resolve().parents[1]
    svc = tap_rsp(load_token(repo))

    out_dir = repo / "reports" / "rsp_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = out_dir / "dp1_sample_by_field.csv"
    if not sample_path.exists():
        print("DP1 sample not found — run alt_truth_probe first to create it")
        return 1
    dp1 = pd.read_csv(sample_path)
    print(f"Loaded DP1 sample: {len(dp1)} positions across {dp1['dp1_field'].nunique()} fields")

    # SN-specific catalogs in VizieR
    # B/sn — Asiago Supernova Catalog (comprehensive, all-sky)
    xmatch_cat(dp1, "vizier:B/sn/sncat", radius_arcsec=3.0,
               label="Asiago SN catalog (B/sn)")

    # OGLE-IV SMC variables — correct ID
    # Known IDs: II/358 = OGLE-IV LMC RR Lyrae, II/359 = OGLE-IV SMC RR Lyrae,
    # VI/158 = OGLE-III SMC eclipsing binaries, J/AcA/68/89 = OGLE-IV SMC cepheids
    for cat_id, label in [
        ("vizier:II/355/rrlyrsmc",   "OGLE-IV SMC RR Lyrae (II/355)"),
        ("vizier:II/358/smc",        "OGLE-IV SMC Cepheids (II/358)"),
        ("vizier:II/356/smceclvar",  "OGLE-IV SMC eclipsing (II/356)"),
        ("vizier:J/AcA/67/297/smclpv", "OGLE-IV SMC LPVs (J/AcA/67/297)"),
        ("vizier:J/AcA/63/21/smc",   "OGLE-IV SMC 2013 (J/AcA/63/21)"),
    ]:
        xmatch_cat(dp1, cat_id, radius_arcsec=1.5, label=label)

    # Gaia DR3 Variables — already tested; now narrow by class
    gaia_var_path = out_dir / "dp1_gaia_var_hits.csv"
    if gaia_var_path.exists():
        gv = pd.read_csv(gaia_var_path)
        print(f"\n=== Gaia DR3 variable classes (from prior probe) ===")
        class_col = [c for c in gv.columns if "Class" in c or "type" in c.lower()]
        print(f"  Available class columns: {class_col}")
        for c in class_col:
            print(f"\n  {c}:")
            print(gv[c].value_counts().head(12))

    # DES-SN catalogs
    # Look for: J/AJ/154/78 (DES 3y cosmology sample), J/MNRAS/499/5641 (DES-SN spec),
    #         III/255/des (DES DR2 catalogs)
    for cat_id, label in [
        ("vizier:J/AJ/164/202/desbright", "DES-SN Y5 brightSample"),
        ("vizier:J/MNRAS/499/5641", "DES-SN spec (MNRAS 499)"),
        ("vizier:J/AJ/162/25", "DES-SN host-z catalog"),
    ]:
        xmatch_cat(dp1, cat_id, radius_arcsec=3.0, label=label)

    # Gaia DR3 QSO candidates (large)
    xmatch_cat(dp1, "vizier:I/354/qsocand", radius_arcsec=1.0,
               label="Gaia DR3 QSO candidates (I/354)")

    # Milliquas compilation of QSOs (most comprehensive)
    xmatch_cat(dp1, "vizier:VII/294/catalog", radius_arcsec=2.0,
               label="Milliquas QSO catalog (VII/294)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
