#!/usr/bin/env python3
"""Catalogue-context labels for Rubin/LSST alert objects, independent of every broker.

metaDEBASS's LSST "other" labels used to come from ALeRCE stamp classes (weak) or Lasair Sherlock (context), and both
are also model inputs. This labels an object "other" only from external catalogues, via the CDS X-Match service:

  gaia_star      Gaia DR3 source within 1.0" with parallax or proper motion significant at > 5 sigma
  gaia_var:<c>   Gaia DR3 variability classification (I/358/vclassre) within 1.0" (any class, incl. AGN)
  simbad:<t>     SIMBAD main type within --simbad-radius (default 1.0") that is a star, a variable star or an AGN/QSO

Galaxy types are never used: supernovae sit on or near galaxies. Objects with a TNS spectroscopic type should be
labelled from TNS, not from this table; pass --exclude to leave them out.

Input : CSV or parquet with object_id, ra, dec (object_id kept as a string).
Output: parquet with object_id, ra, dec, gaia_star, gaia_source_id, gaia_var_class, simbad_main_type, simbad_sep,
        final_class_ternary ('other' or None), label_quality ('context' or None), label_source.

Usage:
  python scripts/build_lsst_catalog_context.py --objects data/lsst_candidates.csv --out data/truth/lsst_catalog_context.parquet
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

GAIA_CAT = "vizier:I/355/gaiadr3"
GAIA_VAR_CAT = "vizier:I/358/vclassre"
GAIA_SIGMA = 5.0

# SIMBAD main types (as returned by CDS X-Match) that mean "not a supernova". Galaxy types are deliberately absent.
SIMBAD_STELLAR = {
    "Star", "HighPM*", "PM*", "WhiteDwarf", "WhiteDwarf_Candidate", "Low-Mass*", "BrownD*", "RGB*", "HorBranch*",
    "Variable*", "RRLyrae", "EclBin", "LPV", "Mira", "Cepheid", "delSctV*", "gammaDorV*", "BYDraV*", "RSCVnV*",
    "CataclyV", "Nova", "DwarfNova", "Eruptive*", "Pulsating*", "YSO", "TTauri*", "Orion_V*", "EmLine*", "Be*",
    "Carbon*", "S*", "SB*", "**", "ChemPec*", "BlueStraggler", "SXPheV*", "RotV*", "Irregular_V*",
    "EllipVar", "Flare*", "BlueSG", "RedSG", "Supergiant", "Giant", "SubDwarf", "MainSequence*", "Evolved*",
    "PostAGB*", "AGB*", "LongPeriodV*_Candidate", "RRLyrae_Candidate", "EclBin_Candidate", "Cepheid_Candidate",
    "CataclyV_Candidate", "Variable*_Candidate", "Star_Candidate",
}
# Broad-line / unobscured AGN only: LINER and Seyfert 2 nuclei also host TDEs and nuclear SNe, so a transient
# there is not clearly AGN variability.
SIMBAD_AGN = {"QSO", "AGN", "Seyfert1", "BLLac", "Blazar",
              "QSO_Candidate", "AGN_Candidate", "Blazar_Candidate", "BLLac_Candidate"}
SIMBAD_NOT_SN = SIMBAD_STELLAR | SIMBAD_AGN


def xmatch(df: pd.DataFrame, cat: str, radius_arcsec: float, label: str, chunk: int = 20000) -> pd.DataFrame:
    """CDS X-Match of df[object_id, ra, dec] against cat; returns all hits with angDist (arcsec)."""
    from astropy import units as u
    from astropy.table import Table
    from astroquery.xmatch import XMatch

    out = []
    for i in range(0, len(df), chunk):
        part = df.iloc[i:i + chunk][["object_id", "ra", "dec"]]
        tab = Table.from_pandas(part)
        t0 = time.time()
        for attempt in range(3):
            try:
                hits = XMatch.query(cat1=tab, cat2=cat, max_distance=radius_arcsec * u.arcsec,
                                    colRA1="ra", colDec1="dec")
                break
            except Exception as exc:  # network hiccup: retry, then fail loudly
                if attempt == 2:
                    raise RuntimeError(f"[{label}] X-Match failed: {type(exc).__name__}: {exc}") from exc
                time.sleep(10 * (attempt + 1))
        print(f"  [{label}] rows {i}-{i + len(part)}: {len(hits):,} hits in {time.time() - t0:.1f}s")
        if len(hits):
            h = hits.to_pandas()
            h["object_id"] = h["object_id"].astype(str)
            out.append(h)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["object_id", "angDist"])


def nearest(h: pd.DataFrame) -> pd.DataFrame:
    return h.sort_values("angDist").drop_duplicates("object_id", keep="first").set_index("object_id")


def label_objects(obj: pd.DataFrame, simbad_radius: float = 1.0) -> pd.DataFrame:
    obj = obj[["object_id", "ra", "dec"]].copy()
    obj["object_id"] = obj["object_id"].astype(str)
    obj = obj.dropna(subset=["ra", "dec"]).drop_duplicates("object_id").reset_index(drop=True)

    g = xmatch(obj, GAIA_CAT, 1.0, "Gaia DR3")
    star_ids, gaia_src = set(), {}
    if len(g):
        plx_sig = (g["Plx"] / g["e_Plx"]).abs()
        pm_sig = np.hypot(g["pmRA"] / g["e_pmRA"], g["pmDE"] / g["e_pmDE"])
        g["_star"] = (plx_sig > GAIA_SIGMA) | (pm_sig > GAIA_SIGMA)
        gs = g[g["_star"]]
        star_ids = set(gs["object_id"])
        gaia_src = nearest(gs)["Source"].astype(str).to_dict() if "Source" in gs else {}
    gv = xmatch(obj, GAIA_VAR_CAT, 1.0, "Gaia DR3 variables")
    gvar = nearest(gv)["Class"].to_dict() if len(gv) else {}
    sb = xmatch(obj, "simbad", simbad_radius, "SIMBAD")
    sbn = nearest(sb) if len(sb) else pd.DataFrame(columns=["main_type", "angDist"])

    obj["gaia_star"] = obj["object_id"].isin(star_ids)
    obj["gaia_source_id"] = obj["object_id"].map(gaia_src)
    obj["gaia_var_class"] = obj["object_id"].map(gvar)
    obj["simbad_main_type"] = obj["object_id"].map(sbn["main_type"].to_dict())
    obj["simbad_sep"] = obj["object_id"].map(sbn["angDist"].to_dict())

    basis = pd.Series(None, index=obj.index, dtype=object)
    sim_hit = obj["simbad_main_type"].isin(SIMBAD_NOT_SN)
    basis[sim_hit] = "simbad:" + obj.loc[sim_hit, "simbad_main_type"].astype(str)
    var_hit = obj["gaia_var_class"].notna()
    basis[var_hit] = "gaia_var:" + obj.loc[var_hit, "gaia_var_class"].astype(str)
    basis[obj["gaia_star"]] = "gaia_star"
    obj["label_source"] = basis.map(lambda b: f"catalog:{b}" if isinstance(b, str) else None)
    has = obj["label_source"].notna()
    obj["final_class_ternary"] = np.where(has, "other", None)
    obj["label_quality"] = np.where(has, "context", None)
    return obj


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objects", required=True, help="CSV/parquet with object_id, ra, dec")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", default=None, help="CSV/parquet with object_id to leave out (e.g. TNS-typed objects)")
    ap.add_argument("--simbad-radius", type=float, default=1.0)
    args = ap.parse_args()

    read = lambda p: pd.read_parquet(p) if str(p).endswith(".parquet") else pd.read_csv(p, dtype={"object_id": str})  # noqa: E731
    obj = read(args.objects)
    obj["object_id"] = obj["object_id"].astype(str)
    if args.exclude:
        ex = set(read(args.exclude)["object_id"].astype(str))
        obj = obj[~obj["object_id"].isin(ex)]
    out = label_objects(obj, simbad_radius=args.simbad_radius)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"{len(out):,} objects; labelled other: {out['final_class_ternary'].notna().sum():,}")
    print(out["label_source"].str.split(":").str[1].value_counts().head(20).to_string())


if __name__ == "__main__":
    main()
