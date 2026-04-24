#!/usr/bin/env python3
"""Build DP1 truth table: Gaia DR3 + SIMBAD + Gaia variables + 15 published SNe.

Input : DP1 DiaObject pool (n>=5) pulled from RSP TAP.
Output: data/truth/dp1_truth.parquet with one row per DP1 diaObjectId.

Truth columns per object (final schema):
  diaObjectId, ra, dec, nDiaSources,
  is_gaia_known_star  (bool, Gaia parallax or pm > 3σ),
  gaia_source_id      (str or null),
  simbad_main_type    (str or null),
  gaia_var_class      (str or null),
  is_known_not_sn     (bool; gaia star OR SIMBAD class implies non-SN),
  is_known_variable   (bool),
  is_published_sn     (bool, from literature crossmatch),
  published_sn_type   (str or null, Superphot+ photometric class),
  published_sn_conf   (float or null, Superphot+ confidence),
  published_sn_name   (str or null, IAU AT name),
  label               (ternary: "snia" | "nonIa_snlike" | "other"; maps to v5d schema).

Usage
-----
    # Small feasibility run: just the 15 published SNe + field-sampled negatives
    python scripts/build_dp1_truth.py --n-pool 3000 --out data/truth/dp1_truth.parquet

    # Larger production pool (recommended 10K-50K per plan §9):
    python scripts/build_dp1_truth.py --n-pool 50000 --batch-per-field 10000

Verified 2026-04-24 on 3K sample: 89% Gaia, 4% SIMBAD, 1% Gaia-var, 15/16 published
matches. Plan: §1, §2b.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from debass_meta.access.rubin_rsp import RSPClient

REPO = Path(__file__).resolve().parents[1]

# DP1 sky fields (measured footprint, plan §2b).
_FIELDS = [
    ("ECDFS",        52.0, 54.0,   -28.5, -27.5),
    ("SMC",           4.0,  8.0,   -73.0, -71.0),
    ("RA59_Dec-49",  58.0, 60.0,   -49.5, -48.5),
    ("RA95_Dec-25",  94.0, 96.0,   -25.5, -24.5),
    ("RA106_Dec-11", 105.0, 107.0, -11.0, -10.0),
    ("RA38_Dec+7",   37.0, 39.0,     6.0,   8.0),
]


def pool_dp1(client: RSPClient, n_per_field: int, min_n_det: int) -> pd.DataFrame:
    """Pull DP1 DiaObject pool using per-field caps (avoids TAP timeouts)."""
    frames: list[pd.DataFrame] = []
    for name, ra0, ra1, d0, d1 in _FIELDS:
        t0 = time.time()
        df = client.pool_diaobjects(
            min_n_det=min_n_det,
            limit=n_per_field,
            ra_range=(ra0, ra1),
            dec_range=(d0, d1),
        )
        df["dp1_field"] = name
        print(f"  {name}: {len(df):,} objects in {time.time()-t0:.1f}s")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def xmatch_vizier(
    dp1: pd.DataFrame, vizier_cat: str, *, radius_arcsec: float, label: str,
) -> pd.DataFrame | None:
    """Bulk CDS X-Match DP1 coordinates against a VizieR catalog."""
    from astropy.table import Table
    from astropy import units as u
    from astroquery.xmatch import XMatch

    t0 = time.time()
    tab = Table.from_pandas(dp1[["diaObjectId", "ra", "dec", "nDiaSources"]])
    try:
        hits = XMatch.query(
            cat1=tab, cat2=vizier_cat,
            max_distance=radius_arcsec * u.arcsec,
            colRA1="ra", colDec1="dec",
        )
    except Exception as e:
        print(f"  [{label}] X-Match error: {type(e).__name__}: {str(e)[:150]}")
        return None
    dt = time.time() - t0
    print(f"  [{label}] {len(hits):,} hits in {dt:.1f}s")
    return hits.to_pandas() if len(hits) else None


def build_truth(
    n_per_field: int, min_n_det: int, published_sne_path: Path, out_path: Path,
) -> None:
    client = RSPClient()

    print(f"Stage 1: pool DP1 DiaObject (n>={min_n_det}, per-field cap {n_per_field})")
    dp1 = pool_dp1(client, n_per_field, min_n_det)
    print(f"  total pool: {len(dp1):,} objects")

    truth = dp1[["diaObjectId", "ra", "dec", "nDiaSources", "dp1_field"]].copy()
    truth["diaObjectId"] = truth["diaObjectId"].astype("int64")

    print("\nStage 2: Gaia DR3 crossmatch (1 arcsec)")
    gaia = xmatch_vizier(truth, "vizier:I/355/gaiadr3", radius_arcsec=1.0, label="Gaia DR3")
    if gaia is not None and len(gaia):
        # Threshold: significant parallax OR significant proper motion
        plx_sig = (gaia["Plx"] / gaia["e_Plx"]).abs() > 3.0
        pm_sig = (gaia["PM"] / (gaia["e_pmRA"] + gaia["e_pmDE"] + 1e-9)) > 3.0
        gaia["_star_sig"] = (plx_sig | pm_sig).astype(bool)
        star_ids = set(gaia.loc[gaia["_star_sig"], "diaObjectId"].astype("int64"))
        gaia_src_map = (
            gaia.sort_values("angDist")
            .drop_duplicates("diaObjectId", keep="first")
            .set_index("diaObjectId")["DR3Name"].to_dict()
        )
    else:
        star_ids, gaia_src_map = set(), {}
    truth["is_gaia_known_star"] = truth["diaObjectId"].isin(star_ids)
    truth["gaia_source_id"] = truth["diaObjectId"].map(gaia_src_map)

    print("\nStage 3: SIMBAD crossmatch (2 arcsec)")
    simbad = xmatch_vizier(truth, "simbad", radius_arcsec=2.0, label="SIMBAD")
    if simbad is not None and len(simbad):
        simbad_map = (
            simbad.sort_values("angDist")
            .drop_duplicates("diaObjectId", keep="first")
            .set_index("diaObjectId")["main_type"].to_dict()
        )
    else:
        simbad_map = {}
    truth["simbad_main_type"] = truth["diaObjectId"].map(simbad_map)

    print("\nStage 4: Gaia DR3 variables (1 arcsec)")
    gvar = xmatch_vizier(truth, "vizier:I/358/vclassre", radius_arcsec=1.0, label="Gaia-var")
    if gvar is not None and len(gvar):
        gvar_map = (
            gvar.sort_values("angDist")
            .drop_duplicates("diaObjectId", keep="first")
            .set_index("diaObjectId")["Class"].to_dict()
        )
    else:
        gvar_map = {}
    truth["gaia_var_class"] = truth["diaObjectId"].map(gvar_map)

    # ---- Derive labels ----
    # "known non-SN": Gaia star OR SIMBAD class in {Star, QSO, WD, RRLyrae, ...}
    simbad_nonsn = {
        "Star", "Galaxy", "QSO", "WhiteDwarf_Candidate", "RRLyrae",
        "EclBin", "LPV", "CataclyV", "AGN", "WhiteDwarf", "Variable*",
        "HighPM*", "EmissionG", "LINER", "Seyfert1", "Seyfert2", "BLLac",
    }
    truth["is_known_not_sn"] = (
        truth["is_gaia_known_star"]
        | truth["simbad_main_type"].isin(simbad_nonsn)
    )
    truth["is_known_variable"] = (
        truth["gaia_var_class"].notna()
        | truth["simbad_main_type"].isin({"RRLyrae", "EclBin", "LPV", "CataclyV", "Variable*"})
    )

    print("\nStage 5: merge 15 published DP1 SN candidates")
    pub = pd.read_csv(published_sne_path)
    pub["diaObjectId"] = pub["diaObjectId"].astype("int64")
    pub_map = pub.set_index("diaObjectId")[
        ["name", "photo_type", "photo_conf"]
    ].to_dict(orient="index")
    truth["is_published_sn"] = truth["diaObjectId"].isin(set(pub["diaObjectId"]))
    truth["published_sn_name"] = truth["diaObjectId"].map(
        lambda x: pub_map.get(int(x), {}).get("name"))
    truth["published_sn_type"] = truth["diaObjectId"].map(
        lambda x: pub_map.get(int(x), {}).get("photo_type"))
    truth["published_sn_conf"] = truth["diaObjectId"].map(
        lambda x: pub_map.get(int(x), {}).get("photo_conf"))

    # Force-include any published SNe missing from the sampled pool
    missing = pub[~pub["diaObjectId"].isin(truth["diaObjectId"])]
    if len(missing):
        add = missing.rename(columns={
            "ra_dp1": "ra", "dec_dp1": "dec", "name": "published_sn_name",
            "photo_type": "published_sn_type", "photo_conf": "published_sn_conf",
        })
        add["dp1_field"] = add.get("field", "unknown")
        add["is_gaia_known_star"] = False
        add["gaia_source_id"] = None
        add["simbad_main_type"] = None
        add["gaia_var_class"] = None
        add["is_known_not_sn"] = False
        add["is_known_variable"] = False
        add["is_published_sn"] = True
        keep = [c for c in truth.columns if c in add.columns]
        truth = pd.concat([truth, add[keep]], ignore_index=True)
        print(f"  force-added {len(missing)} published SNe missing from field sample")

    # Ternary label (v5d schema).
    def _label(row: pd.Series) -> str | None:
        if row.get("is_published_sn"):
            t = str(row.get("published_sn_type") or "").lower()
            if "ia" in t and "non" not in t:
                return "snia"
            if t:
                return "nonIa_snlike"
            return "nonIa_snlike"  # untyped published transients still treated as SN-like
        if row.get("is_known_not_sn") or row.get("is_known_variable"):
            return "other"
        return None  # unlabeled
    truth["label"] = truth.apply(_label, axis=1)

    # ---- Summary ----
    n = len(truth)
    print(f"\n=== Truth table summary (N={n:,}) ===")
    print(f"  Gaia-known stars    : {int(truth['is_gaia_known_star'].sum()):>6,}  "
          f"({truth['is_gaia_known_star'].mean()*100:.1f}%)")
    print(f"  SIMBAD typed        : {int(truth['simbad_main_type'].notna().sum()):>6,}  "
          f"({truth['simbad_main_type'].notna().mean()*100:.1f}%)")
    print(f"  Gaia variables      : {int(truth['gaia_var_class'].notna().sum()):>6,}  "
          f"({truth['gaia_var_class'].notna().mean()*100:.1f}%)")
    print(f"  Known non-SN        : {int(truth['is_known_not_sn'].sum()):>6,}")
    print(f"  Published SN matches: {int(truth['is_published_sn'].sum()):>6,}")
    print(f"  Final label coverage: "
          f"snia={int((truth['label']=='snia').sum())}, "
          f"nonIa={int((truth['label']=='nonIa_snlike').sum())}, "
          f"other={int((truth['label']=='other').sum())}, "
          f"unlabeled={int(truth['label'].isna().sum())}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    truth.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path} ({n:,} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n-pool", type=int, default=3000,
                    help="Total pool approx size (split across 6 fields). Default 3000.")
    ap.add_argument("--min-n-det", type=int, default=5)
    ap.add_argument("--published",
                    default=str(REPO / "reports/rsp_probe/dp1_published_sne_hits.csv"),
                    help="Path to 15-SN published-candidates CSV.")
    ap.add_argument("--out", default=str(REPO / "data/truth/dp1_truth.parquet"))
    args = ap.parse_args()

    per_field = max(args.n_pool // len(_FIELDS), 1)
    build_truth(per_field, args.min_n_det, Path(args.published), Path(args.out))


if __name__ == "__main__":
    main()
