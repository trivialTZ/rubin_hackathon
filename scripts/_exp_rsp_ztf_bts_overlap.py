#!/usr/bin/env python3
"""ZTF-BTS × DP1 overlap probe.

ZTF-BTS (Bright Transient Survey) is the single largest source of spec-typed
SNe in the Rubin/ZTF era (2018+). Public CSV download.
"""
from __future__ import annotations
import io, os, sys
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


def tap_rsp(token):
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    return pyvo.dal.TAPService("https://data.lsst.cloud/api/tap", session=s)


def fetch_bts_csv(cache_path: Path) -> pd.DataFrame:
    """Download the ZTF-BTS Explorer catalog."""
    if cache_path.exists():
        print(f"Using cached BTS CSV: {cache_path}")
        return pd.read_csv(cache_path)
    print("Downloading ZTF-BTS from sites.astro.caltech.edu/ztf/bts/explorer.php ...")
    url = "https://sites.astro.caltech.edu/ztf/bts/explorer.php?format=csv"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(r.text)
    df = pd.read_csv(io.StringIO(r.text))
    print(f"Downloaded {len(df)} BTS rows → {cache_path}")
    return df


def main():
    repo = Path(__file__).resolve().parents[1]
    svc = tap_rsp(load_token(repo))

    # 1. Download BTS
    bts_path = repo / "data" / "ztf_bts.csv"
    bts = fetch_bts_csv(bts_path)
    print(f"\nBTS schema: {list(bts.columns)[:15]}")

    # Find ra/dec cols
    ra_col = next((c for c in bts.columns if c.lower() in ("ra", "radeg", "ra_deg")), None)
    dec_col = next((c for c in bts.columns if c.lower() in ("dec", "decdeg", "dec_deg", "decl")), None)
    print(f"Detected ra_col={ra_col} dec_col={dec_col}")
    if ra_col is None or dec_col is None:
        print("Could not detect RA/Dec columns. First row:")
        print(bts.head(1).T)
        return 1

    # Parse RA/Dec if they're sexagesimal
    def _parse_coord(s, is_ra):
        try:
            return float(s)
        except (TypeError, ValueError):
            return None

    # Convert sexagesimal if present
    try:
        bts["ra_deg"] = pd.to_numeric(bts[ra_col], errors="coerce")
        bts["dec_deg"] = pd.to_numeric(bts[dec_col], errors="coerce")
        if bts["ra_deg"].isna().all():
            # Sexagesimal format — convert
            from astropy.coordinates import Angle
            c = SkyCoord(bts[ra_col].astype(str).tolist(),
                         bts[dec_col].astype(str).tolist(),
                         unit=(u.hourangle, u.deg))
            bts["ra_deg"] = c.ra.deg
            bts["dec_deg"] = c.dec.deg
    except Exception as e:
        print(f"Coordinate parse failed: {e}")
        return 1

    # Filter to DP1 footprint region (Dec -72 to +10 covers all DP1 fields)
    bts_footprint = bts[(bts["dec_deg"] >= -75) & (bts["dec_deg"] <= 10)].dropna(subset=["ra_deg", "dec_deg"]).copy()
    print(f"\nBTS after Dec cut [-75, 10]: {len(bts_footprint)} rows")

    # Type column
    type_col = next((c for c in bts.columns if c.lower() in ("type", "class", "classification")), None)
    print(f"Type column: {type_col}")
    if type_col:
        print(f"\nBTS type distribution (DP1-footprint cut):")
        print(bts_footprint[type_col].value_counts().head(15))

    # 2. Pull DP1 positions — either reuse cached 3K sample or pull fresh 2K/field
    sample_path = repo / "reports" / "rsp_probe" / "dp1_sample_by_field.csv"
    if sample_path.exists():
        dp1 = pd.read_csv(sample_path)
        print(f"\nUsing cached DP1 sample: {len(dp1)} positions across {dp1['dp1_field'].nunique()} fields")
    else:
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
            q = (f"SELECT TOP 2000 diaObjectId, ra, dec, nDiaSources "
                 f"FROM dp1.DiaObject WHERE nDiaSources >= 5 "
                 f"AND ra BETWEEN {ra0} AND {ra1} AND dec BETWEEN {d0} AND {d1}")
            df = svc.search(q).to_table().to_pandas()
            df["dp1_field"] = name
            frames.append(df)
            print(f"  {name}: {len(df)} DP1 positions")
        dp1 = pd.concat(frames, ignore_index=True)
        print(f"Total DP1: {len(dp1)}")

    # 3. Crossmatch at 2" radius
    dp1c = SkyCoord(ra=dp1["ra"].values, dec=dp1["dec"].values, unit="deg")
    btsc = SkyCoord(ra=bts_footprint["ra_deg"].values,
                    dec=bts_footprint["dec_deg"].values, unit="deg")
    idx, sep, _ = dp1c.match_to_catalog_sky(btsc)
    sep_as = sep.to(u.arcsec).value

    for radius in [2.0, 5.0, 10.0]:
        mask = sep_as <= radius
        hits = dp1[mask].reset_index(drop=True).copy()
        matched = bts_footprint.iloc[idx[mask]].reset_index(drop=True)
        print(f"\n=== ZTF-BTS × DP1 at {radius}\" ===")
        print(f"  {mask.sum()} hits of {len(dp1)} DP1 positions")
        if mask.sum() > 0:
            if type_col:
                hits["bts_type"] = matched[type_col].values
                print("  Types:")
                print(hits["bts_type"].value_counts().head(10))
            for c in ["IAUID", "ztfid", "ZTFID", "name", "IAU", "iauname", "redshift", "peakmag", "peakt"]:
                if c in matched.columns:
                    hits[c.lower()] = matched[c].values
            hits["sep_arcsec"] = sep_as[mask]
            hits["dp1_field"] = dp1[mask]["dp1_field"].values
            print(f"  by DP1 field:")
            print(hits.groupby("dp1_field").size())
            if radius == 2.0:
                out = repo / "reports" / "rsp_probe" / "dp1_ztfbts_hits.csv"
                hits.to_csv(out, index=False)
                print(f"\n  Saved {out}")
                print(hits.head(20).to_string())
            elif radius == 10.0:
                break

    return 0


if __name__ == "__main__":
    sys.exit(main())
