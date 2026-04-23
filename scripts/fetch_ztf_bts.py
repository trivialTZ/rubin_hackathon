#!/usr/bin/env python3
"""Fetch the ZTF Bright Transient Survey (BTS) Explorer CSV.

The BTS Explorer at https://sites.astro.caltech.edu/ztf/bts/explorer.php
exposes a CSV endpoint for the ~11,500 spectroscopically-confirmed transients
with ZTF IDs, TNS names, types (Ia, II, Ib/c, IIn, SLSN-I/II, TDE, nova,
LBV, ILRT), redshifts, and host properties.

This is DEBASS Truth Tier 1.5 (real-sky, spec-confirmed, fine subtypes),
sitting between TNS spectroscopic and broker consensus.

Output: data/truth/ztf_bts.parquet

Usage:
    python3 scripts/fetch_ztf_bts.py
    python3 scripts/fetch_ztf_bts.py --output data/truth/ztf_bts.parquet
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from debass_meta.access.tns import map_tns_type_to_ternary

_BTS_CSV_URL = "https://sites.astro.caltech.edu/ztf/bts/explorer.php?format=csv"
_BTS_HTML_FALLBACK_URL = "https://sites.astro.caltech.edu/ztf/bts/explorer.php"


def _download_csv(url: str, timeout: int = 60) -> pd.DataFrame:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    body = r.text
    # The endpoint sometimes wraps the CSV in HTML with a <pre> block.
    # Try direct parse first; if that fails, strip HTML.
    try:
        return pd.read_csv(io.StringIO(body))
    except pd.errors.ParserError:
        lower = body.lower()
        if "<pre>" in lower:
            start = lower.index("<pre>") + len("<pre>")
            end = lower.index("</pre>", start)
            return pd.read_csv(io.StringIO(body[start:end].strip()))
        raise


def _normalise_bts(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalise column names and map subtype → DEBASS ternary."""
    col_map = {
        # Canonical BTS headers: ZTFID,IAUID,RA,Dec,peakt,peakfilt,peakmag,
        # peakabs,duration,rise,fade,type,redshift,b,A_V
        "ZTFID": "object_id",
        "ztfid": "object_id",
        "IAUID": "tns_name",
        "iauid": "tns_name",
        "Type": "bts_type",
        "type": "bts_type",
        "RA": "ra",
        "ra": "ra",
        "Dec": "dec",
        "dec": "dec",
        "peakt": "peak_mjd",
        "peakmag": "peak_mag",
        "peakabs": "peak_abs_mag",
        "peakfilt": "peak_filter",
        "redshift": "redshift",
        "z": "redshift",
        "duration": "duration",
        "rise": "rise",
        "fade": "fade",
        "b": "galactic_b",
        "A_V": "galactic_av",
    }
    df = df.rename(columns={c: col_map[c] for c in df.columns if c in col_map})
    keep_cols = [c for c in df.columns if c in {
        "object_id", "tns_name", "bts_type", "ra", "dec",
        "peak_mjd", "peak_mag", "peak_abs_mag", "peak_filter",
        "redshift", "duration", "rise", "fade", "galactic_b", "galactic_av",
    }]
    df = df[keep_cols].copy()
    df = df[df["object_id"].notna()].copy()
    df["object_id"] = df["object_id"].astype(str).str.strip()

    def _map(ty: str | None) -> str | None:
        if not ty:
            return None
        s = str(ty).strip()
        # BTS uses "SN Ia", "SN II", etc.; feed through the TNS mapper
        return map_tns_type_to_ternary("SN " + s) if not s.startswith("SN") else map_tns_type_to_ternary(s)

    df["ternary"] = df["bts_type"].map(_map)
    df["label_source"] = "ztf_bts"
    df["label_quality"] = "spectroscopic"  # BTS is spec-confirmed by definition
    return df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch ZTF BTS Explorer CSV → parquet")
    parser.add_argument("--output", default="data/truth/ztf_bts.parquet")
    parser.add_argument("--url", default=_BTS_CSV_URL)
    args = parser.parse_args()

    print(f"Fetching ZTF BTS CSV from {args.url}")
    try:
        df = _download_csv(args.url)
    except Exception as exc:
        print(f"  [error] primary URL failed: {exc}")
        print(f"  retrying with {_BTS_HTML_FALLBACK_URL}")
        df = _download_csv(_BTS_HTML_FALLBACK_URL)

    print(f"  Raw BTS rows: {len(df)}")
    df = _normalise_bts(df)
    print(f"  Normalised rows: {len(df)}, {df['ternary'].value_counts().to_dict()}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Wrote {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
