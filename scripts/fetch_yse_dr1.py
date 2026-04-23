#!/usr/bin/env python3
"""Fetch Young Supernova Experiment (YSE) DR1 from Zenodo.

YSE DR1 (Aleo+ 2023): 1,975 SNe observed Nov 2019–Dec 2021, PS1 griz + ZTF gr,
includes 1,048 spectroscopic Ia, 339 II, 96 Ib/c plus photometric classes.

This feeds DEBASS Truth Tier 1 (spec-confirmed) as a low-z cross-check on
ZTF BTS.  Size ≈ 5 MB total — fast to integrate.

Zenodo: https://zenodo.org/records/7317476
Paper:  arXiv 2211.07128

Output: data/truth/yse_dr1.parquet
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

_YSE_CLASSIFICATION_URL = (
    "https://zenodo.org/record/7317476/files/"
    "parsnip_results_for_ysedr1_table_A1_full_for_online.csv"
)


def _fetch_csv(url: str, timeout: int = 120) -> pd.DataFrame:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    # Zenodo schema (Aleo+ 2023 Table A1):
    #   Object, RA, Dec, Spec. Class, Redshift $z$, Prediction, Confidence,
    #   p_SNII, p_SNIa, p_SNIbc
    rename = {
        "Object": "tns_name",       # TNS name without AT/SN prefix
        "RA": "ra",
        "Dec": "dec",
        "Spec. Class": "spec_class",
        "Redshift, $z$": "redshift",
        "Redshift $z$": "redshift",
        "Prediction": "parsnip_class",
        "Confidence": "parsnip_confidence",
    }
    df = df.rename(columns={c: rename[c] for c in df.columns if c in rename})
    if "tns_name" not in df.columns:
        raise KeyError(f"Expected Object/tns_name column, got {list(df.columns)}")
    if "object_id" not in df.columns:
        df["object_id"] = df["tns_name"].astype(str)

    def _classify(s):
        raw = str(s).strip()
        if not raw or raw in ("unknown", "-", "None"):
            return None
        # Normalise "SNII" → "SN II", "SNIa-norm" → "SN Ia-norm", "Ia" → "SN Ia"
        if raw.startswith("SN") and not raw.startswith("SN "):
            raw = "SN " + raw[2:]
        elif not raw.startswith("SN") and raw not in ("TDE", "Nova", "AGN"):
            raw = "SN " + raw
        return map_tns_type_to_ternary(raw)

    df["ternary"] = df["spec_class"].map(_classify)
    df["label_source"] = "yse_dr1"
    df["label_quality"] = df["spec_class"].apply(
        lambda s: "spectroscopic"
        if s and str(s).strip() not in ("", "unknown", "-", "None")
        else "tns_untyped"
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch YSE DR1 classifications")
    parser.add_argument("--output", default="data/truth/yse_dr1.parquet")
    parser.add_argument("--url", default=_YSE_CLASSIFICATION_URL)
    args = parser.parse_args()

    print(f"Fetching YSE DR1 from {args.url}")
    try:
        df = _fetch_csv(args.url)
    except Exception as exc:
        print(f"  [error] download failed: {exc}")
        sys.exit(1)

    print(f"  Raw rows: {len(df)}")
    df = _normalise(df)
    print(f"  Ternary counts: {df['ternary'].value_counts(dropna=False).to_dict()}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Wrote {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
