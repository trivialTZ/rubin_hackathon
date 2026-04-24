"""Merge LSST truth rows into the main truth table → object_truth_v5b.parquet."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--existing", default="data/truth/object_truth.parquet")
    ap.add_argument("--lsst", default="data/truth_candidates/object_truth_lsst.parquet")
    ap.add_argument("--output", default="data/truth/object_truth_v5b.parquet")
    args = ap.parse_args()

    ztf = pd.read_parquet(args.existing)
    lsst = pd.read_parquet(args.lsst)
    ztf["object_id"] = ztf["object_id"].astype(str)
    lsst["object_id"] = lsst["object_id"].astype(str)

    # Align types: ZTF parquet has str-typed redshift / consensus cols; LSST wrote floats.
    # Cast LSST to match so pyarrow concat schemas are compatible.
    str_cols = ["redshift", "tns_redshift", "tns_ra", "tns_dec",
                "consensus_experts", "consensus_n_agree", "consensus_n_total",
                "tns_has_spectra"]
    for c in str_cols:
        if c in lsst.columns:
            lsst[c] = lsst[c].apply(lambda v: None if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v))
        if c in ztf.columns:
            ztf[c] = ztf[c].apply(lambda v: None if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v))

    print(f"ZTF truth rows: {len(ztf)}")
    print(f"LSST truth rows: {len(lsst)}")

    # Sanity: no duplicate object_id across files
    overlap = set(ztf["object_id"]) & set(lsst["object_id"])
    assert not overlap, f"ZTF/LSST object_id overlap: {len(overlap)} ids (first 5: {list(overlap)[:5]})"

    out = pd.concat([ztf, lsst], ignore_index=True, sort=False)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)

    print(f"\nmerged {len(out)} rows -> {args.output}")
    print(f"  ZTF: {len(ztf)}, LSST: {len(lsst)}, total: {len(out)}")


if __name__ == "__main__":
    main()
