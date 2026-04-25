#!/usr/bin/env python3
"""Smoke-test ORACLE1_ELAsTiCC_lite on cached DP1 lightcurves.

Verifies:
  1. Model loads with our local weights
  2. DP1 LC parquet → ORACLE-shaped astropy.table.Table conversion works
  3. score() returns sensible per-class probabilities

If this works, we can wire ORACLE into rsp_dp1.py as the 5th local expert.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table

# Use our local downloaded weights, not the package default
ORACLE_LITE_DIR = "artifacts/local_experts/oracle/models/ELAsTiCC-lite/revived-star-159"
os.environ.setdefault("DEBASS_ORACLE_REPO", "artifacts/local_experts/oracle")
os.environ.setdefault("DEBASS_ORACLE_WEIGHTS", ORACLE_LITE_DIR)


# SNANA FLUXCAL has zero point ZP=27.5 mag in count units; DP1 psfFlux is in
# nanojansky (AB ZP = 31.4). Multiplicative conversion:
#   FLUXCAL = 10**((27.5 - 31.4) / 2.5) * psfFlux_nJy ≈ 0.02754 * psfFlux_nJy
NJY_TO_FLUXCAL = 10 ** ((27.5 - 31.4) / 2.5)


def dp1_to_oracle_table(df_lc: pd.DataFrame) -> Table:
    """DP1 DiaSource parquet → ORACLE-shaped astropy Table.

    ORACLE/ELAsTiCC2 expects:
        MJD, FLUXCAL (SNANA counts at ZP=27.5), FLUXCALERR,
        BAND (single letter, capital Y for y-band), PHOTFLAG (4096 = detection).

    DP1 cached parquets store psfFlux in nanojansky and use lowercase 'y',
    so we convert flux scale and uppercase the y band. PHOTFLAG = 4096 for
    every row (cached parquets are detection-only).
    """
    df_lc = df_lc.sort_values("midpointMjdTai").reset_index(drop=True)
    n = len(df_lc)
    bands = df_lc["band"].astype(str).str.replace("y", "Y", regex=False).values
    return Table({
        "MJD":         df_lc["midpointMjdTai"].astype(float).values,
        "FLUXCAL":     (df_lc["psfFlux"].astype(float) * NJY_TO_FLUXCAL).values,
        "FLUXCALERR":  (df_lc["psfFluxErr"].astype(float) * NJY_TO_FLUXCAL).values,
        "BAND":        bands,
        "PHOTFLAG":    np.full(n, 4096, dtype=np.int64),
    })


def main() -> None:
    files = sorted(glob.glob("data/lightcurves/dp1/*.parquet"))
    if not files:
        raise SystemExit("No cached DP1 LCs found")
    print(f"Found {len(files)} cached DP1 LCs.")

    from oracle.pretrained.ELAsTiCC import ORACLE1_ELAsTiCC_lite
    print(f"Loading ORACLE1_ELAsTiCC_lite from {ORACLE_LITE_DIR}")
    model = ORACLE1_ELAsTiCC_lite(model_dir=ORACLE_LITE_DIR)
    print()

    n_ok = 0
    n_too_few = 0
    n_err = 0
    for f in files:
        df = pd.read_parquet(f)
        oid = str(df["diaObjectId"].iloc[0])
        if len(df) < 3:
            print(f"{oid}: only {len(df)} dets, skip")
            n_too_few += 1
            continue
        try:
            tbl = dp1_to_oracle_table(df)
            scores = model.score(tbl)
            preds = model.predict(tbl)
            # `scores` is dict {depth_level: DataFrame}.  The leaf level (max depth)
            # contains the 19 fine-grained classes; collapse them to ternary later.
            leaf_depth = max(scores)
            leaf = scores[leaf_depth]
            top_class = preds[leaf_depth]
            top_prob = float(leaf.iloc[0].max())
            # Sum over SN-like leaves
            sn_like = [c for c in leaf.columns
                       if any(s in c for s in ["SNIa", "SN91bg", "SNIax", "SNII", "SNIbc",
                                               "SLSN", "IIn", "IIb", "TDE"])]
            ia_only = [c for c in leaf.columns if c == "SNIa"]
            p_sn = float(leaf[sn_like].iloc[0].sum()) if sn_like else 0.0
            p_ia = float(leaf[ia_only].iloc[0].sum()) if ia_only else 0.0
            print(f"{oid}: n={len(df):>3}, top={top_class:<12} (p={top_prob:.3f}), "
                  f"p(SN-like)={p_sn:.3f}, p(Ia)={p_ia:.3f}")
            n_ok += 1
        except Exception as exc:
            print(f"{oid}: ERROR {type(exc).__name__}: {exc}")
            n_err += 1

    print(f"\nSUMMARY: ok={n_ok}, too-few={n_too_few}, errors={n_err}")


if __name__ == "__main__":
    main()
