#!/usr/bin/env python3
"""End-to-end smoke for ORACLE wired into the rsp_dp1 pipeline.

Calls `_load_local_experts()` + `_score_one_epoch_serial()` on a single
cached DP1 LC. Prints the projection columns the pipeline would write
to gold for `oracle_lsst`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure ORACLE finds its weights
os.environ.setdefault(
    "DEBASS_ORACLE_WEIGHTS",
    "artifacts/local_experts/oracle/models/ELAsTiCC-lite/revived-star-159",
)

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from debass_meta.features.detection import normalize_lightcurve
from debass_meta.ingest.rsp_dp1 import (
    LOCAL_EXPERT_KEYS,
    _init_experts,
    _score_one_epoch_serial,
)


def main() -> None:
    print(f"LOCAL_EXPERT_KEYS = {LOCAL_EXPERT_KEYS}")
    experts = _init_experts()
    print(f"Loaded experts: {sorted(experts)}")
    for k, e in experts.items():
        avail = getattr(e, "_available", "?")
        print(f"  {k}: avail={avail}")
    print()

    # Pick the cached n=181 LC (highest-detection SN we have)
    lc_file = "data/lightcurves/dp1/611255759837069401.parquet"
    df = pd.read_parquet(lc_file)
    norm = normalize_lightcurve(df.sort_values("midpointMjdTai").to_dict("records"))
    print(f"Normalized {len(norm)} detections from {lc_file}")
    if not norm:
        return
    last_mjd = max(d["mjd"] for d in norm)
    epoch_jd = last_mjd + 2400000.5

    cols = _score_one_epoch_serial(
        experts, "611255759837069401", norm, epoch_jd, len(norm),
        skip_keys=("supernnova",),  # batch path skipped in real pipeline
    )

    # Print everything related to oracle
    print("\n--- oracle_lsst projection columns ---")
    for k, v in sorted(cols.items()):
        if "oracle" in k:
            print(f"  {k} = {v}")

    # And salt3 (we just fixed it)
    print("\n--- salt3_chi2 projection columns ---")
    for k, v in sorted(cols.items()):
        if "salt3" in k:
            print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
