#!/usr/bin/env python3
"""Smoke-test the salt3_chi2 LSST band fix on a cached DP1 lightcurve.

Loads one of the 14 cached DP1 SN parquets, normalizes detections (so each
det has survey='LSST'), runs salt3_chi2.predict_epoch at the latest epoch,
and prints what it returns. Expected outcome:
    available=True, class_probabilities={'Ia': p, 'II': 1-p}
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from debass_meta.experts.local.salt3_fit import Salt3Chi2Expert, _band_name
from debass_meta.features.detection import normalize_lightcurve


def main() -> None:
    files = sorted(glob.glob("data/lightcurves/dp1/*.parquet"))
    if not files:
        raise SystemExit("No cached DP1 LCs found")
    print(f"Found {len(files)} cached DP1 LCs; testing on each.")
    expert = Salt3Chi2Expert()
    print(f"Salt3 available: {expert._available}")

    n_ok = 0
    n_no_probs = 0
    n_too_few = 0
    for f in files:
        df = pd.read_parquet(f)
        oid = str(df["diaObjectId"].iloc[0])
        raw = df.sort_values("midpointMjdTai").to_dict("records")
        norm = normalize_lightcurve(raw)
        if not norm:
            print(f"{oid}: empty after normalize")
            continue
        # Verify band mapping for the first detection
        sample_band = _band_name(norm[0])
        last_mjd = max(d.get("mjd", 0) for d in norm)
        epoch_jd = last_mjd + 2400000.5
        out = expert.predict_epoch(oid, norm, epoch_jd)
        cps = out.class_probabilities
        if cps:
            n_ok += 1
            print(f"{oid}: n_det={len(norm)}, sample_band={sample_band}, "
                  f"avail={out.available}, p_Ia={cps.get('Ia',0):.3f}, "
                  f"Δχ²={out.raw_output.get('delta_chi2','?')}")
        else:
            reason = out.raw_output.get("reason", "?")
            if "detections" in reason:
                n_too_few += 1
            else:
                n_no_probs += 1
            ia_err = out.raw_output.get("ia_fit", {}).get("error", "?")
            print(f"{oid}: n_det={len(norm)}, sample_band={sample_band}, "
                  f"avail={out.available}, NO PROBS  reason={reason}")
            if ia_err != "?":
                print(f"    ia_fit error: {ia_err[:200]}")
    print(f"\nSUMMARY: {n_ok}/{len(files)} fits succeeded; "
          f"{n_no_probs} no-probs; {n_too_few} too-few-detections.")


if __name__ == "__main__":
    main()
