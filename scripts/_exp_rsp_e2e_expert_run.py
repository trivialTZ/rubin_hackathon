#!/usr/bin/env python3
"""End-to-end: pull a DP1 lightcurve → run lc_features_bv → report prediction."""
from __future__ import annotations
import os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pyvo, requests
import numpy as np


def load_token(repo_root):
    tok = os.environ.get("RSP_TOKEN")
    if tok: return tok
    for line in (repo_root / ".env").read_text().splitlines():
        s = line.strip()
        if s.startswith("RSP_TOKEN="):
            return s.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit("no token")


def tap(url, token):
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    return pyvo.dal.TAPService(url, session=s)


def pull_dp1_lightcurve(svc, dia_object_id):
    tbl = svc.search(
        f"SELECT band, midpointMjdTai, psfFlux, psfFluxErr, "
        f"snr, reliability "
        f"FROM dp1.DiaSource WHERE diaObjectId = {dia_object_id} "
        f"ORDER BY midpointMjdTai"
    ).to_table()
    # Convert to list[dict] schema that local experts expect
    detections = []
    for r in tbl:
        if r["psfFlux"] is None or r["psfFluxErr"] is None:
            continue
        detections.append({
            "mjd": float(r["midpointMjdTai"]),
            "band": str(r["band"]).strip(),
            "flux": float(r["psfFlux"]),
            "fluxerr": float(r["psfFluxErr"]),
            "snr": float(r["snr"]) if r["snr"] is not None else None,
            "reliability": float(r["reliability"]) if r["reliability"] is not None else None,
        })
    return detections


def main():
    # Pick a DP1 object with reasonable light-curve shape
    # Previously tested: 611254935203348736 (ECDFS, n=40, r/i/z, MJD 60623-60653)
    target = 611254935203348736

    svc = tap("https://data.lsst.cloud/api/tap", load_token(REPO))
    print(f"Pulling DP1 lightcurve for {target}...")
    lc = pull_dp1_lightcurve(svc, target)
    print(f"Got {len(lc)} detections")
    print(f"First 2: {lc[:2]}")

    # Band + SNR sanity
    by_band = {}
    for d in lc:
        by_band.setdefault(d["band"], []).append(d)
    for b, dets in sorted(by_band.items()):
        snrs = [d["snr"] for d in dets if d["snr"] is not None]
        fluxes = [d["flux"] for d in dets]
        print(f"  band {b}: {len(dets)} dets, snr mean {np.mean(snrs) if snrs else float('nan'):.2f}, "
              f"flux range [{min(fluxes):.1f}, {max(fluxes):.1f}]")

    # Instantiate lc_features_bv
    print("\n=== Run lc_features_bv ===")
    try:
        from debass_meta.experts.local.lc_features import LcFeaturesExpert
    except ImportError as e:
        print(f"Import failed: {e}")
        return 1
    expert = LcFeaturesExpert()
    print(f"Available: {expert._available}")
    if not expert._available:
        print("Expert not loaded — need trained head at default path.")
        print(f"Path: {expert.model_path}")
        return 1

    # Score at last-detection epoch
    last_mjd = max(d["mjd"] for d in lc)
    last_jd = last_mjd + 2400000.5
    print(f"Scoring at epoch_jd={last_jd} (last MJD {last_mjd:.2f})")

    out = expert.predict_epoch(str(target), lc, last_jd)
    print(f"\nExpert output:")
    print(f"  available: {out.available}")
    print(f"  class_probabilities: {out.class_probabilities}")
    print(f"  raw: {out.raw_output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
