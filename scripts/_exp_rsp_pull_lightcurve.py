#!/usr/bin/env python3
"""Experiment: pull one real DP1 lightcurve via TAP. Verify columns + sanity."""
from __future__ import annotations
import os, sys
from pathlib import Path
import pyvo, requests


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


def main():
    repo_root = Path(__file__).resolve().parents[1]
    svc = tap("https://data.lsst.cloud/api/tap", load_token(repo_root))

    # Step A: find an SN-candidate (n_det in [10, 40]) in ECDFS
    print("=== Step A: find SN-candidate in ECDFS ===")
    sample = svc.search(
        "SELECT TOP 5 diaObjectId, ra, dec, nDiaSources "
        "FROM dp1.DiaObject "
        "WHERE ra BETWEEN 52 AND 54 AND dec BETWEEN -28.5 AND -27.5 "
        "AND nDiaSources BETWEEN 10 AND 40 "
        "ORDER BY nDiaSources DESC"
    ).to_table()
    print(sample)
    if len(sample) == 0:
        print("No SN-candidate found in ECDFS n∈[10,40]")
        return 1

    target_id = int(sample[0]["diaObjectId"])
    target_ra = float(sample[0]["ra"])
    target_dec = float(sample[0]["dec"])
    target_n = int(sample[0]["nDiaSources"])
    print(f"\nTarget: {target_id}  ra={target_ra:.4f} dec={target_dec:.4f} n={target_n}")

    # Step B: pull all DiaSource rows for this object
    print(f"\n=== Step B: pull lightcurve for diaObjectId={target_id} ===")
    lc = svc.search(
        f"SELECT diaSourceId, diaObjectId, band, midpointMjdTai, "
        f"ra, dec, snr, psfFlux, psfFluxErr, scienceFlux, scienceFluxErr, "
        f"reliability, extendedness, visit, detector "
        f"FROM dp1.DiaSource "
        f"WHERE diaObjectId = {target_id} "
        f"ORDER BY midpointMjdTai"
    ).to_table()
    print(f"Pulled {len(lc)} DiaSource rows")
    print(lc)

    # Step C: sanity checks
    print(f"\n=== Step C: sanity checks ===")
    mjd_min = min(lc["midpointMjdTai"])
    mjd_max = max(lc["midpointMjdTai"])
    print(f"  MJD range: {mjd_min:.3f} - {mjd_max:.3f}  (span {mjd_max - mjd_min:.2f} days)")
    print(f"  Bands present: {sorted(set(lc['band']))}")
    print(f"  SNR range: {min(lc['snr']):.1f} - {max(lc['snr']):.1f}")
    print(f"  PSF flux range: {min(lc['psfFlux']):.2f} - {max(lc['psfFlux']):.2f}")
    print(f"  Reliability range: {min(lc['reliability']):.3f} - {max(lc['reliability']):.3f}")
    print(f"  Extendedness range: {min(lc['extendedness']):.3f} - {max(lc['extendedness']):.3f}")

    # Step D: save for downstream experiments
    out_dir = repo_root / "reports" / "rsp_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    lc_path = out_dir / f"dp1_lc_{target_id}.ecsv"
    lc.write(str(lc_path), overwrite=True, format="ascii.ecsv")
    print(f"\nSaved lightcurve to {lc_path}")
    print(f"\nKEY: diaObjectId={target_id} ra={target_ra} dec={target_dec} n={target_n}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
