#!/usr/bin/env python3
"""Build a self-contained HTML viewer comparing DEBaSS (DECam) and LSST (DP2).

Combines, for every object with both datasets:
  - DEBaSS SNANA forced photometry (griz)     -> data/debass_photometry/
  - LSST DP2 DiaSource detections (ugrizy)    -> reports/rsp_debass38/diasources.parquet
  - LSST DP2 deep-coadd cutouts (PNG, 20")    -> data/lsst_cutouts/

Both fluxes are put on a common nJy scale so they share ONE y-axis:
  LSST  psfFlux is already nJy (AB zeropoint 31.4)
  DEBaSS FLUXCAL uses SNANA's ZP 27.5  ->  nJy = FLUXCAL * 10**((31.4-27.5)/2.5)
That factor (36.31) was verified empirically against same-night paired epochs.

DP2 ships deep coadds only -- no per-epoch visit or difference images until full
DP2 (Oct-Dec 2026) -- so the LSST image panel is a static reference, not a time
series. DECam pixels are embargoed at NOIRLab until 2026-12-29.

Output: reports/rsp_debass38/debass_lsst_viewer.html
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RPT = REPO / "reports" / "rsp_debass38"
PHOT = REPO / "data" / "debass_photometry"
CUT = REPO / "data" / "lsst_cutouts"
OUT = RPT / "debass_lsst_viewer.html"

# SNANA FLUXCAL (ZP 27.5) -> nanojansky (ZP 31.4)
FLUXCAL_TO_NJY = 10 ** ((31.4 - 27.5) / 2.5)
PAIR_WINDOW_D = 2.0
LSST_BANDS = ["u", "g", "r", "i", "z", "y"]

# PHOTFLAG error bits from the DEBaSS README (forcePhoto.c): 8 flux-fit failed,
# 16 off-CCD, 32 masked area, 64 PSF failed, 128 no stamp, 256 negative pixels,
# 512 aper/psf mismatch. Their union is 1016; a row with any of them set is a
# failed measurement, not a faint one, and must not be plotted.
PHOTFLAG_ERROR_MASK = 1016


def read_snana(path: Path) -> pd.DataFrame:
    """Parse OBS: rows out of a SNANA-format light-curve file."""
    rows, cols = [], None
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("VARLIST:"):
            cols = line.split()[1:]
        elif line.startswith("OBS:") and cols:
            vals = line.split()[1:]
            if len(vals) >= len(cols):
                rows.append(dict(zip(cols, vals[:len(cols)])))
    df = pd.DataFrame(rows)
    for c in ("MJD", "FLUXCAL", "FLUXCALERR", "PHOTFLAG", "PSF"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["MJD", "FLUXCAL"]) if len(df) else df


def build_objects(cov: pd.DataFrame, lsst: pd.DataFrame,
                  tns: pd.DataFrame) -> list[dict]:
    from astropy.time import Time
    disc = dict(zip(tns["name"].astype(str).str.strip(), tns["discoverydate"]))
    objs = []
    both = cov[cov.debass_file.notna() & cov.matched_diaObjectId.notna()]
    for i in range(len(both)):
        snid = both["SNID"].iloc[i]
        f = PHOT / str(both["debass_file"].iloc[i])
        if not f.exists():
            print(f"  {snid}: missing {f.name}, skipping")
            continue
        d = read_snana(f)
        n_raw = len(d)
        if len(d) and "PHOTFLAG" in d:
            flags = d["PHOTFLAG"].fillna(0).astype("int64")
            d = d[(flags & PHOTFLAG_ERROR_MASK) == 0]
        n_cut = n_raw - len(d)
        l = lsst[lsst.SNID == snid].sort_values("midpointMjdTai")
        try:
            dmjd = float(Time(pd.to_datetime(disc[snid])).mjd)
        except Exception:  # noqa: BLE001
            dmjd = float("nan")

        decam = [{
            "mjd": round(float(r["MJD"]), 4),
            "band": str(r["FLT"]),
            "flux": round(float(r["FLUXCAL"]) * FLUXCAL_TO_NJY, 2),
            "ferr": round(float(r.get("FLUXCALERR") or 0) * FLUXCAL_TO_NJY, 2),
            "expnum": str(r.get("EXPNUM", "") or ""),
            "nite": str(r.get("NITE", "") or ""),
            "flag": int(r["PHOTFLAG"]) if pd.notna(r.get("PHOTFLAG")) else 0,
            "mate": None, "mate_dt": None,
        } for r in d.to_dict("records")]

        lsst_ep = [{
            "mjd": round(float(r["midpointMjdTai"]), 4),
            "band": str(r["band"]),
            "flux": round(float(r["psfFlux"]), 2),
            "ferr": round(float(r["psfFluxErr"]), 2),
            "snr": round(float(r["snr"]), 2),
            "rel": (None if pd.isna(r["reliability"])
                    else round(float(r["reliability"]), 3)),
            "visit": str(int(r["visit"])),
            "mate": None, "mate_dt": None,
        } for r in l.to_dict("records")]

        # Same-band pairs inside the window; nearest LSST epoch per DECam epoch.
        # This is many-to-one: several DECam epochs may claim the same LSST epoch.
        pairs = []
        for di, dp in enumerate(decam):
            cand = [(abs(lp["mjd"] - dp["mjd"]), li)
                    for li, lp in enumerate(lsst_ep) if lp["band"] == dp["band"]]
            cand = [c for c in cand if c[0] <= PAIR_WINDOW_D]
            if cand:
                dt, li = min(cand)
                pairs.append({"d": di, "l": li, "dt": round(dt, 3),
                              "band": dp["band"]})
        # Resolve each epoch's displayed partner up front, so the UI never has to
        # guess. For an LSST epoch claimed by several DECam epochs, the partner is
        # the temporally nearest one -- picking "the first pair found" would show
        # an arbitrary, usually wrong, mate.
        for p in pairs:
            dp = decam[p["d"]]
            if dp.get("mate") is None or p["dt"] < dp["mate_dt"]:
                dp["mate"], dp["mate_dt"] = p["l"], p["dt"]
            lp = lsst_ep[p["l"]]
            if lp.get("mate") is None or p["dt"] < lp["mate_dt"]:
                lp["mate"], lp["mate_dt"] = p["d"], p["dt"]

        objs.append({
            "snid": snid,
            "ra": float(both["tns_ra"].iloc[i]),
            "dec": float(both["tns_dec"].iloc[i]),
            "type": (None if pd.isna(both["tns_type"].iloc[i])
                     else str(both["tns_type"].iloc[i])),
            "z": (None if pd.isna(both["tns_z"].iloc[i])
                  else float(both["tns_z"].iloc[i])),
            "status": str(both["Status"].iloc[i]),
            "debass_snid": int(both["debass_snid"].iloc[i]),
            # From the parquet's int64 column, NOT the coverage CSV: that CSV
            # stores the id in float64 scientific notation (7.579403004286077e+17),
            # which has already lost the low digits of a ~1e17 value.
            "dia_id": (str(int(l["diaObjectId"].iloc[0])) if len(l) else None),
            "n_decam_flagged": int(n_cut),
            "disc_mjd": None if np.isnan(dmjd) else round(dmjd, 3),
            "decam": decam,
            "lsst": lsst_ep,
            "pairs": pairs,
        })
    return objs


def load_cutouts(snids: list[str]) -> dict:
    out = {}
    if not CUT.exists():
        return out
    for s in snids:
        for b in LSST_BANDS:
            p = CUT / f"{s}_{b}.png"
            if p.exists():
                out.setdefault(s, {})[b] = base64.b64encode(
                    p.read_bytes()).decode("ascii")
    return out


def main() -> int:
    cov = pd.read_csv(RPT / "coverage_debass_x_lsst.csv")
    lsst = pd.read_parquet(RPT / "diasources.parquet")
    tns = pd.read_csv(REPO / "data" / "tns_public_objects.csv",
                      skiprows=1, low_memory=False)
    objs = build_objects(cov, lsst, tns)
    cuts = load_cutouts([o["snid"] for o in objs])

    n_pairs = sum(len(o["pairs"]) for o in objs)
    n_dec = sum(len(o["decam"]) for o in objs)
    n_lsst = sum(len(o["lsst"]) for o in objs)
    n_cut = sum(len(v) for v in cuts.values())
    print(f"objects={len(objs)} decam_epochs={n_dec} lsst_epochs={n_lsst} "
          f"pairs={n_pairs} cutouts={n_cut}")

    payload = {
        "objects": objs,
        "cutouts": cuts,
        "meta": {
            "n_obj": len(objs), "n_decam": n_dec, "n_lsst": n_lsst,
            "n_pairs": n_pairs, "n_cutouts": n_cut,
            "pair_window_d": PAIR_WINDOW_D,
            "fluxcal_to_njy": round(FLUXCAL_TO_NJY, 4),
        },
    }
    tpl = (REPO / "scripts" / "viewer_template.html").read_text()
    html = tpl.replace("/*__PAYLOAD__*/", json.dumps(payload, separators=(",", ":")))
    OUT.write_text(html)
    print(f"-> {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
