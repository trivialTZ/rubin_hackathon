"""Per-epoch lightcurve feature extractor.

All features are computed from a TRUNCATED lightcurve (detections 1..n_det only).
This ensures there is NO data leakage — features at n_det=4 only use information
that would be available after the 4th detection.

Bands:
  fid=1 → g-band
  fid=2 → r-band
  fid=3 → i-band (LSST)

Features computed:
  Temporal:
    t_since_first     days since first detection
    t_baseline        time span from first to last detection in slice
  Brightness (per band and combined):
    mag_first         magnitude of first detection
    mag_last          magnitude at this epoch
    mag_min           brightest magnitude seen so far
    mag_mean          mean magnitude
    mag_std           std of magnitudes
    mag_range         max - min magnitude
  Rise/decline:
    dmag_dt           magnitude change per day (linear slope over all detections)
    dmag_first_last   mag_last - mag_first (positive = fading, negative = rising)
  Color (requires both g and r):
    color_gr          g - r at closest epochs
    color_gr_slope    rate of color change per day
  Detection count:
    n_det             total detections (= the epoch index, always included)
    n_det_g           detections in g-band
    n_det_r           detections in r-band
  Quality:
    mean_rb           mean real-bogus score (if available)
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Feature column names in the order they are returned by extract_features()
FEATURE_NAMES = [
    "n_det", "n_det_g", "n_det_r",
    "t_since_first", "t_baseline",
    "mag_first", "mag_last", "mag_min", "mag_mean", "mag_std", "mag_range",
    "dmag_dt", "dmag_first_last",
    "mag_first_g", "mag_last_g", "mag_mean_g", "mag_std_g",
    "mag_first_r", "mag_last_r", "mag_mean_r", "mag_std_r",
    "color_gr", "color_gr_slope",
    "mean_rb",
]


def extract_features(detections: list[dict[str, Any]]) -> dict[str, float]:
    """Compute lightcurve features from a list of detection dicts.

    detections: list of dicts with keys mjd, magpsf, sigmapsf, fid, isdiffpos, rb (optional)
    Returns a dict mapping feature_name -> float (NaN when not computable).
    """
    feats: dict[str, float] = {k: np.nan for k in FEATURE_NAMES}

    if not detections:
        feats["n_det"] = 0.0
        return feats

    # Sort by MJD
    dets = sorted(detections, key=lambda d: d.get("mjd") or d.get("jd") or 0)

    # --- All-band arrays ---
    mjds  = np.array([d.get("mjd") or d.get("jd") or 0 for d in dets], dtype=float)
    mags  = np.array([d.get("magpsf") for d in dets], dtype=float)
    valid = ~np.isnan(mags)

    mjds_v = mjds[valid]
    mags_v = mags[valid]

    feats["n_det"] = float(len(dets))

    if len(mjds_v) == 0:
        return feats

    t0 = mjds_v[0]
    feats["t_since_first"]   = float(mjds_v[-1] - t0)
    feats["t_baseline"]      = float(mjds_v[-1] - t0)
    feats["mag_first"]       = float(mags_v[0])
    feats["mag_last"]        = float(mags_v[-1])
    feats["mag_min"]         = float(np.nanmin(mags_v))   # brightest
    feats["mag_mean"]        = float(np.nanmean(mags_v))
    feats["mag_std"]         = float(np.nanstd(mags_v)) if len(mags_v) > 1 else 0.0
    feats["mag_range"]       = float(np.nanmax(mags_v) - np.nanmin(mags_v))
    feats["dmag_first_last"] = float(mags_v[-1] - mags_v[0])

    # Linear slope dmag/dt over all valid detections
    if len(mjds_v) >= 2:
        try:
            with np.errstate(all="ignore"):
                slope, _ = np.polyfit(mjds_v - t0, mags_v, 1)
            if np.isfinite(slope):
                feats["dmag_dt"] = float(slope)
        except Exception:
            pass

    # --- Per-band ---
    for fid, band in [(1, "g"), (2, "r")]:
        band_dets = [d for d in dets if d.get("fid") == fid]
        feats[f"n_det_{band}"] = float(len(band_dets))
        if not band_dets:
            continue
        bmags = np.array([d.get("magpsf") for d in band_dets], dtype=float)
        bvalid = bmags[~np.isnan(bmags)]
        if len(bvalid) == 0:
            continue
        feats[f"mag_first_{band}"] = float(bvalid[0])
        feats[f"mag_last_{band}"]  = float(bvalid[-1])
        feats[f"mag_mean_{band}"]  = float(np.nanmean(bvalid))
        feats[f"mag_std_{band}"]   = float(np.nanstd(bvalid)) if len(bvalid) > 1 else 0.0

    # --- Color g-r ---
    g_dets = [(d.get("mjd") or d.get("jd") or 0, d.get("magpsf"))
              for d in dets if d.get("fid") == 1 and d.get("magpsf") is not None]
    r_dets = [(d.get("mjd") or d.get("jd") or 0, d.get("magpsf"))
              for d in dets if d.get("fid") == 2 and d.get("magpsf") is not None]

    if g_dets and r_dets:
        # Nearest-in-time g-r pair
        g_arr = np.array(g_dets)
        r_arr = np.array(r_dets)
        # For each g detection, find closest r
        color_vals, color_mjds = [], []
        for g_t, g_m in g_arr:
            dt = np.abs(r_arr[:, 0] - g_t)
            j = dt.argmin()
            if dt[j] < 3.0:  # within 3 days
                color_vals.append(g_m - r_arr[j, 1])
                color_mjds.append(g_t)
        if color_vals:
            feats["color_gr"] = float(np.mean(color_vals))
            if len(color_vals) >= 2:
                try:
                    with np.errstate(all="ignore"):
                        cslope, _ = np.polyfit(
                            np.array(color_mjds) - color_mjds[0],
                            color_vals, 1
                        )
                    if np.isfinite(cslope):
                        feats["color_gr_slope"] = float(cslope)
                except Exception:
                    pass

    # --- Real-bogus ---
    rbs = [d.get("rb") or d.get("drb") for d in dets if (d.get("rb") or d.get("drb")) is not None]
    if rbs:
        feats["mean_rb"] = float(np.mean([float(x) for x in rbs]))

    return feats


def extract_features_at_each_epoch(
    detections: list[dict[str, Any]],
    max_n_det: int = 20,
) -> list[dict[str, Any]]:
    """Return a list of feature dicts, one per detection epoch 1..min(len,max_n_det).

    Each dict has 'n_det' and 'alert_mjd' plus all FEATURE_NAMES.
    The lightcurve is truncated to exactly n_det detections before computing features.
    """
    dets_sorted = sorted(
        [d for d in detections if d.get("isdiffpos") in ("t", "1", 1, True, "true")],
        key=lambda d: d.get("mjd") or d.get("jd") or 0,
    )
    if not dets_sorted:
        dets_sorted = sorted(
            detections,
            key=lambda d: d.get("mjd") or d.get("jd") or 0,
        )

    results = []
    for i in range(min(len(dets_sorted), max_n_det)):
        n_det = i + 1
        truncated = dets_sorted[:n_det]
        feats = extract_features(truncated)
        feats["n_det"] = float(n_det)  # ensure exact count
        feats["alert_mjd"] = float(
            truncated[-1].get("mjd") or truncated[-1].get("jd") or 0
        )
        results.append(feats)

    return results
