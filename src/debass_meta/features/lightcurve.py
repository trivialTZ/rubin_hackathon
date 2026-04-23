"""Per-epoch lightcurve feature extractor.

All features are computed from a TRUNCATED lightcurve (detections 1..n_det only).
This ensures there is NO data leakage — features at n_det=4 only use information
that would be available after the 4th detection.

Supports both ZTF (g, r) and LSST (u, g, r, i, z, y) bands.  For ZTF objects
the LSST-only band features (u, i, z, y) and extra colour features (r-i, i-z,
z-y) are NaN — LightGBM handles this natively.

Detections should be passed through :func:`detection.normalize_lightcurve` first
so that ``band``, ``mag``, ``quality``, etc. are in the common schema.  For
backward compatibility the extractor also handles legacy ZTF dicts (``fid``,
``magpsf``, ``rb``/``drb``) directly by normalizing on the fly.

Features computed
-----------------
  Detection counts (8):
    n_det                 total detections
    n_det_{u,g,r,i,z,y}  per-band detection counts
  Temporal (2):
    t_since_first         days since first detection
    t_baseline            = t_since_first
  All-band brightness (8):
    mag_first, mag_last, mag_min, mag_mean, mag_std, mag_range
    dmag_dt               mag change per day (linear slope)
    dmag_first_last       mag_last - mag_first
  Per-band brightness (24):
    mag_first_{b}, mag_last_{b}, mag_mean_{b}, mag_std_{b}  for b in ugrizy
  Colours (8):
    color_gr, color_gr_slope
    color_ri, color_ri_slope
    color_iz, color_iz_slope
    color_zy, color_zy_slope
  Quality (1):
    mean_quality          mean quality score (rb/drb for ZTF, reliability for LSST)
  Survey flag (1):
    survey_is_lsst        0.0 = ZTF, 1.0 = LSST
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .detection import (
    CANONICAL_BANDS,
    ZTF_FID_TO_BAND,
    normalize_detection,
)

# Colour pairs: (blue-band, red-band)
COLOR_PAIRS = [("g", "r"), ("r", "i"), ("i", "z"), ("z", "y")]

# Feature column names in the order they are returned by extract_features()
FEATURE_NAMES = [
    # Detection counts (8)
    "n_det",
    "n_det_u", "n_det_g", "n_det_r", "n_det_i", "n_det_z", "n_det_y",
    # Temporal (2)
    "t_since_first", "t_baseline",
    # All-band brightness (8)
    "mag_first", "mag_last", "mag_min", "mag_mean", "mag_std", "mag_range",
    "dmag_dt", "dmag_first_last",
    # Per-band brightness: u g r i z y × (first, last, mean, std) = 24
    "mag_first_u", "mag_last_u", "mag_mean_u", "mag_std_u",
    "mag_first_g", "mag_last_g", "mag_mean_g", "mag_std_g",
    "mag_first_r", "mag_last_r", "mag_mean_r", "mag_std_r",
    "mag_first_i", "mag_last_i", "mag_mean_i", "mag_std_i",
    "mag_first_z", "mag_last_z", "mag_mean_z", "mag_std_z",
    "mag_first_y", "mag_last_y", "mag_mean_y", "mag_std_y",
    # Colours (8)
    "color_gr", "color_gr_slope",
    "color_ri", "color_ri_slope",
    "color_iz", "color_iz_slope",
    "color_zy", "color_zy_slope",
    # Quality (1)
    "mean_quality",
    # Survey flag (1)
    "survey_is_lsst",
]

# Legacy alias kept so existing code that imports ``mean_rb`` keeps working.
_LEGACY_QUALITY_ALIAS = "mean_rb"


def _ensure_normalized(det: dict[str, Any]) -> dict[str, Any]:
    """If the detection is already normalized (has ``band`` + ``mag`` keys) return
    it as-is; otherwise normalize on the fly for backward compatibility."""
    if "band" in det and "mag" in det and "quality" in det:
        return det
    return normalize_detection(det)


def _get_band_dets(
    normalized_dets: list[dict[str, Any]],
    band: str,
) -> tuple[list[float], list[float]]:
    """Return (mjds, mags) for a single band, filtering NaN magnitudes."""
    mjds, mags = [], []
    for d in normalized_dets:
        if d.get("band") != band:
            continue
        m = d.get("mag")
        if m is not None and np.isfinite(m):
            mjds.append(d["mjd"])
            mags.append(m)
    return mjds, mags


def _compute_color_pair(
    normalized_dets: list[dict[str, Any]],
    band1: str,
    band2: str,
    feats: dict[str, float],
    *,
    max_dt: float = 3.0,
) -> None:
    """Compute nearest-in-time colour (band1 − band2) and its slope."""
    key_color = f"color_{band1}{band2}"
    key_slope = f"color_{band1}{band2}_slope"

    b1_dets = [(d["mjd"], d["mag"]) for d in normalized_dets
               if d.get("band") == band1 and d.get("mag") is not None and np.isfinite(d["mag"])]
    b2_dets = [(d["mjd"], d["mag"]) for d in normalized_dets
               if d.get("band") == band2 and d.get("mag") is not None and np.isfinite(d["mag"])]

    if not b1_dets or not b2_dets:
        return

    b1_arr = np.array(b1_dets)
    b2_arr = np.array(b2_dets)

    color_vals: list[float] = []
    color_mjds: list[float] = []
    for b1_t, b1_m in b1_arr:
        dt = np.abs(b2_arr[:, 0] - b1_t)
        j = int(dt.argmin())
        if dt[j] < max_dt:
            color_vals.append(b1_m - b2_arr[j, 1])
            color_mjds.append(b1_t)

    if color_vals:
        feats[key_color] = float(np.mean(color_vals))
        if len(color_vals) >= 2 and len(set(color_mjds)) >= 2:
            try:
                with np.errstate(all="ignore"):
                    cslope, _ = np.polyfit(
                        np.array(color_mjds) - color_mjds[0],
                        color_vals,
                        1,
                    )
                if np.isfinite(cslope):
                    feats[key_slope] = float(cslope)
            except Exception:
                pass


def extract_features(detections: list[dict[str, Any]]) -> dict[str, float]:
    """Compute lightcurve features from a list of detection dicts.

    Accepts both raw (ZTF/LSST) and pre-normalized detections.
    Returns a dict mapping feature_name → float (NaN when not computable).
    """
    feats: dict[str, float] = {k: np.nan for k in FEATURE_NAMES}

    if not detections:
        feats["n_det"] = 0.0
        return feats

    # Normalize on the fly if needed
    ndets = [_ensure_normalized(d) for d in detections]

    # Sort by MJD
    ndets.sort(key=lambda d: d.get("mjd") or 0)

    # --- Survey flag ---
    surveys = [d.get("survey", "ZTF") for d in ndets]
    feats["survey_is_lsst"] = 1.0 if any(s == "LSST" for s in surveys) else 0.0

    # --- All-band arrays ---
    mjds = np.array([d.get("mjd") or 0 for d in ndets], dtype=float)
    mags = np.array([d.get("mag") for d in ndets], dtype=float)
    valid = ~np.isnan(mags)
    mjds_v = mjds[valid]
    mags_v = mags[valid]

    # n_det counts all detections (including those with NaN mag); per-band
    # counts and mag features use only valid mags.  This is intentional:
    # n_det reflects the alert stream count, not the photometry count.
    feats["n_det"] = float(len(ndets))

    if len(mjds_v) == 0:
        return feats

    t0 = mjds_v[0]
    feats["t_since_first"] = float(mjds_v[-1] - t0)
    feats["t_baseline"] = float(mjds_v[-1] - t0)
    feats["mag_first"] = float(mags_v[0])
    feats["mag_last"] = float(mags_v[-1])
    feats["mag_min"] = float(np.nanmin(mags_v))
    feats["mag_mean"] = float(np.nanmean(mags_v))
    feats["mag_std"] = float(np.nanstd(mags_v)) if len(mags_v) > 1 else 0.0
    feats["mag_range"] = float(np.nanmax(mags_v) - np.nanmin(mags_v))
    feats["dmag_first_last"] = float(mags_v[-1] - mags_v[0])

    # Linear slope dmag/dt
    if len(mjds_v) >= 2 and np.unique(mjds_v).size >= 2:
        try:
            with np.errstate(all="ignore"):
                slope, _ = np.polyfit(mjds_v - t0, mags_v, 1)
            if np.isfinite(slope):
                feats["dmag_dt"] = float(slope)
        except Exception:
            pass

    # --- Per-band features ---
    for band in CANONICAL_BANDS:
        band_count = sum(1 for d in ndets if d.get("band") == band)
        feats[f"n_det_{band}"] = float(band_count)
        if band_count == 0:
            continue
        _, bmags_list = _get_band_dets(ndets, band)
        if not bmags_list:
            continue
        bvalid = np.array(bmags_list, dtype=float)
        feats[f"mag_first_{band}"] = float(bvalid[0])
        feats[f"mag_last_{band}"] = float(bvalid[-1])
        feats[f"mag_mean_{band}"] = float(np.nanmean(bvalid))
        feats[f"mag_std_{band}"] = float(np.nanstd(bvalid)) if len(bvalid) > 1 else 0.0

    # --- Colours ---
    for band1, band2 in COLOR_PAIRS:
        _compute_color_pair(ndets, band1, band2, feats)

    # --- Quality ---
    quals = [d.get("quality") for d in ndets
             if d.get("quality") is not None and np.isfinite(d.get("quality"))]
    if quals:
        feats["mean_quality"] = float(np.mean([float(q) for q in quals]))

    return feats


def extract_features_at_each_epoch(
    detections: list[dict[str, Any]],
    max_n_det: int = 20,
) -> list[dict[str, Any]]:
    """Return a list of feature dicts, one per detection epoch 1..min(len, max_n_det).

    Each dict has ``n_det`` and ``alert_mjd`` plus all FEATURE_NAMES.
    The lightcurve is truncated to exactly n_det detections before computing features.
    """
    # Normalize first
    ndets = [_ensure_normalized(d) for d in detections]

    # Filter to positive detections
    pos_dets = [d for d in ndets if d.get("is_positive", True)]
    if not pos_dets:
        pos_dets = ndets

    pos_dets.sort(key=lambda d: d.get("mjd") or 0)

    results = []
    for i in range(min(len(pos_dets), max_n_det)):
        n_det = i + 1
        truncated = pos_dets[:n_det]
        feats = extract_features(truncated)
        feats["n_det"] = float(n_det)
        feats["alert_mjd"] = float(truncated[-1].get("mjd") or 0)
        results.append(feats)

    return results
