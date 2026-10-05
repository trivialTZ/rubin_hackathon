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

# ------------------------------------------------------------------ #
# fusion_v11 negative-flux features (B3/B10).                          #
# ------------------------------------------------------------------ #
# These are computed by the gold BUILDER (build_snapshots_fusion.py
# _extract_object_rows) from the negatives-INCLUDED window returned by
# ``truncated_detection_windows`` — NOT by ``extract_features`` (which never
# sees negatives, so the locked base-51 columns stay value-identical: G5b).
# Kept as a SEPARATE list; FEATURE_NAMES stays 51.  ``DEFAULT_FEATURES`` in
# ``models/early_meta.py`` is extended in sync (CLAUDE.md contract).
NEG_FEATURE_NAMES = [
    "n_det_neg",          # # negative detections in the window (t <= t(Nth pos))
    "frac_neg",           # n_det_neg / window_size   (0.0 when window empty)
    "n_pos_det",          # # POSITIVE detections in the window (0 on ZTF fallback)
    "t_since_last_pos",   # days between the last two positive detections (0 if <2)
    "neg_run_frac",       # longest consecutive negative run / window_size
]

# fusion_v11 gold flag column (SEPARATE from FEATURE_NAMES and NEG_FEATURE_NAMES;
# NOT a model feature — a per-row bookkeeping flag added by the gold BUILDER).
# 1.0 only on ZTF all-negative fallback rows (0 real positives, epochs kept alive
# by B3); scopes the `n_det == n_pos_det` tripwire and the `n_pos_det >= 1`
# metrics/G2 denominator gate (spec §2.3, deviation #30).
LC_FALLBACK_ALL_NEGATIVE = "lc_fallback_all_negative"


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


def _resolve_window_survey(
    normalized_dets: list[dict[str, Any]], survey: str | None
) -> str:
    """Resolve the survey used for the all-negative fallback gate.

    ``survey`` may be ``"LSST"``/``"ZTF"`` (explicit) or ``"auto"``/None
    (inferred from the normalized detections' ``survey`` stamps).
    """
    if survey is not None and str(survey).lower() != "auto":
        return "LSST" if str(survey).upper() == "LSST" else "ZTF"
    return "LSST" if any(d.get("survey") == "LSST" for d in normalized_dets) else "ZTF"


def truncated_detection_windows(
    detections: list[dict[str, Any]],
    *,
    survey: str = "auto",
    max_n_det: int = 20,
    lsst_all_negative_fallback: bool = False,
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """Canonical fusion_v11 truncation helper (B10 — single implementation).

    For each epoch ``N`` in ``1..min(#positives, max_n_det)`` returns a tuple
    ``(pos_prefix, full_window)``:

      * ``pos_prefix`` — the first ``N`` POSITIVE detections (MJD-sorted).  This
        is the EXACT list the base-51 extractor and the EXT features consume, so
        those columns stay value-identical to v10 on ZTF (G5b).  For non-LSST
        surveys the all-negative fallback is KEPT (an all-negative ZTF LC uses
        every detection as its "positive" prefix — 11% of ZTF LCs depend on it).
      * ``full_window`` — every detection (positive OR negative) with
        ``mjd <= mjd(Nth positive)`` (MJD-sorted).  Feeds the 5 NEG features and
        the v11 sequence tokens.  ``full_window`` contains EXACTLY the ``N``
        positives of ``pos_prefix`` plus the negatives up to that time, so the
        builder tripwire ``n_det == n_pos_det`` always holds (tie-safe).

    Survey gating (B3): the all-negative fallback is REMOVED for LSST — an LSST
    object with 0 positive detections yields ``[]`` (0 epoch rows).  Opt-in
    ``lsst_all_negative_fallback=True`` (v13g) applies the ZTF fallback to LSST as
    well: on Rubin the SN light of ~15% of the spectroscopic SNe sits in the
    difference-imaging template, so every alert detection is negative.

    The base extractor, the gold builder and P5's ``sequence_dataset`` all route
    through this one helper (no lockstep copies).
    """
    ndets = [_ensure_normalized(d) for d in detections]
    ndets.sort(key=lambda d: d.get("mjd") or 0)
    resolved = _resolve_window_survey(ndets, survey)

    pos_dets = [d for d in ndets if d.get("is_positive", True)]
    if not pos_dets:
        if resolved == "LSST" and not lsst_all_negative_fallback:
            return []
        pos_dets = ndets  # all-negative fallback (ZTF; LSST only when opted in)

    pos_id_set = {id(d) for d in pos_dets}
    neg_dets = [d for d in ndets if id(d) not in pos_id_set]

    windows: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
    for i in range(min(len(pos_dets), max_n_det)):
        pos_prefix = pos_dets[: i + 1]
        t_nth = pos_prefix[-1].get("mjd") or 0
        neg_in_window = [d for d in neg_dets if (d.get("mjd") or 0) <= t_nth]
        full_window = sorted(
            pos_prefix + neg_in_window, key=lambda d: d.get("mjd") or 0
        )
        windows.append((pos_prefix, full_window))
    return windows


def compute_neg_features(
    pos_prefix: list[dict[str, Any]],
    full_window: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute the 5 :data:`NEG_FEATURE_NAMES` from one epoch window.

    ``pos_prefix`` and ``full_window`` come from the SAME
    :func:`truncated_detection_windows` call.  Positivity is read from each
    detection's ``is_positive`` flag (NOT prefix membership) so the values are
    honest on the ZTF all-negative FALLBACK window too: there ``full_window`` is
    entirely negative, so ``n_pos_det == 0`` and ``n_det_neg == len(window)``
    (spec §2.3, deviation #30).  ``full_window`` MUST be MJD-sorted (the helper
    sorts it) because ``neg_run_frac`` is order-sensitive.  Every feature is a
    pure function of the window (all dets with ``t <= t(Nth pos)``), so appending
    FUTURE negative detections cannot change any value at a fixed ``n_pos_det``
    (G5).

    On a normal (non-fallback) window ``n_pos_det`` equals the number of positive
    detections, which equals ``len(pos_prefix)`` and the extractor's ``n_det`` —
    so the builder tripwire ``n_det == n_pos_det`` holds on every NON-fallback
    row.  On a fallback row ``n_pos_det == 0 != n_det``; that row is flagged
    ``lc_fallback_all_negative == 1`` and excluded from the tripwire and from the
    ``n_pos_det >= 1`` metrics/G2 denominator.  ``pos_prefix`` is accepted for
    call-site symmetry; classification uses ``full_window``.
    """
    n_win = len(full_window)
    # Classify each window detection by its is_positive flag (window is MJD-sorted).
    is_neg = [not bool(d.get("is_positive")) for d in full_window]
    n_neg = sum(is_neg)
    n_pos = n_win - n_neg          # actual POSITIVE detections in the window
    frac_neg = (n_neg / n_win) if n_win > 0 else 0.0

    longest = cur = 0
    for neg in is_neg:
        if neg:
            cur += 1
            if cur > longest:
                longest = cur
        else:
            cur = 0
    neg_run_frac = (longest / n_win) if n_win > 0 else 0.0

    pos_mjds = sorted(
        float(d.get("mjd") or 0.0) for d in full_window if bool(d.get("is_positive"))
    )
    if len(pos_mjds) >= 2:
        t_since_last_pos = float(pos_mjds[-1] - pos_mjds[-2])
    else:
        t_since_last_pos = 0.0

    return {
        "n_det_neg": float(n_neg),
        "frac_neg": float(frac_neg),
        "n_pos_det": float(n_pos),
        "t_since_last_pos": float(t_since_last_pos),
        "neg_run_frac": float(neg_run_frac),
    }


def extract_features_at_each_epoch(
    detections: list[dict[str, Any]],
    max_n_det: int = 20,
    *,
    survey: str = "auto",
    lsst_all_negative_fallback: bool = False,
) -> list[dict[str, Any]]:
    """Return a list of feature dicts, one per detection epoch 1..min(len, max_n_det).

    Each dict has ``n_det`` and ``alert_mjd`` plus all FEATURE_NAMES.
    The lightcurve is truncated to exactly n_det POSITIVE detections before
    computing features (via :func:`truncated_detection_windows`).  ``survey``
    controls the all-negative fallback gate (kept for ZTF, removed for LSST unless
    ``lsst_all_negative_fallback``).
    """
    windows = truncated_detection_windows(
        detections, survey=survey, max_n_det=max_n_det,
        lsst_all_negative_fallback=lsst_all_negative_fallback,
    )
    results = []
    for pos_prefix, _full_window in windows:
        n_det = len(pos_prefix)
        feats = extract_features(pos_prefix)
        feats["n_det"] = float(n_det)
        feats["alert_mjd"] = float(pos_prefix[-1].get("mjd") or 0)
        results.append(feats)

    return results
