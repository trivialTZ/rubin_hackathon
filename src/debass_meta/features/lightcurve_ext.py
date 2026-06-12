"""Extended no-leakage lightcurve features for fusion_v8 (50 features).

Kept separate from :mod:`lightcurve` so the locked v6e2 path is untouched.
Input contract: the SAME truncated, positive-filtered, MJD-sorted normalized
detection list consumed by :func:`lightcurve.extract_features` (see
``extract_features_at_each_epoch``).  For backward compatibility raw ZTF/LSST
dicts are normalized on the fly, exactly like the base extractor.

No-leakage argument
-------------------
``extract_ext_features`` is a *pure* function of the detection list it is
given — no I/O, no global state.  The fusion_v8 gold builder passes the
truncated prefix (detections 1..n_det only), so every feature below is a
function of those detections plus per-detection metadata stamped on them at
alert time (``sigmapsf``, ``diffmaglim``, ``distnr``, ``rb``/``drb``,
``corrected`` for ZTF; ``psfFluxErr``, ``snr``, ``reliability`` for LSST DP1).
Extending the lightcurve with FUTURE detections cannot change the value at a
fixed n_det (tested in ``tests/test_lightcurve_ext.py``).

Feature groups (design doc §1.2, listing order = EXT_FEATURE_NAMES order)
-------------------------------------------------------------------------
  Cadence / sampling (10):
    dt_mean, dt_median, dt_min, dt_max, dt_std   gaps g_k = mjd_{k+1}-mjd_k
                                                  (NaN if n_det<2)
    n_nights              unique floor(mjd)
    dets_per_night_max    max detections in a single night
    intranight_frac       frac of gaps < 0.5 d
    max_gap_frac          dt_max / t_baseline (NaN if t_baseline == 0)
    band_alternation_rate frac of consecutive det pairs in different bands
  Photometric error / SNR (8):
    magerr_mean, magerr_min, magerr_max, magerr_last
    snr_mean, snr_max, snr_last     snr = 1.0857/magerr (ZTF) or native (LSST)
    dmag_dt_wls           weighted-least-squares slope, weights 1/magerr^2
  Detection significance (4) — ZTF only (LSST NaN):
    sig_above_lim_mean, sig_above_lim_min, sig_above_lim_last
                          per-detection diffmaglim - magpsf
    diffmaglim_mean
  Variability / periodicity proxies (12) — n_det>=3 (NaN below):
    eta_vn                von Neumann ratio mean(diff(m)^2)/var(m, ddof=1);
                          low for smooth SN evolution, ~2 for noise/oscillation
    eta_vn_g, eta_vn_r    per-band variants (>=3 dets in band, else NaN)
    flip_rate             sign changes of consecutive dmag / (n-2)
    longest_run_frac      longest same-sign dmag run / (n-1)
    chi2_const_dof        sum(((m-mbar_w)/sigma)^2)/(n-1), inv-var weighted mean
    stetson_J_gr          Stetson J on nearest-in-time g,r pairs (<3 d),
                          NaN if <2 pairs
    stetson_K             Stetson K kurtosis proxy, all-band
    amp_iqr               IQR(m) robust amplitude
    mad_mag               median absolute deviation of m
    frac_brighter_first   frac of dets brighter (mag <) than the first det
    amp_per_err           amp_iqr / magerr_mean
  Lomb-Scargle (4) — only n_det>=8 in best band, else NaN:
    ls_power_peak, ls_logperiod_peak, ls_power_ratio_21, ls_n_used
    (grid 0.05-50 d; astropy.timeseries.LombScargle, scipy fallback)
  Shape / rise (8):
    rise_rate_g, rise_rate_r   per-band linear slope (mag/day; neg = brightening)
    dmag_dt_recent3            slope over the last 3 detections
    curvature_t2               quadratic coefficient of mag(t), n_det>=4
    brightening_frac           frac of consecutive pairs that brighten
    t_since_brightest          days since the min-mag detection
    is_pre_peak                1.0 if the last det is the brightest so far
    color_gr_at_brightest      nearest-in-time g-r at brightest epoch (<3 d)
  Reference-source context, static (4):
    distnr_mean, distnr_min    ZTF distance to nearest reference source (arcsec);
                               LSST NaN
    quality_min                min rb/drb (ZTF) or reliability (LSST)
    corrected_frac             ZTF frac of dets with corrected=True; LSST NaN

All features NaN when not computable (LightGBM-native missing handling).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .detection import BAND_TO_IDX
from .lightcurve import _ensure_normalized

# Feature column names, exactly in design §1.2 listing order.
EXT_FEATURE_NAMES: list[str] = [
    # Cadence / sampling (10)
    "dt_mean", "dt_median", "dt_min", "dt_max", "dt_std",
    "n_nights", "dets_per_night_max", "intranight_frac",
    "max_gap_frac", "band_alternation_rate",
    # Photometric error / SNR (8)
    "magerr_mean", "magerr_min", "magerr_max", "magerr_last",
    "snr_mean", "snr_max", "snr_last",
    "dmag_dt_wls",
    # Detection significance (4) — ZTF only
    "sig_above_lim_mean", "sig_above_lim_min", "sig_above_lim_last",
    "diffmaglim_mean",
    # Variability / periodicity proxies (12)
    "eta_vn", "eta_vn_g", "eta_vn_r",
    "flip_rate", "longest_run_frac", "chi2_const_dof",
    "stetson_J_gr", "stetson_K",
    "amp_iqr", "mad_mag", "frac_brighter_first", "amp_per_err",
    # Lomb-Scargle (4)
    "ls_power_peak", "ls_logperiod_peak", "ls_power_ratio_21", "ls_n_used",
    # Shape / rise (8)
    "rise_rate_g", "rise_rate_r", "dmag_dt_recent3", "curvature_t2",
    "brightening_frac", "t_since_brightest", "is_pre_peak",
    "color_gr_at_brightest",
    # Reference-source context (4)
    "distnr_mean", "distnr_min", "quality_min", "corrected_frac",
]

# --- Tunables (frozen for fusion_v8) ---
_INTRANIGHT_DT = 0.5      # days; a gap below this counts as intranight
_PAIR_MAX_DT = 3.0        # days; nearest-in-time pairing window (colors, Stetson J)
_VAR_MIN_DETS = 3         # variability group gate
_LS_MIN_DETS = 8          # Lomb-Scargle gate (per best band)
_LS_PERIOD_MIN = 0.05     # days
_LS_PERIOD_MAX = 50.0     # days
_LS_N_GRID = 5000         # log-spaced trial periods (deterministic grid)


def _isfinite(value: Any) -> bool:
    """True when *value* is a finite float (None/str/NaN safe)."""
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _polyfit_coeff(
    t: np.ndarray,
    y: np.ndarray,
    deg: int,
    *,
    w: np.ndarray | None = None,
) -> float:
    """Leading polyfit coefficient (slope for deg=1, curvature for deg=2).

    Returns NaN when the fit is impossible (too few distinct times) or
    numerically fails — matching the base extractor's defensive idiom.
    """
    if len(t) < deg + 1 or np.unique(t).size < deg + 1:
        return float("nan")
    try:
        with np.errstate(all="ignore"):
            coeffs = np.polyfit(t - t[0], y, deg, w=w)
        c = float(coeffs[0])
        return c if math.isfinite(c) else float("nan")
    except Exception:
        return float("nan")


def _von_neumann(mags: np.ndarray) -> float:
    """von Neumann ratio mean(diff(m)^2) / var(m, ddof=1); NaN if n<3 or var==0."""
    if len(mags) < _VAR_MIN_DETS:
        return float("nan")
    var = float(np.var(mags, ddof=1))
    if not math.isfinite(var) or var <= 0:
        return float("nan")
    diffs = np.diff(mags)
    return float(np.mean(diffs ** 2) / var)


def _longest_same_sign_run(signs: np.ndarray) -> int:
    """Length of the longest run of equal consecutive values."""
    longest, current = 1, 1
    for k in range(1, len(signs)):
        if signs[k] == signs[k - 1]:
            current += 1
        else:
            current = 1
        longest = max(longest, current)
    return longest


def _nearest_pairs(
    primary: list[tuple[float, float, float]],
    secondary: list[tuple[float, float, float]],
    max_dt: float,
) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """Nearest-in-time pairs (one per *primary* det) within *max_dt* days.

    Each element of *primary*/*secondary* is ``(mjd, mag, magerr)``.
    Mirrors the pairing convention of lightcurve._compute_color_pair
    (secondary detections may be reused).
    """
    if not primary or not secondary:
        return []
    sec_t = np.array([s[0] for s in secondary], dtype=float)
    pairs = []
    for p in primary:
        dt = np.abs(sec_t - p[0])
        j = int(dt.argmin())
        if dt[j] < max_dt:
            pairs.append((p, secondary[j]))
    return pairs


def _stetson_j(
    pairs: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
) -> float:
    """Stetson J statistic over paired two-band observations; NaN if <2 pairs."""
    n = len(pairs)
    if n < 2:
        return float("nan")
    m1 = np.array([p[0][1] for p in pairs], dtype=float)
    e1 = np.array([p[0][2] for p in pairs], dtype=float)
    m2 = np.array([p[1][1] for p in pairs], dtype=float)
    e2 = np.array([p[1][2] for p in pairs], dtype=float)
    if np.any(e1 <= 0) or np.any(e2 <= 0):
        return float("nan")
    factor = math.sqrt(n / (n - 1.0))
    d1 = factor * (m1 - m1.mean()) / e1
    d2 = factor * (m2 - m2.mean()) / e2
    p_k = d1 * d2
    return float(np.mean(np.sign(p_k) * np.sqrt(np.abs(p_k))))


def _stetson_k(mags: np.ndarray, errs: np.ndarray) -> float:
    """Stetson K kurtosis proxy; ~0.798 for Gaussian noise; NaN if n<3."""
    n = len(mags)
    if n < _VAR_MIN_DETS:
        return float("nan")
    factor = math.sqrt(n / (n - 1.0))
    delta = factor * (mags - mags.mean()) / errs
    denom = float(np.sqrt(np.mean(delta ** 2)))
    if not math.isfinite(denom) or denom <= 0:
        return float("nan")
    return float(np.mean(np.abs(delta)) / denom)


def _lomb_scargle(
    t: np.ndarray,
    m: np.ndarray,
    e: np.ndarray | None,
    feats: dict[str, float],
) -> None:
    """Fill the 4 ls_* features from a deterministic 0.05-50 d period grid.

    Computed on the truncated lightcurve only (no leakage).  Prefers
    astropy.timeseries.LombScargle; falls back to scipy.signal.lombscargle.
    """
    periods = np.geomspace(_LS_PERIOD_MIN, _LS_PERIOD_MAX, _LS_N_GRID)
    freqs = 1.0 / periods
    try:
        from astropy.timeseries import LombScargle

        dy = e if e is not None and np.all(np.isfinite(e)) and np.all(e > 0) else None
        power = np.asarray(LombScargle(t, m, dy).power(freqs), dtype=float)
    except ImportError:  # pragma: no cover — astropy pinned in requirements
        from scipy.signal import lombscargle

        power = lombscargle(t, m - m.mean(), 2.0 * np.pi * freqs, normalize=True)
        power = np.asarray(power, dtype=float)
    if not np.any(np.isfinite(power)):
        return
    power = np.where(np.isfinite(power), power, -np.inf)
    ipk = int(np.argmax(power))
    feats["ls_power_peak"] = float(power[ipk])
    feats["ls_logperiod_peak"] = float(np.log10(periods[ipk]))
    feats["ls_n_used"] = float(len(t))

    try:
        from scipy.signal import find_peaks

        peak_idx, _ = find_peaks(power)
        if len(peak_idx) >= 2:
            top2 = np.sort(power[peak_idx])[-2:]
            if top2[-1] > 0:
                feats["ls_power_ratio_21"] = float(top2[0] / top2[1])
    except Exception:
        pass


def extract_ext_features(detections: list[dict[str, Any]]) -> dict[str, float]:
    """Compute the 50 extended features from a (truncated) detection list.

    Parameters
    ----------
    detections : list of dict
        The same truncated, positive-filtered, MJD-sorted normalized
        detection list consumed by :func:`lightcurve.extract_features`.
        Raw ZTF/LSST dicts are normalized on the fly for compatibility.

    Returns
    -------
    dict mapping every name in ``EXT_FEATURE_NAMES`` to a float
    (NaN when not computable).  Pure function: no I/O, deterministic.
    """
    feats: dict[str, float] = {k: float("nan") for k in EXT_FEATURE_NAMES}
    if not detections:
        return feats

    ndets = [_ensure_normalized(d) for d in detections]
    ndets.sort(key=lambda d: d.get("mjd") or 0)

    # --- Detections with a finite time (cadence sample) ---
    tdets = [d for d in ndets if _isfinite(d.get("mjd"))]
    t_all = np.array([float(d["mjd"]) for d in tdets], dtype=float)

    # --- Detections with finite time AND finite magnitude (photometry sample) ---
    vdets = [d for d in tdets if _isfinite(d.get("mag"))]
    t_v = np.array([float(d["mjd"]) for d in vdets], dtype=float)
    m_v = np.array([float(d["mag"]) for d in vdets], dtype=float)
    n_v = len(vdets)

    # ------------------------------------------------------------------
    # Cadence / sampling (10)
    # ------------------------------------------------------------------
    if len(t_all) >= 1:
        nights = np.floor(t_all)
        _, counts = np.unique(nights, return_counts=True)
        feats["n_nights"] = float(len(counts))
        feats["dets_per_night_max"] = float(counts.max())
    if len(t_all) >= 2:
        gaps = np.diff(t_all)
        feats["dt_mean"] = float(np.mean(gaps))
        feats["dt_median"] = float(np.median(gaps))
        feats["dt_min"] = float(np.min(gaps))
        feats["dt_max"] = float(np.max(gaps))
        feats["dt_std"] = float(np.std(gaps))
        feats["intranight_frac"] = float(np.mean(gaps < _INTRANIGHT_DT))
        t_baseline = float(t_all[-1] - t_all[0])
        if t_baseline > 0:
            feats["max_gap_frac"] = float(np.max(gaps) / t_baseline)
        bands = [d.get("band") for d in tdets]
        feats["band_alternation_rate"] = float(
            np.mean([bands[k] != bands[k + 1] for k in range(len(bands) - 1)])
        )

    # ------------------------------------------------------------------
    # Photometric error / SNR (8)
    # ------------------------------------------------------------------
    edets = [d for d in tdets if _isfinite(d.get("magerr")) and float(d["magerr"]) > 0]
    if edets:
        errs = np.array([float(d["magerr"]) for d in edets], dtype=float)
        feats["magerr_mean"] = float(np.mean(errs))
        feats["magerr_min"] = float(np.min(errs))
        feats["magerr_max"] = float(np.max(errs))
        feats["magerr_last"] = float(errs[-1])
    sdets = [d for d in tdets if _isfinite(d.get("snr"))]
    if sdets:
        snrs = np.array([float(d["snr"]) for d in sdets], dtype=float)
        feats["snr_mean"] = float(np.mean(snrs))
        feats["snr_max"] = float(np.max(snrs))
        feats["snr_last"] = float(snrs[-1])
    # WLS slope on detections with finite mag AND positive magerr
    wdets = [d for d in vdets if _isfinite(d.get("magerr")) and float(d["magerr"]) > 0]
    if len(wdets) >= 2:
        t_w = np.array([float(d["mjd"]) for d in wdets], dtype=float)
        m_w = np.array([float(d["mag"]) for d in wdets], dtype=float)
        e_w = np.array([float(d["magerr"]) for d in wdets], dtype=float)
        # np.polyfit weights multiply residuals: w=1/sigma minimizes chi^2.
        feats["dmag_dt_wls"] = _polyfit_coeff(t_w, m_w, 1, w=1.0 / e_w)

    # ------------------------------------------------------------------
    # Detection significance (4) — ZTF only (LSST dicts lack diffmaglim → NaN)
    # ------------------------------------------------------------------
    limdets = [
        d for d in tdets
        if _isfinite(d.get("diffmaglim")) and _isfinite(d.get("mag"))
    ]
    if limdets:
        sig = np.array(
            [float(d["diffmaglim"]) - float(d["mag"]) for d in limdets], dtype=float
        )
        feats["sig_above_lim_mean"] = float(np.mean(sig))
        feats["sig_above_lim_min"] = float(np.min(sig))
        feats["sig_above_lim_last"] = float(sig[-1])
    lims = [float(d["diffmaglim"]) for d in tdets if _isfinite(d.get("diffmaglim"))]
    if lims:
        feats["diffmaglim_mean"] = float(np.mean(lims))

    # ------------------------------------------------------------------
    # Variability / periodicity proxies (12) — all gated at n>=3
    # ------------------------------------------------------------------
    if n_v >= _VAR_MIN_DETS:
        feats["eta_vn"] = _von_neumann(m_v)
        diffs = np.diff(m_v)
        signs = np.sign(diffs)
        flips = sum(
            1 for k in range(len(signs) - 1) if signs[k] * signs[k + 1] < 0
        )
        feats["flip_rate"] = float(flips) / float(n_v - 2)
        feats["longest_run_frac"] = float(_longest_same_sign_run(signs)) / float(
            len(signs)
        )
        feats["amp_iqr"] = float(
            np.percentile(m_v, 75) - np.percentile(m_v, 25)
        )
        feats["mad_mag"] = float(np.median(np.abs(m_v - np.median(m_v))))
        feats["frac_brighter_first"] = float(np.mean(m_v < m_v[0]))
        if _isfinite(feats["magerr_mean"]) and feats["magerr_mean"] > 0:
            feats["amp_per_err"] = feats["amp_iqr"] / feats["magerr_mean"]

    for band in ("g", "r"):
        bmags = np.array(
            [float(d["mag"]) for d in vdets if d.get("band") == band], dtype=float
        )
        feats[f"eta_vn_{band}"] = _von_neumann(bmags)

    # chi2 around weighted mean + Stetson K need per-detection errors
    cdets = [d for d in vdets if _isfinite(d.get("magerr")) and float(d["magerr"]) > 0]
    if len(cdets) >= _VAR_MIN_DETS:
        m_c = np.array([float(d["mag"]) for d in cdets], dtype=float)
        e_c = np.array([float(d["magerr"]) for d in cdets], dtype=float)
        w = 1.0 / e_c ** 2
        mbar_w = float(np.sum(w * m_c) / np.sum(w))
        feats["chi2_const_dof"] = float(
            np.sum(((m_c - mbar_w) / e_c) ** 2) / (len(cdets) - 1)
        )
        feats["stetson_K"] = _stetson_k(m_c, e_c)

    g_dets = [
        (float(d["mjd"]), float(d["mag"]), float(d["magerr"]))
        for d in vdets
        if d.get("band") == "g" and _isfinite(d.get("magerr")) and float(d["magerr"]) > 0
    ]
    r_dets = [
        (float(d["mjd"]), float(d["mag"]), float(d["magerr"]))
        for d in vdets
        if d.get("band") == "r" and _isfinite(d.get("magerr")) and float(d["magerr"]) > 0
    ]
    feats["stetson_J_gr"] = _stetson_j(_nearest_pairs(g_dets, r_dets, _PAIR_MAX_DT))

    # ------------------------------------------------------------------
    # Lomb-Scargle (4) — best band (most valid mags), only n>=8 there
    # ------------------------------------------------------------------
    band_counts: dict[str, int] = {}
    for d in vdets:
        b = d.get("band")
        if b:
            band_counts[b] = band_counts.get(b, 0) + 1
    if band_counts:
        # Most detections wins; ties broken by canonical band order (u..y).
        best_band = max(
            band_counts,
            key=lambda b: (band_counts[b], -BAND_TO_IDX.get(b, 99)),
        )
        if band_counts[best_band] >= _LS_MIN_DETS:
            bd = [d for d in vdets if d.get("band") == best_band]
            t_b = np.array([float(d["mjd"]) for d in bd], dtype=float)
            m_b = np.array([float(d["mag"]) for d in bd], dtype=float)
            e_list = [
                float(d["magerr"])
                if _isfinite(d.get("magerr")) and float(d["magerr"]) > 0
                else float("nan")
                for d in bd
            ]
            e_b = np.array(e_list, dtype=float)
            if np.unique(t_b).size >= _LS_MIN_DETS:
                _lomb_scargle(t_b, m_b, e_b, feats)

    # ------------------------------------------------------------------
    # Shape / rise (8)
    # ------------------------------------------------------------------
    for band in ("g", "r"):
        bd = [d for d in vdets if d.get("band") == band]
        if len(bd) >= 2:
            t_b = np.array([float(d["mjd"]) for d in bd], dtype=float)
            m_b = np.array([float(d["mag"]) for d in bd], dtype=float)
            feats[f"rise_rate_{band}"] = _polyfit_coeff(t_b, m_b, 1)
    if n_v >= 3:
        feats["dmag_dt_recent3"] = _polyfit_coeff(t_v[-3:], m_v[-3:], 1)
    if n_v >= 4:
        feats["curvature_t2"] = _polyfit_coeff(t_v, m_v, 2)
    if n_v >= 2:
        feats["brightening_frac"] = float(np.mean(np.diff(m_v) < 0))
    if n_v >= 1:
        i_min = int(np.argmin(m_v))
        feats["t_since_brightest"] = float(t_v[-1] - t_v[i_min])
        feats["is_pre_peak"] = 1.0 if m_v[-1] <= float(np.min(m_v)) else 0.0
        t_star = float(t_v[i_min])
        g_near = [
            (abs(float(d["mjd"]) - t_star), float(d["mag"]))
            for d in vdets
            if d.get("band") == "g" and abs(float(d["mjd"]) - t_star) < _PAIR_MAX_DT
        ]
        r_near = [
            (abs(float(d["mjd"]) - t_star), float(d["mag"]))
            for d in vdets
            if d.get("band") == "r" and abs(float(d["mjd"]) - t_star) < _PAIR_MAX_DT
        ]
        if g_near and r_near:
            feats["color_gr_at_brightest"] = min(g_near)[1] - min(r_near)[1]

    # ------------------------------------------------------------------
    # Reference-source context (4) — stamped on the alerts, leak-free
    # ------------------------------------------------------------------
    distnrs = [float(d["distnr"]) for d in ndets if _isfinite(d.get("distnr"))]
    if distnrs:
        feats["distnr_mean"] = float(np.mean(distnrs))
        feats["distnr_min"] = float(np.min(distnrs))
    quals = [float(d["quality"]) for d in ndets if _isfinite(d.get("quality"))]
    if quals:
        feats["quality_min"] = float(np.min(quals))
    corrected = [
        bool(d["corrected"]) for d in ndets
        if "corrected" in d and d["corrected"] is not None
    ]
    if corrected:
        feats["corrected_frac"] = float(np.mean(corrected))

    return feats
