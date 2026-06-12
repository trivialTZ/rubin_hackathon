"""Tests for the fusion_v8 extended lightcurve features (lightcurve_ext).

Covers (per the fusion_v8 spec, group G1):
  (a) hand-computed mini-lightcurve assertions (eta_vn, flip_rate, dt_*,
      magerr_*, frac_brighter_first, is_pre_peak, ...);
  (b) the no-leakage test — features at truncation n_det=k are identical
      whether or not future detections exist beyond k;
  (c) NaN-correctness — a single-detection lightcurve yields NaN for every
      feature requiring n>=2 / n>=3;
  (d) LSST detections (no distnr/diffmaglim/corrected) yield NaN for the
      ZTF-only features without crashing;
  (e) Lomb-Scargle engages only at n_det>=8 in the best band and recovers a
      known injected period.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from debass_meta.features.lightcurve import extract_features_at_each_epoch
from debass_meta.features.lightcurve_ext import (
    EXT_FEATURE_NAMES,
    extract_ext_features,
)

_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE_ZTF_JSON = _ROOT / "data" / "lightcurves" / "ZTF17aaaaaan.json"
_DP1_DIR = _ROOT / "data" / "lightcurves" / "dp1"

_LN10 = math.log(10.0)


# ---------------------------------------------------------------------------
# Synthetic lightcurve builders
# ---------------------------------------------------------------------------

def _ztf_det(
    mjd: float,
    fid: int,
    magpsf: float,
    sigmapsf: float = 0.05,
    *,
    diffmaglim: float = 20.5,
    distnr: float = 1.0,
    rb: float = 0.9,
    corrected: bool = True,
) -> dict:
    return {
        "mjd": mjd,
        "fid": fid,
        "magpsf": magpsf,
        "sigmapsf": sigmapsf,
        "isdiffpos": "t",
        "diffmaglim": diffmaglim,
        "distnr": distnr,
        "rb": rb,
        "corrected": corrected,
    }


def _lsst_det(
    mjd: float,
    band: str,
    flux_njy: float,
    fluxerr_njy: float = 200.0,
    *,
    reliability: float = 0.95,
) -> dict:
    return {
        "midpointMjdTai": mjd,
        "band": band,
        "psfFlux": flux_njy,
        "psfFluxErr": fluxerr_njy,
        "reliability": reliability,
        "snr": flux_njy / fluxerr_njy,
    }


def _mini_lightcurve() -> list[dict]:
    """4-detection ZTF lightcurve with hand-computable feature values."""
    return [
        _ztf_det(100.0, 1, 19.0, 0.10, diffmaglim=20.5, distnr=1.0, rb=0.90,
                 corrected=True),
        _ztf_det(101.0, 2, 18.5, 0.08, diffmaglim=20.4, distnr=1.2, rb=0.95,
                 corrected=True),
        _ztf_det(101.2, 1, 18.2, 0.05, diffmaglim=20.6, distnr=0.8, rb=0.99,
                 corrected=False),
        _ztf_det(103.0, 2, 18.4, 0.06, diffmaglim=20.3, distnr=1.1, rb=0.97,
                 corrected=True),
    ]


def _long_lightcurve(n: int = 12) -> list[dict]:
    """Deterministic n-detection two-band ZTF lightcurve for leakage tests."""
    rng = np.random.default_rng(42)
    dets = []
    mjd = 60000.0
    for k in range(n):
        mjd += float(rng.uniform(0.2, 2.5))
        fid = 1 if k % 2 == 0 else 2
        mag = 19.5 - 0.15 * k + float(rng.normal(0, 0.03))
        dets.append(_ztf_det(mjd, fid, mag, 0.04 + 0.01 * (k % 3)))
    return dets


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

def test_ext_feature_names_contract() -> None:
    assert len(EXT_FEATURE_NAMES) == 50
    assert len(set(EXT_FEATURE_NAMES)) == 50
    feats = extract_ext_features(_mini_lightcurve())
    assert set(feats.keys()) == set(EXT_FEATURE_NAMES)
    # Spot-check the pinned group ordering (design §1.2 listing order).
    assert EXT_FEATURE_NAMES[0] == "dt_mean"
    assert EXT_FEATURE_NAMES[9] == "band_alternation_rate"
    assert EXT_FEATURE_NAMES[10] == "magerr_mean"
    assert EXT_FEATURE_NAMES[17] == "dmag_dt_wls"
    assert EXT_FEATURE_NAMES[18] == "sig_above_lim_mean"
    assert EXT_FEATURE_NAMES[22] == "eta_vn"
    assert EXT_FEATURE_NAMES[34] == "ls_power_peak"
    assert EXT_FEATURE_NAMES[38] == "rise_rate_g"
    assert EXT_FEATURE_NAMES[46] == "distnr_mean"
    assert EXT_FEATURE_NAMES[49] == "corrected_frac"


def test_empty_lightcurve_all_nan() -> None:
    feats = extract_ext_features([])
    assert all(math.isnan(v) for v in feats.values())


# ---------------------------------------------------------------------------
# (a) Hand-computed mini-lightcurve
# ---------------------------------------------------------------------------

def test_hand_computed_cadence() -> None:
    feats = extract_ext_features(_mini_lightcurve())
    # gaps: 1.0, 0.2, 1.8
    assert feats["dt_mean"] == pytest.approx(1.0)
    assert feats["dt_median"] == pytest.approx(1.0)
    assert feats["dt_min"] == pytest.approx(0.2)
    assert feats["dt_max"] == pytest.approx(1.8)
    assert feats["dt_std"] == pytest.approx(math.sqrt((0.0 + 0.64 + 0.64) / 3.0))
    # nights: floor = 100, 101, 101, 103
    assert feats["n_nights"] == pytest.approx(3.0)
    assert feats["dets_per_night_max"] == pytest.approx(2.0)
    assert feats["intranight_frac"] == pytest.approx(1.0 / 3.0)
    assert feats["max_gap_frac"] == pytest.approx(1.8 / 3.0)
    # bands g, r, g, r → every consecutive pair alternates
    assert feats["band_alternation_rate"] == pytest.approx(1.0)


def test_hand_computed_magerr_snr() -> None:
    feats = extract_ext_features(_mini_lightcurve())
    # magerrs: 0.10, 0.08, 0.05, 0.06
    assert feats["magerr_mean"] == pytest.approx(0.0725)
    assert feats["magerr_min"] == pytest.approx(0.05)
    assert feats["magerr_max"] == pytest.approx(0.10)
    assert feats["magerr_last"] == pytest.approx(0.06)
    # snr = 2.5 / (ln10 * magerr)  (≈ 1.0857 / magerr)
    assert feats["snr_last"] == pytest.approx(2.5 / (_LN10 * 0.06))
    assert feats["snr_max"] == pytest.approx(2.5 / (_LN10 * 0.05))
    expected_snr_mean = np.mean([2.5 / (_LN10 * e) for e in (0.10, 0.08, 0.05, 0.06)])
    assert feats["snr_mean"] == pytest.approx(expected_snr_mean)


def test_hand_computed_wls_slope() -> None:
    feats = extract_ext_features(_mini_lightcurve())
    # Independent closed-form WLS (weights 1/sigma^2) of mag vs time.
    t = np.array([100.0, 101.0, 101.2, 103.0])
    m = np.array([19.0, 18.5, 18.2, 18.4])
    e = np.array([0.10, 0.08, 0.05, 0.06])
    w = 1.0 / e ** 2
    sw, swx, swy = w.sum(), (w * t).sum(), (w * m).sum()
    swxx, swxy = (w * t * t).sum(), (w * t * m).sum()
    slope = (sw * swxy - swx * swy) / (sw * swxx - swx ** 2)
    assert feats["dmag_dt_wls"] == pytest.approx(slope, rel=1e-6)


def test_hand_computed_detection_significance() -> None:
    feats = extract_ext_features(_mini_lightcurve())
    # diffmaglim - magpsf: 1.5, 1.9, 2.4, 1.9
    assert feats["sig_above_lim_mean"] == pytest.approx(1.925)
    assert feats["sig_above_lim_min"] == pytest.approx(1.5)
    assert feats["sig_above_lim_last"] == pytest.approx(1.9)
    assert feats["diffmaglim_mean"] == pytest.approx(20.45)


def test_hand_computed_variability() -> None:
    feats = extract_ext_features(_mini_lightcurve())
    # mags (time order): 19.0, 18.5, 18.2, 18.4
    # diffs: -0.5, -0.3, +0.2 → mean(diff^2) = 0.38/3
    # var(ddof=1) = 0.3475/3 → eta_vn = 0.38/0.3475
    assert feats["eta_vn"] == pytest.approx(0.38 / 0.3475)
    # diff signs (-,-,+): one flip over n-2=2 comparisons
    assert feats["flip_rate"] == pytest.approx(0.5)
    # longest same-sign run = 2 of 3 diffs
    assert feats["longest_run_frac"] == pytest.approx(2.0 / 3.0)
    # 3 of 4 dets brighter (mag <) than the first (19.0)
    assert feats["frac_brighter_first"] == pytest.approx(0.75)
    # median = 18.45 → |devs| = 0.55, 0.05, 0.25, 0.05 → MAD = 0.15
    assert feats["mad_mag"] == pytest.approx(0.15)
    assert feats["amp_iqr"] == pytest.approx(0.275)
    assert feats["amp_per_err"] == pytest.approx(0.275 / 0.0725)
    # chi2 around the inverse-variance weighted mean, independent computation
    m = np.array([19.0, 18.5, 18.2, 18.4])
    e = np.array([0.10, 0.08, 0.05, 0.06])
    w = 1.0 / e ** 2
    mbar = (w * m).sum() / w.sum()
    chi2 = (((m - mbar) / e) ** 2).sum() / 3.0
    assert feats["chi2_const_dof"] == pytest.approx(chi2)


def test_hand_computed_shape() -> None:
    feats = extract_ext_features(_mini_lightcurve())
    # 2 of 3 consecutive pairs brighten (mag decreases)
    assert feats["brightening_frac"] == pytest.approx(2.0 / 3.0)
    # brightest (18.2) at mjd 101.2, last det at 103.0
    assert feats["t_since_brightest"] == pytest.approx(1.8)
    # last det (18.4) is NOT the brightest (18.2)
    assert feats["is_pre_peak"] == pytest.approx(0.0)
    # brightest epoch 101.2: nearest g = 18.2 (dt 0), nearest r = 18.5 (dt 0.2)
    assert feats["color_gr_at_brightest"] == pytest.approx(18.2 - 18.5)
    # curvature: only 4 dets → quadratic fit defined; check finite
    assert math.isfinite(feats["curvature_t2"])


def test_is_pre_peak_when_last_is_brightest() -> None:
    dets = [
        _ztf_det(100.0, 1, 19.0, 0.05),
        _ztf_det(101.0, 1, 18.5, 0.05),
        _ztf_det(102.0, 1, 18.0, 0.05),
    ]
    feats = extract_ext_features(dets)
    assert feats["is_pre_peak"] == pytest.approx(1.0)
    assert feats["t_since_brightest"] == pytest.approx(0.0)
    # strictly monotone brightening: no flips, run covers all diffs
    assert feats["flip_rate"] == pytest.approx(0.0)
    assert feats["longest_run_frac"] == pytest.approx(1.0)
    assert feats["brightening_frac"] == pytest.approx(1.0)


def test_hand_computed_reference_context() -> None:
    feats = extract_ext_features(_mini_lightcurve())
    assert feats["distnr_mean"] == pytest.approx((1.0 + 1.2 + 0.8 + 1.1) / 4.0)
    assert feats["distnr_min"] == pytest.approx(0.8)
    assert feats["quality_min"] == pytest.approx(0.90)
    assert feats["corrected_frac"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# (b) No-leakage: prefix features unchanged by future detections
# ---------------------------------------------------------------------------

def test_no_future_leakage() -> None:
    """Features at truncation n_det=k must be identical whether or not the
    lightcurve has detections beyond k (the gold builder truncates first)."""
    full = _long_lightcurve(12)
    short = full[:5]

    # Emulate the builder: extract_features_at_each_epoch truncation order.
    epoch_rows_short = extract_features_at_each_epoch(short)
    epoch_rows_full = extract_features_at_each_epoch(full)
    assert len(epoch_rows_short) == 5
    assert len(epoch_rows_full) == 12

    for k in range(1, 6):
        feats_short = extract_ext_features(short[:k])
        feats_full_prefix = extract_ext_features(full[:k])
        for name in EXT_FEATURE_NAMES:
            a, b = feats_short[name], feats_full_prefix[name]
            assert (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b), (
                f"leakage at n_det={k}, feature {name}: {a} vs {b}"
            )

    # Sanity: later epochs DO change n-dependent features (not frozen).
    f5 = extract_ext_features(full[:5])
    f12 = extract_ext_features(full[:12])
    assert f5["dt_mean"] != pytest.approx(f12["dt_mean"])


# ---------------------------------------------------------------------------
# (c) NaN-correctness for short lightcurves
# ---------------------------------------------------------------------------

_NEEDS_N2 = [
    "dt_mean", "dt_median", "dt_min", "dt_max", "dt_std",
    "intranight_frac", "max_gap_frac", "band_alternation_rate",
    "dmag_dt_wls", "brightening_frac",
    "rise_rate_g", "rise_rate_r",
]
_NEEDS_N3 = [
    "eta_vn", "eta_vn_g", "eta_vn_r", "flip_rate", "longest_run_frac",
    "chi2_const_dof", "stetson_J_gr", "stetson_K",
    "amp_iqr", "mad_mag", "frac_brighter_first", "amp_per_err",
    "dmag_dt_recent3",
]
_NEEDS_MORE = [
    "curvature_t2",                                          # n>=4
    "ls_power_peak", "ls_logperiod_peak", "ls_power_ratio_21", "ls_n_used",  # n>=8
]


def test_single_detection_nan_correctness() -> None:
    feats = extract_ext_features([_ztf_det(100.0, 1, 19.0, 0.10)])
    for name in _NEEDS_N2 + _NEEDS_N3 + _NEEDS_MORE:
        assert math.isnan(feats[name]), f"{name} should be NaN at n_det=1"
    # color needs both bands within 3 d → NaN with a single g det
    assert math.isnan(feats["color_gr_at_brightest"])
    # Computable at n=1:
    assert feats["n_nights"] == pytest.approx(1.0)
    assert feats["dets_per_night_max"] == pytest.approx(1.0)
    assert feats["magerr_mean"] == pytest.approx(0.10)
    assert feats["magerr_last"] == pytest.approx(0.10)
    assert feats["snr_last"] == pytest.approx(2.5 / (_LN10 * 0.10))
    assert feats["sig_above_lim_last"] == pytest.approx(20.5 - 19.0)
    assert feats["diffmaglim_mean"] == pytest.approx(20.5)
    assert feats["t_since_brightest"] == pytest.approx(0.0)
    assert feats["is_pre_peak"] == pytest.approx(1.0)
    assert feats["distnr_mean"] == pytest.approx(1.0)
    assert feats["quality_min"] == pytest.approx(0.9)
    assert feats["corrected_frac"] == pytest.approx(1.0)


def test_two_detections_n3_features_nan() -> None:
    dets = [_ztf_det(100.0, 1, 19.0, 0.10), _ztf_det(101.0, 2, 18.5, 0.08)]
    feats = extract_ext_features(dets)
    for name in _NEEDS_N3 + _NEEDS_MORE:
        assert math.isnan(feats[name]), f"{name} should be NaN at n_det=2"
    # n>=2 features engage
    assert feats["dt_mean"] == pytest.approx(1.0)
    assert feats["dt_std"] == pytest.approx(0.0)
    assert feats["band_alternation_rate"] == pytest.approx(1.0)
    assert feats["brightening_frac"] == pytest.approx(1.0)
    assert math.isfinite(feats["dmag_dt_wls"])


# ---------------------------------------------------------------------------
# (d) LSST detections: ZTF-only features NaN, no crash
# ---------------------------------------------------------------------------

_ZTF_ONLY = [
    "sig_above_lim_mean", "sig_above_lim_min", "sig_above_lim_last",
    "diffmaglim_mean", "distnr_mean", "distnr_min", "corrected_frac",
]


def test_lsst_detections_ztf_only_features_nan() -> None:
    dets = [
        _lsst_det(60630.0, "g", 3000.0, 200.0, reliability=0.99),
        _lsst_det(60631.0, "r", 3500.0, 210.0, reliability=0.97),
        _lsst_det(60632.0, "g", 4200.0, 190.0, reliability=0.95),
        _lsst_det(60633.5, "r", 4800.0, 205.0, reliability=0.98),
    ]
    feats = extract_ext_features(dets)
    for name in _ZTF_ONLY:
        assert math.isnan(feats[name]), f"{name} should be NaN for LSST"
    # Survey-agnostic features still engage
    assert feats["quality_min"] == pytest.approx(0.95)
    assert feats["dt_mean"] == pytest.approx(3.5 / 3.0)
    assert math.isfinite(feats["magerr_mean"])
    assert math.isfinite(feats["snr_last"])
    assert math.isfinite(feats["eta_vn"])
    # rising flux → brightening mags
    assert feats["brightening_frac"] == pytest.approx(1.0)
    assert feats["is_pre_peak"] == pytest.approx(1.0)


def test_lsst_negative_flux_does_not_crash() -> None:
    dets = [
        _lsst_det(60630.0, "g", 3000.0),
        _lsst_det(60631.0, "g", -50.0),   # negative flux → NaN mag
        _lsst_det(60632.0, "g", 3200.0),
    ]
    feats = extract_ext_features(dets)
    assert set(feats.keys()) == set(EXT_FEATURE_NAMES)
    # Only 2 valid mags → variability group NaN
    assert math.isnan(feats["eta_vn"])
    # cadence uses all 3 timestamped detections
    assert feats["n_nights"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# (e) Lomb-Scargle gating and period recovery
# ---------------------------------------------------------------------------

def _periodic_lightcurve(n: int, period: float = 1.7) -> list[dict]:
    rng = np.random.default_rng(42)
    times = np.sort(60000.0 + rng.uniform(0.0, 25.0, n))
    dets = []
    for t in times:
        mag = 18.0 + 0.6 * math.sin(2.0 * math.pi * t / period)
        dets.append(_ztf_det(float(t), 1, mag, 0.02))
    return dets


def test_lomb_scargle_gated_below_8() -> None:
    feats = extract_ext_features(_periodic_lightcurve(7))
    for name in ("ls_power_peak", "ls_logperiod_peak", "ls_power_ratio_21",
                 "ls_n_used"):
        assert math.isnan(feats[name]), f"{name} should be NaN at n_det=7"


def test_lomb_scargle_recovers_injected_period() -> None:
    period = 1.7
    feats = extract_ext_features(_periodic_lightcurve(30, period))
    assert feats["ls_n_used"] == pytest.approx(30.0)
    assert feats["ls_power_peak"] > 0.5  # strong periodic signal
    recovered = 10.0 ** feats["ls_logperiod_peak"]
    assert abs(recovered - period) / period < 0.05
    if math.isfinite(feats["ls_power_ratio_21"]):
        assert 0.0 <= feats["ls_power_ratio_21"] <= 1.0


def test_lomb_scargle_best_band_gate_is_per_band() -> None:
    """9 total dets but only 5 in the best band → LS stays NaN."""
    dets = _periodic_lightcurve(5)
    for k, t in enumerate((60001.3, 60004.7, 60009.1, 60013.9)):
        dets.append(_ztf_det(t, 2, 18.3 + 0.05 * k, 0.03))
    feats = extract_ext_features(dets)
    assert math.isnan(feats["ls_power_peak"])
    assert math.isnan(feats["ls_n_used"])


# ---------------------------------------------------------------------------
# Real-data smoke tests (skip when data/ absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _SAMPLE_ZTF_JSON.exists(), reason="sample ZTF LC absent")
def test_real_ztf_lightcurve_smoke() -> None:
    dets = json.loads(_SAMPLE_ZTF_JSON.read_text())
    feats = extract_ext_features(dets[:10])
    assert set(feats.keys()) == set(EXT_FEATURE_NAMES)
    assert math.isfinite(feats["dt_mean"])
    assert math.isfinite(feats["distnr_mean"])      # ZTF carries distnr
    assert math.isfinite(feats["diffmaglim_mean"])  # and diffmaglim


@pytest.mark.skipif(
    not (_DP1_DIR.exists() and any(_DP1_DIR.glob("*.parquet"))),
    reason="DP1 lightcurves absent",
)
def test_real_dp1_lightcurve_smoke() -> None:
    pd = pytest.importorskip("pandas")
    path = sorted(_DP1_DIR.glob("*.parquet"))[0]
    df = pd.read_parquet(path)
    dets = df.to_dict("records")
    feats = extract_ext_features(dets[:10])
    assert set(feats.keys()) == set(EXT_FEATURE_NAMES)
    for name in _ZTF_ONLY:
        assert math.isnan(feats[name]), f"{name} should be NaN on DP1"
    assert math.isfinite(feats["quality_min"])  # reliability present
