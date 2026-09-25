"""fusion_v11 P2: negative-flux LC features + the canonical truncation helper.

Covers B2 (LSST is_positive = psfFlux>0 and not isNegative), B3 (LSST all-negative
⇒ 0 epochs; ZTF all-negative fallback KEPT) and B10 (one canonical
``truncated_detection_windows`` helper feeding base/EXT positives-only prefixes
and the negatives-included window).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from debass_meta.features.detection import normalize_detection
from debass_meta.features.lightcurve import (
    NEG_FEATURE_NAMES,
    compute_neg_features,
    truncated_detection_windows,
)


# ------------------------------------------------------------------ #
# Fixtures                                                            #
# ------------------------------------------------------------------ #

def _lsst(mjd: float, flux: float, *, is_negative: bool = False) -> dict:
    return {
        "midpointMjdTai": mjd,
        "band": "r",
        "psfFlux": flux,
        "psfFluxErr": 50.0,
        "reliability": 0.9,
        "snr": 10.0,
        "isNegative": is_negative,
    }


def _ztf(mjd: float, mag: float, *, positive: bool = True) -> dict:
    return {
        "mjd": mjd, "fid": 1, "magpsf": mag, "sigmapsf": 0.1,
        "isdiffpos": "t" if positive else "f", "rb": 0.9,
    }


# ------------------------------------------------------------------ #
# B2: LSST is_positive rule                                          #
# ------------------------------------------------------------------ #

def test_lsst_is_positive_requires_positive_flux() -> None:
    assert normalize_detection(_lsst(100.0, 1000.0))["is_positive"] is True
    # Negative flux, isNegative NOT set → the old bug flagged this positive.
    assert normalize_detection(_lsst(100.0, -500.0))["is_positive"] is False
    # Positive flux but flagged negative → not positive.
    assert normalize_detection(_lsst(100.0, 900.0, is_negative=True))["is_positive"] is False
    # Missing flux → not positive.
    assert normalize_detection(
        {"midpointMjdTai": 100.0, "band": "r"})["is_positive"] is False


def test_ztf_is_positive_rule_untouched() -> None:
    assert normalize_detection(_ztf(100.0, 20.0, positive=True))["is_positive"] is True
    assert normalize_detection(_ztf(100.0, 20.0, positive=False))["is_positive"] is False


# ------------------------------------------------------------------ #
# B10: canonical truncation helper                                   #
# ------------------------------------------------------------------ #

def _mjd(d: dict) -> float:
    return float(d.get("mjd") or 0.0)


def test_windows_positives_only_prefix_and_full_window() -> None:
    lc = [
        _lsst(100.0, 1000.0),           # pos
        _lsst(101.0, -500.0),           # neg
        _lsst(102.0, 1200.0),           # pos
        _lsst(103.0, -300.0),           # neg
        _lsst(104.0, 900.0),            # pos
    ]
    windows = truncated_detection_windows(lc, survey="LSST", max_n_det=20)
    assert len(windows) == 3  # 3 positive detections

    for pos_prefix, full_window in windows:
        # pos_prefix: positives only, MJD-sorted.
        assert all(d["is_positive"] for d in pos_prefix)
        assert [_mjd(d) for d in pos_prefix] == sorted(_mjd(d) for d in pos_prefix)
        # full_window: sorted, ends at the Nth positive, only dets <= that time.
        t_nth = _mjd(pos_prefix[-1])
        assert [_mjd(d) for d in full_window] == sorted(_mjd(d) for d in full_window)
        assert all(_mjd(d) <= t_nth + 1e-9 for d in full_window)
        # positives in the window are EXACTLY the prefix (tie-safe tripwire).
        assert sum(1 for d in full_window if d["is_positive"]) == len(pos_prefix)

    # epoch 2 window = [100(pos), 101(neg), 102(pos)]
    _, w2 = windows[1]
    assert [_mjd(d) for d in w2] == [100.0, 101.0, 102.0]


def test_windows_auto_survey_inference() -> None:
    lc = [_lsst(100.0, 1000.0), _lsst(101.0, -500.0)]
    auto = truncated_detection_windows(lc, survey="auto", max_n_det=20)
    explicit = truncated_detection_windows(lc, survey="LSST", max_n_det=20)
    assert len(auto) == len(explicit) == 1


# ------------------------------------------------------------------ #
# B3: survey-gated all-negative fallback                             #
# ------------------------------------------------------------------ #

def test_lsst_all_negative_yields_zero_epochs() -> None:
    lc = [_lsst(100.0, -1000.0), _lsst(101.0, -500.0), _lsst(102.0, -800.0)]
    assert truncated_detection_windows(lc, survey="LSST", max_n_det=20) == []


def test_ztf_all_negative_fallback_kept() -> None:
    lc = [_ztf(100.0, 20.0, positive=False),
          _ztf(101.0, 20.1, positive=False),
          _ztf(102.0, 20.2, positive=False)]
    windows = truncated_detection_windows(lc, survey="ZTF", max_n_det=20)
    # Fallback keeps the epoch rows alive (11% of ZTF LCs depend on it): every
    # detection is used as the "prefix" so the base-51 columns stay identical.
    assert len(windows) == 3
    pos_prefix, full_window = windows[-1]
    assert len(pos_prefix) == 3
    # NEG features count positivity by the is_positive FLAG (spec §2.3,
    # deviation #30): the fallback window is entirely negative, so n_pos_det == 0
    # and n_det_neg == len(window).  These rows are flagged
    # lc_fallback_all_negative == 1 by the builder and excluded from the
    # n_pos_det >= 1 metrics/G2 denominator.
    neg = compute_neg_features(pos_prefix, full_window)
    assert neg["n_pos_det"] == 0.0
    assert neg["n_det_neg"] == 3.0
    assert neg["frac_neg"] == 1.0
    assert neg["neg_run_frac"] == 1.0


# ------------------------------------------------------------------ #
# NEG feature correctness                                            #
# ------------------------------------------------------------------ #

def test_neg_feature_values() -> None:
    lc = [
        _lsst(100.0, 1000.0),   # pos
        _lsst(101.0, -500.0),   # neg
        _lsst(102.0, 1200.0),   # pos
        _lsst(103.0, -300.0),   # neg
        _lsst(104.0, 900.0),    # pos
    ]
    windows = truncated_detection_windows(lc, survey="LSST", max_n_det=20)

    n1 = compute_neg_features(*windows[0])
    assert n1 == {"n_det_neg": 0.0, "frac_neg": 0.0, "n_pos_det": 1.0,
                  "t_since_last_pos": 0.0, "neg_run_frac": 0.0}

    n2 = compute_neg_features(*windows[1])
    assert n2["n_pos_det"] == 2.0
    assert n2["n_det_neg"] == 1.0
    assert n2["frac_neg"] == pytest.approx(1 / 3)
    assert n2["neg_run_frac"] == pytest.approx(1 / 3)
    assert n2["t_since_last_pos"] == pytest.approx(2.0)  # 102 - 100

    n3 = compute_neg_features(*windows[2])
    assert n3["n_pos_det"] == 3.0
    assert n3["n_det_neg"] == 2.0
    assert n3["frac_neg"] == pytest.approx(2 / 5)
    # pos,neg,pos,neg,pos → longest neg run = 1 of 5.
    assert n3["neg_run_frac"] == pytest.approx(1 / 5)
    assert n3["t_since_last_pos"] == pytest.approx(2.0)  # 104 - 102


def test_neg_run_frac_counts_consecutive() -> None:
    lc = [
        _lsst(100.0, 1000.0),   # pos
        _lsst(101.0, -500.0),   # neg
        _lsst(101.5, -400.0),   # neg (consecutive)
        _lsst(102.0, 1200.0),   # pos
    ]
    windows = truncated_detection_windows(lc, survey="LSST", max_n_det=20)
    n2 = compute_neg_features(*windows[1])
    assert n2["n_det_neg"] == 2.0
    # two consecutive negatives in a 4-det window.
    assert n2["neg_run_frac"] == pytest.approx(2 / 4)


def test_neg_feature_names_complete() -> None:
    lc = [_lsst(100.0, 1000.0), _lsst(101.0, -500.0)]
    windows = truncated_detection_windows(lc, survey="LSST", max_n_det=20)
    feats = compute_neg_features(*windows[0])
    assert set(feats) == set(NEG_FEATURE_NAMES)
