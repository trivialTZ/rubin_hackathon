"""fusion_v11 P2 — G5-style leakage + positive-only epoch tripwires.

Appending FUTURE negative detections (mjd beyond the Nth positive) must not
change ANY feature at a fixed ``n_pos_det``; and ``n_det == n_pos_det`` must hold
on every built row (the positive-only epoch contract, B3).
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from debass_meta.features.lightcurve import (
    compute_neg_features,
    truncated_detection_windows,
)


def _load_snapmod():
    path = _ROOT / "scripts" / "build_snapshots_fusion.py"
    spec = importlib.util.spec_from_file_location("_v11_snap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


snapmod = _load_snapmod()

# Numeric object ids infer as LSST dia object ids.
_LSST_SHORT = "170028485647532096"
_LSST_LONG = "170028485647532097"
_LSST_ALLNEG = "170028485647532098"


def _lsst(mjd: float, flux: float, *, is_negative: bool = False) -> dict:
    return {
        "midpointMjdTai": mjd, "band": "r", "psfFlux": flux,
        "psfFluxErr": 50.0, "reliability": 0.9, "snr": 10.0,
        "isNegative": is_negative,
    }


# ------------------------------------------------------------------ #
# Helper-level future-negative invariance                            #
# ------------------------------------------------------------------ #

def test_future_negatives_do_not_change_windows() -> None:
    short = [_lsst(100.0, 1000.0), _lsst(101.0, -500.0),
             _lsst(102.0, 1200.0), _lsst(104.0, 900.0)]
    # Same detections + FUTURE negatives beyond the last positive (104).
    long = short + [_lsst(105.0, -700.0), _lsst(106.0, -650.0)]

    w_short = truncated_detection_windows(short, survey="LSST", max_n_det=20)
    w_long = truncated_detection_windows(long, survey="LSST", max_n_det=20)
    assert len(w_short) == len(w_long) == 3

    for (ps, fs), (pl, fl) in zip(w_short, w_long):
        assert [d["mjd"] for d in ps] == [d["mjd"] for d in pl]
        assert [d["mjd"] for d in fs] == [d["mjd"] for d in fl]
        assert compute_neg_features(ps, fs) == compute_neg_features(pl, fl)


# ------------------------------------------------------------------ #
# Builder-level G5 + tripwire                                        #
# ------------------------------------------------------------------ #

def _write(lc_dir: Path, oid: str, dets: list[dict]) -> None:
    with open(lc_dir / f"{oid}.json", "w") as fh:
        json.dump(dets, fh)


def test_future_negatives_do_not_change_built_rows(tmp_path: Path) -> None:
    lc_dir = tmp_path / "lc"
    lc_dir.mkdir()
    short = [_lsst(100.0, 1000.0), _lsst(101.0, -500.0),
             _lsst(102.0, 1200.0), _lsst(104.0, 900.0)]
    long = short + [_lsst(105.0, -700.0), _lsst(106.0, -650.0)]
    _write(lc_dir, _LSST_SHORT, short)
    _write(lc_dir, _LSST_LONG, long)

    _, _, short_rows = snapmod._extract_object_rows(
        _LSST_SHORT, lc_dir_str=str(lc_dir), max_n_det=20)
    _, _, long_rows = snapmod._extract_object_rows(
        _LSST_LONG, lc_dir_str=str(lc_dir), max_n_det=20)

    assert len(short_rows) == len(long_rows) == 3
    for a, b in zip(short_rows, long_rows):
        assert set(a) == set(b)
        for key in a:
            va, vb = a[key], b[key]
            if isinstance(va, float) and isinstance(vb, float):
                assert (math.isnan(va) and math.isnan(vb)) or va == pytest.approx(vb), key
            else:
                assert va == vb, key


def test_built_rows_satisfy_n_pos_det_tripwire(tmp_path: Path) -> None:
    lc_dir = tmp_path / "lc"
    lc_dir.mkdir()
    dets = [_lsst(100.0, 1000.0), _lsst(101.0, -500.0),
            _lsst(102.0, 1200.0), _lsst(103.0, -300.0), _lsst(104.0, 900.0)]
    _write(lc_dir, _LSST_SHORT, dets)

    _, _, rows = snapmod._extract_object_rows(
        _LSST_SHORT, lc_dir_str=str(lc_dir), max_n_det=20)
    assert len(rows) == 3  # 3 positive detections
    for row in rows:
        assert int(row["n_pos_det"]) == int(row["n_det"])
        assert row["lc_fallback_all_negative"] == 0.0  # not a fallback object
    # NEG features grow as negatives accumulate within the window.
    assert [int(r["n_det_neg"]) for r in rows] == [0, 1, 2]


def test_lsst_all_negative_builds_zero_rows(tmp_path: Path) -> None:
    lc_dir = tmp_path / "lc"
    lc_dir.mkdir()
    dets = [_lsst(100.0, -1000.0), _lsst(101.0, -500.0), _lsst(102.0, -800.0)]
    _write(lc_dir, _LSST_ALLNEG, dets)

    oid, lc_source, rows = snapmod._extract_object_rows(
        _LSST_ALLNEG, lc_dir_str=str(lc_dir), max_n_det=20)
    assert rows == []
    # No positives ⇒ the LSST fallback is removed ⇒ not fallback-dependent.
    assert lc_source["all_negative_fallback"] is False


def test_missing_is_positive_tripwire() -> None:
    # A detection dict that has never been normalized lacks 'is_positive'.
    pos = [{"mjd": 100.0, "is_positive": True}]
    bad_window = [{"mjd": 100.0}]  # no is_positive key
    with pytest.raises(AssertionError, match="is_positive"):
        snapmod._neg_features_or_nan("obj", 1, pos, bad_window)


def test_n_pos_det_zero_on_fallback() -> None:
    # ZTF all-negative fallback (spec §2.3, deviation #30): positivity is read
    # from the is_positive flag, so the entirely-negative window has n_pos_det==0
    # and n_det_neg == len(window).  n_det != n_pos_det here — the row is scoped
    # OUT of the tripwire by the lc_fallback_all_negative flag.
    lc = [{"mjd": 100.0, "fid": 1, "magpsf": 20.0, "sigmapsf": 0.1, "isdiffpos": "f"},
          {"mjd": 101.0, "fid": 1, "magpsf": 20.1, "sigmapsf": 0.1, "isdiffpos": "f"}]
    windows = truncated_detection_windows(lc, survey="ZTF", max_n_det=20)
    assert len(windows) == 2
    neg = compute_neg_features(*windows[-1])
    assert neg["n_pos_det"] == 0.0
    assert neg["n_det_neg"] == 2.0


def test_extracted_fallback_rows_kept_and_flagged(tmp_path: Path) -> None:
    # End-to-end via the worker: a ZTF all-negative object still yields rows
    # (fallback kept), reported by lc_source["all_negative_fallback"] and the
    # per-row lc_fallback_all_negative flag; those rows carry n_pos_det == 0 and
    # are excluded from the n_det == n_pos_det tripwire (deviation #30).
    lc_dir = tmp_path / "lc"
    lc_dir.mkdir()
    dets = [{"mjd": 100.0, "fid": 1, "magpsf": 20.0, "sigmapsf": 0.1, "isdiffpos": "f"},
            {"mjd": 101.0, "fid": 1, "magpsf": 20.1, "sigmapsf": 0.1, "isdiffpos": "f"}]
    _write(lc_dir, "ZTFfakeALLNEG", dets)
    _oid, lc_source, rows = snapmod._extract_object_rows(
        "ZTFfakeALLNEG", lc_dir_str=str(lc_dir), max_n_det=20)
    assert len(rows) == 2
    assert lc_source["all_negative_fallback"] is True
    for row in rows:
        assert row["lc_fallback_all_negative"] == 1.0
        assert int(row["n_pos_det"]) == 0
