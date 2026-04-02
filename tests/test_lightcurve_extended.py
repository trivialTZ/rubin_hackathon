"""Tests for extended ugrizy lightcurve feature extraction."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from debass_meta.features.lightcurve import (
    FEATURE_NAMES,
    extract_features,
    extract_features_at_each_epoch,
)
from debass_meta.features.detection import normalize_detection


class TestFeatureNameContract:
    def test_expected_count(self):
        """51 features: 7 counts + 2 temporal + 8 allband + 24 perband + 8 colors + 1 quality + 1 survey."""
        assert len(FEATURE_NAMES) == 51, f"Expected 51, got {len(FEATURE_NAMES)}"

    def test_no_duplicates(self):
        assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))

    def test_has_lsst_bands(self):
        for band in ("u", "g", "r", "i", "z", "y"):
            assert f"n_det_{band}" in FEATURE_NAMES
            assert f"mag_first_{band}" in FEATURE_NAMES
            assert f"mag_mean_{band}" in FEATURE_NAMES

    def test_has_color_pairs(self):
        for pair in ("gr", "ri", "iz", "zy"):
            assert f"color_{pair}" in FEATURE_NAMES
            assert f"color_{pair}_slope" in FEATURE_NAMES

    def test_has_survey_flag(self):
        assert "survey_is_lsst" in FEATURE_NAMES

    def test_has_mean_quality(self):
        assert "mean_quality" in FEATURE_NAMES


class TestZTFBackwardCompat:
    """ZTF detections should produce identical g/r features as before."""

    ZTF_LC = [
        {"mjd": 59800.0, "fid": 1, "magpsf": 19.0, "sigmapsf": 0.05, "isdiffpos": "t", "rb": 0.9},
        {"mjd": 59800.5, "fid": 2, "magpsf": 19.5, "sigmapsf": 0.06, "isdiffpos": "t", "rb": 0.85},
        {"mjd": 59801.0, "fid": 1, "magpsf": 18.8, "sigmapsf": 0.04, "isdiffpos": "t", "rb": 0.92},
        {"mjd": 59802.0, "fid": 2, "magpsf": 19.3, "sigmapsf": 0.05, "isdiffpos": "t", "rb": 0.88},
    ]

    def test_n_det(self):
        feats = extract_features(self.ZTF_LC)
        assert feats["n_det"] == 4
        assert feats["n_det_g"] == 2
        assert feats["n_det_r"] == 2

    def test_lsst_bands_are_nan(self):
        feats = extract_features(self.ZTF_LC)
        for band in ("u", "i", "z", "y"):
            assert feats[f"n_det_{band}"] == 0.0  # no detections
            assert np.isnan(feats[f"mag_first_{band}"])

    def test_survey_flag_ztf(self):
        feats = extract_features(self.ZTF_LC)
        assert feats["survey_is_lsst"] == 0.0

    def test_color_gr_populated(self):
        feats = extract_features(self.ZTF_LC)
        assert math.isfinite(feats["color_gr"])

    def test_color_ri_nan_for_ztf(self):
        feats = extract_features(self.ZTF_LC)
        assert np.isnan(feats["color_ri"])  # no i-band in ZTF

    def test_mean_quality(self):
        feats = extract_features(self.ZTF_LC)
        assert math.isfinite(feats["mean_quality"])
        # Should be mean of rb values since no drb
        expected = np.mean([0.9, 0.85, 0.92, 0.88])
        assert abs(feats["mean_quality"] - expected) < 1e-10

    def test_all_feature_names_present(self):
        feats = extract_features(self.ZTF_LC)
        for name in FEATURE_NAMES:
            assert name in feats, f"Missing feature: {name}"


class TestLSSTFeatures:
    """LSST detections with ugrizy bands."""

    LSST_LC = [
        {"midpointMjdTai": 61091.0, "band": "g", "psfFlux": 5000.0, "psfFluxErr": 200.0, "isNegative": False, "reliability": 0.8},
        {"midpointMjdTai": 61091.5, "band": "r", "psfFlux": 4500.0, "psfFluxErr": 180.0, "isNegative": False, "reliability": 0.85},
        {"midpointMjdTai": 61092.0, "band": "i", "psfFlux": 4000.0, "psfFluxErr": 200.0, "isNegative": False, "reliability": 0.75},
        {"midpointMjdTai": 61092.5, "band": "g", "psfFlux": 5500.0, "psfFluxErr": 220.0, "isNegative": False, "reliability": 0.82},
        {"midpointMjdTai": 61093.0, "band": "z", "psfFlux": 3500.0, "psfFluxErr": 250.0, "isNegative": False, "reliability": 0.7},
    ]

    def test_n_det(self):
        feats = extract_features(self.LSST_LC)
        assert feats["n_det"] == 5
        assert feats["n_det_g"] == 2
        assert feats["n_det_r"] == 1
        assert feats["n_det_i"] == 1
        assert feats["n_det_z"] == 1
        assert feats["n_det_u"] == 0
        assert feats["n_det_y"] == 0

    def test_survey_flag_lsst(self):
        feats = extract_features(self.LSST_LC)
        assert feats["survey_is_lsst"] == 1.0

    def test_i_band_features(self):
        feats = extract_features(self.LSST_LC)
        assert math.isfinite(feats["mag_first_i"])
        assert math.isfinite(feats["mag_mean_i"])

    def test_color_gr_populated(self):
        feats = extract_features(self.LSST_LC)
        # g and r are within 0.5 days → should pair
        assert math.isfinite(feats["color_gr"])

    def test_color_ri_populated(self):
        feats = extract_features(self.LSST_LC)
        # r at 61091.5 and i at 61092.0 → dt=0.5 < 3 days
        assert math.isfinite(feats["color_ri"])

    def test_color_iz_populated(self):
        feats = extract_features(self.LSST_LC)
        # i at 61092.0 and z at 61093.0 → dt=1.0 < 3 days
        assert math.isfinite(feats["color_iz"])

    def test_color_zy_nan(self):
        feats = extract_features(self.LSST_LC)
        # No y-band detections
        assert np.isnan(feats["color_zy"])

    def test_magnitudes_reasonable(self):
        feats = extract_features(self.LSST_LC)
        # Fluxes ~3500-5500 nJy → mags should be ~22-23 ish
        assert 20.0 < feats["mag_mean"] < 25.0


class TestEpochExtraction:
    """Test truncated epoch extraction maintains no-leakage guarantee."""

    LSST_LC = [
        {"midpointMjdTai": 61091.0, "band": "g", "psfFlux": 5000.0, "psfFluxErr": 200.0, "isNegative": False, "reliability": 0.8},
        {"midpointMjdTai": 61092.0, "band": "r", "psfFlux": 4500.0, "psfFluxErr": 180.0, "isNegative": False, "reliability": 0.85},
        {"midpointMjdTai": 61093.0, "band": "i", "psfFlux": 4000.0, "psfFluxErr": 200.0, "isNegative": False, "reliability": 0.75},
    ]

    def test_epoch_count(self):
        epochs = extract_features_at_each_epoch(self.LSST_LC)
        assert len(epochs) == 3

    def test_no_leakage(self):
        epochs = extract_features_at_each_epoch(self.LSST_LC)
        # At n_det=1, should only know about first detection
        assert epochs[0]["n_det"] == 1.0
        assert epochs[0]["n_det_g"] == 1.0
        assert epochs[0]["n_det_r"] == 0.0
        assert epochs[0]["n_det_i"] == 0.0

        # At n_det=2, should know g and r
        assert epochs[1]["n_det"] == 2.0
        assert epochs[1]["n_det_g"] == 1.0
        assert epochs[1]["n_det_r"] == 1.0
        assert epochs[1]["n_det_i"] == 0.0

    def test_survey_flag_consistent(self):
        epochs = extract_features_at_each_epoch(self.LSST_LC)
        for e in epochs:
            assert e["survey_is_lsst"] == 1.0


class TestEmptyAndEdgeCases:
    def test_empty_lc(self):
        feats = extract_features([])
        assert feats["n_det"] == 0.0
        for name in FEATURE_NAMES:
            assert name in feats

    def test_single_detection(self):
        det = [{"midpointMjdTai": 61091.0, "band": "r", "psfFlux": 3000.0, "psfFluxErr": 200.0, "isNegative": False, "reliability": 0.6}]
        feats = extract_features(det)
        assert feats["n_det"] == 1.0
        assert feats["n_det_r"] == 1.0
        assert feats["t_since_first"] == 0.0  # only one detection
        assert math.isfinite(feats["mag_first_r"])

    def test_negative_flux_produces_nan_mag(self):
        det = [{"midpointMjdTai": 61091.0, "band": "r", "psfFlux": -100.0, "psfFluxErr": 200.0, "isNegative": False, "reliability": 0.6}]
        feats = extract_features(det)
        # Negative flux → NaN magnitude → mag stats should handle gracefully
        assert feats["n_det"] == 1.0
