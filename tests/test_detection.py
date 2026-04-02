"""Tests for the survey-aware detection normalization module."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from debass_meta.features.detection import (
    CANONICAL_BANDS,
    ZTF_FID_TO_BAND,
    flux_to_mag,
    mag_to_flux,
    normalize_detection,
    normalize_lightcurve,
)


# ---- flux ↔ mag round-trip ----

class TestFluxMagConversion:
    def test_known_values(self):
        # 3631 Jy = 3.631e12 nJy should give mag ≈ 0
        mag, _ = flux_to_mag(3.631e12)
        assert abs(mag - 0.0) < 0.01

    def test_round_trip_mag_to_flux(self):
        for mag_in in [15.0, 18.5, 22.0, 25.0]:
            flux, _ = mag_to_flux(mag_in)
            mag_out, _ = flux_to_mag(flux)
            assert abs(mag_out - mag_in) < 1e-10, f"round-trip failed for mag={mag_in}"

    def test_round_trip_flux_to_mag(self):
        for flux_in in [100.0, 1000.0, 50000.0]:
            mag, _ = flux_to_mag(flux_in)
            flux_out, _ = mag_to_flux(mag)
            assert abs(flux_out - flux_in) / flux_in < 1e-10

    def test_error_propagation(self):
        mag, magerr = flux_to_mag(1000.0, 100.0)
        assert mag is not None and math.isfinite(mag)
        assert magerr is not None and magerr > 0

        flux, fluxerr = mag_to_flux(20.0, 0.1)
        assert flux is not None and math.isfinite(flux)
        assert fluxerr is not None and fluxerr > 0

    def test_negative_flux(self):
        mag, magerr = flux_to_mag(-500.0)
        assert math.isnan(mag)

    def test_none_flux(self):
        mag, magerr = flux_to_mag(None)
        assert math.isnan(mag)


# ---- ZTF normalization ----

class TestNormalizeZTF:
    ZTF_DET = {
        "mjd": 59800.5,
        "fid": 1,
        "magpsf": 19.5,
        "sigmapsf": 0.08,
        "isdiffpos": "t",
        "rb": 0.85,
        "drb": 0.92,
    }

    def test_basic_fields(self):
        n = normalize_detection(self.ZTF_DET)
        assert n["survey"] == "ZTF"
        assert n["mjd"] == 59800.5
        assert n["band"] == "g"
        assert n["band_idx"] == 1  # g = index 1
        assert abs(n["mag"] - 19.5) < 1e-10
        assert n["is_positive"] is True

    def test_quality_prefers_drb(self):
        n = normalize_detection(self.ZTF_DET)
        assert n["quality"] == 0.92  # drb preferred over rb

    def test_quality_falls_back_to_rb(self):
        det = {**self.ZTF_DET, "drb": None}
        n = normalize_detection(det)
        assert n["quality"] == 0.85

    def test_flux_derived_from_mag(self):
        n = normalize_detection(self.ZTF_DET)
        assert n["flux"] is not None and math.isfinite(n["flux"])
        # round-trip: flux → mag should give back ~19.5
        mag_back, _ = flux_to_mag(n["flux"])
        assert abs(mag_back - 19.5) < 1e-8

    def test_band_mapping(self):
        for fid, expected_band in ZTF_FID_TO_BAND.items():
            det = {**self.ZTF_DET, "fid": fid}
            n = normalize_detection(det)
            assert n["band"] == expected_band

    def test_auto_detect_ztf(self):
        n = normalize_detection(self.ZTF_DET, survey="auto")
        assert n["survey"] == "ZTF"


# ---- LSST normalization ----

class TestNormalizeLSST:
    LSST_DET = {
        "midpointMjdTai": 61091.035,
        "band": "i",
        "psfFlux": 1488.75,
        "psfFluxErr": 256.865,
        "isNegative": False,
        "reliability": 0.133,
        "snr": 5.59,
        "diaSourceId": 170032882292621441,
        "ra": 62.365,
        "decl": -46.135,
    }

    def test_basic_fields(self):
        n = normalize_detection(self.LSST_DET)
        assert n["survey"] == "LSST"
        assert n["mjd"] == 61091.035
        assert n["band"] == "i"
        assert n["band_idx"] == 3  # i = index 3
        assert n["is_positive"] is True
        assert n["quality"] == 0.133
        assert n["snr"] == 5.59

    def test_magnitude_from_flux(self):
        n = normalize_detection(self.LSST_DET)
        assert math.isfinite(n["mag"])
        # psfFlux=1488.75 nJy → mag ≈ -2.5*log10(1488.75) + 31.4 ≈ 23.47
        expected_mag = -2.5 * math.log10(1488.75) + 31.4
        assert abs(n["mag"] - expected_mag) < 0.001

    def test_magerr_from_fluxerr(self):
        n = normalize_detection(self.LSST_DET)
        assert n["magerr"] is not None
        assert n["magerr"] > 0

    def test_flux_preserved(self):
        n = normalize_detection(self.LSST_DET)
        assert n["flux"] == 1488.75
        assert n["fluxerr"] == 256.865

    def test_negative_detection(self):
        det = {**self.LSST_DET, "isNegative": True}
        n = normalize_detection(det)
        assert n["is_positive"] is False

    def test_auto_detect_lsst(self):
        n = normalize_detection(self.LSST_DET, survey="auto")
        assert n["survey"] == "LSST"

    def test_all_bands(self):
        for band in CANONICAL_BANDS:
            det = {**self.LSST_DET, "band": band}
            n = normalize_detection(det)
            assert n["band"] == band


# ---- Lightcurve normalization ----

class TestNormalizeLightcurve:
    def test_mixed_empty(self):
        result = normalize_lightcurve([])
        assert result == []

    def test_ztf_list(self):
        dets = [
            {"mjd": 59800.0, "fid": 1, "magpsf": 19.0, "sigmapsf": 0.05, "isdiffpos": "t", "rb": 0.9},
            {"mjd": 59801.0, "fid": 2, "magpsf": 19.2, "sigmapsf": 0.06, "isdiffpos": "t", "rb": 0.85},
        ]
        result = normalize_lightcurve(dets)
        assert len(result) == 2
        assert result[0]["band"] == "g"
        assert result[1]["band"] == "r"
        assert all(d["survey"] == "ZTF" for d in result)

    def test_lsst_list(self):
        dets = [
            {"midpointMjdTai": 61091.0, "band": "i", "psfFlux": 1500.0, "psfFluxErr": 200.0, "reliability": 0.5},
            {"midpointMjdTai": 61092.0, "band": "g", "psfFlux": 2000.0, "psfFluxErr": 150.0, "reliability": 0.7},
        ]
        result = normalize_lightcurve(dets, survey="LSST")
        assert len(result) == 2
        assert result[0]["band"] == "i"
        assert result[1]["band"] == "g"
        assert all(d["survey"] == "LSST" for d in result)


# ---- Fixture integration ----

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "raw"


class TestFixtureIntegration:
    @pytest.mark.skipif(
        not (FIXTURES_DIR / "lasair" / "170032882292621441_object.json").exists(),
        reason="LSST Lasair fixture not available",
    )
    def test_lsst_lasair_fixture(self):
        fixture = FIXTURES_DIR / "lasair" / "170032882292621441_object.json"
        with open(fixture) as fh:
            data = json.load(fh)
        sources = data.get("diaSourcesList") or []
        assert len(sources) > 0
        result = normalize_lightcurve(sources, survey="LSST")
        assert len(result) == len(sources)
        for d in result:
            assert d["survey"] == "LSST"
            assert d["band"] in CANONICAL_BANDS
            assert math.isfinite(d["mag"])
            assert d["mjd"] > 60000  # LSST era
