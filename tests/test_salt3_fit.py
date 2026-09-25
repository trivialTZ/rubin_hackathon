"""SALT3 χ² expert: the p(Ia) mapping (fusion v13c) and a live sncosmo fit."""
from __future__ import annotations

import math

import pytest

from debass_meta.experts.local.salt3_fit import Salt3Chi2Expert, _stable_sigmoid, ia_probability


def test_stable_sigmoid_does_not_overflow():
    # math.exp(-x/2) overflowed for x < -1420 and the collector dropped the epoch
    assert _stable_sigmoid(-5000.0) == 0.0
    assert _stable_sigmoid(5000.0) == 1.0
    assert _stable_sigmoid(0.0) == 0.5


def test_ia_probability_extreme_preferences_are_finite():
    p = ia_probability(chi2_ia=4000.0, ndof_ia=15, chi2_nonia=10.0, ndof_nonia=17, n_points=20)
    assert 0.0 <= p < 0.5
    p = ia_probability(chi2_ia=10.0, ndof_ia=15, chi2_nonia=4000.0, ndof_nonia=17, n_points=20)
    assert 0.5 < p <= 1.0


def test_ia_probability_does_not_saturate_with_n():
    # 2 χ² units per point better than the II fit, over 100 well-fitted points
    p = ia_probability(100.0, 95, 300.0, 97, n_points=100)
    assert 0.6 < p < 0.9
    # the old mapping, sigmoid(Δχ²/2), is 1.0 there
    assert 1.0 / (1.0 + math.exp(-(300.0 - 100.0) / 2.0)) == 1.0


def test_ia_probability_poor_fits_are_pulled_to_half():
    # same Δχ², but both templates fit badly (e.g. SN light in the template)
    good = ia_probability(20.0, 15, 60.0, 17, n_points=20)
    poor = ia_probability(600.0, 15, 640.0, 17, n_points=20)
    assert 0.5 < poor < good
    assert poor - 0.5 < 0.05


def test_ia_probability_invariant_to_error_scale():
    # errors underestimated by a common factor: Birge rescaling cancels it
    assert ia_probability(20.0, 15, 60.0, 17, 20) == pytest.approx(ia_probability(600.0, 15, 1800.0, 17, 20), abs=0.02)


def test_ia_probability_penalises_salt3_extra_parameters():
    # equal χ²: SALT3 (5 free parameters) vs Nugent (3) -> favours the simpler model
    assert ia_probability(5.0, 5, 5.0, 7, n_points=10) < 0.5


def _synthetic_ia_lightcurve():
    sncosmo = pytest.importorskip("sncosmo")
    import numpy as np

    model = sncosmo.Model(source="salt3")
    model.set(z=0.05, t0=61000.0, x0=1e-4, x1=0.0, c=0.0)
    rng = np.random.default_rng(1)
    dets = []
    for i, t in enumerate(np.arange(60990.0, 61030.0, 3.0)):
        band = ("lsstg", "lsstr", "lssti")[i % 3]
        flux = float(model.bandflux(band, t, zp=25.0, zpsys="ab"))
        err = max(0.05 * flux, 1.0)
        dets.append({"mjd": float(t), "band": band[-1], "survey": "LSST",
                     "flux": flux + float(rng.normal(0, err)), "fluxerr": err})
    return dets


def test_live_fit_prefers_ia_and_records_summary():
    dets = _synthetic_ia_lightcurve()
    expert = Salt3Chi2Expert()
    if not expert._available:
        pytest.skip("sncosmo unavailable")
    out = expert.predict_epoch("x", dets, dets[-1]["mjd"] + 2400000.5 + 0.1)
    assert out.available
    assert out.class_probabilities["Ia"] > 0.5
    summary = out.raw_output["summary"]
    assert summary["n_points"] == len(dets)
    assert summary["chi2_ia"] < summary["chi2_nonia"]


def test_negative_only_lightcurve_is_unavailable():
    # SN light in the template: every difference flux negative -> no fit, marked unavailable
    dets = _synthetic_ia_lightcurve()
    for d in dets:
        d["flux"] = -abs(d["flux"])
    expert = Salt3Chi2Expert()
    if not expert._available:
        pytest.skip("sncosmo unavailable")
    out = expert.predict_epoch("x", dets, dets[-1]["mjd"] + 2400000.5 + 0.1)
    assert not out.available
    assert out.class_probabilities == {}


def test_too_few_points_is_unavailable():
    out = Salt3Chi2Expert().predict_epoch("x", [{"mjd": 61000.0, "band": "g", "survey": "LSST",
                                                 "flux": 10.0, "fluxerr": 1.0}], 2461001.0)
    assert not out.available
