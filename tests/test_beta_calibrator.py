"""v13g smooth head-1 calibrator (``--head1-calibrator lsst:beta``).

Synthetic data only.  Covers: beta calibration is monotone and has no plateaus
where isotonic does; a negative coefficient is refitted away so the map stays
monotone; per-row weights are respected; pickle round trip; the per-survey
override is opt-in (defaults unchanged, old metadata loads) and persisted.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_hierarchical_followup import _build_dataset  # noqa: E402

from debass_meta.models.calibrate import BetaCalibrator, IsotonicCalibrator  # noqa: E402
from debass_meta.models.hierarchical_followup import (  # noqa: E402
    AvailabilityDropout,
    HierarchicalFollowup,
    _fit_binary_calibrator,
    _IdentityBinaryCalibrator,
    parse_head1_calibrator_specs,
)
from debass_meta.models.pooled_trust import PlattCalibrator  # noqa: E402

SEED = 42


def _toy(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    p = np.clip(rng.beta(0.6, 0.6, n), 1e-4, 1 - 1e-4)
    z = 3.0 * np.log(p / (1 - p)) * 0.5 - 0.5
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return p, y


def test_beta_monotone_and_smooth():
    p, y = _toy()
    cal = BetaCalibrator().fit(p, y)
    grid = np.linspace(0.0, 1.0, 501)
    out = cal.transform(grid)
    assert np.all(np.diff(out) >= -1e-12)
    assert np.all((out >= 0) & (out <= 1)) and np.all(np.isfinite(out))
    assert cal.a >= 0 and cal.b >= 0
    # no plateaus: distinct inputs -> distinct outputs where isotonic ties
    iso = IsotonicCalibrator().fit(p, y).transform(p)
    beta = cal.transform(p)
    assert len(np.unique(iso)) < len(np.unique(p)) / 3
    assert len(np.unique(beta)) == len(np.unique(p))


def test_beta_negative_coefficient_refit_stays_monotone():
    grid = np.linspace(0.0, 1.0, 401)
    n_refit = 0
    for seed in range(12):
        rng = np.random.default_rng(seed)
        p = np.clip(rng.beta(0.4, 0.4, 150), 1e-4, 1 - 1e-4)
        # labels depend on p only through its upper tail -> ln p tends to come out negative
        y = (rng.random(150) < np.where(p > 0.9, 0.9, 0.12)).astype(int)
        full = BetaCalibrator._lr(BetaCalibrator._features(p), y, None).coef_[0]
        cal = BetaCalibrator().fit(p, y)
        n_refit += int(full.min() < 0)
        if full.min() < 0:
            assert cal.a == 0.0 and cal.b > 0
        assert cal.a >= 0 and cal.b >= 0
        assert np.all(np.diff(cal.transform(grid)) >= -1e-12)
    assert n_refit >= 5            # the refit path is exercised
    # anti-correlated labels: both coefficients negative -> the weighted base rate
    p = np.linspace(0.05, 0.95, 100)
    y = (p < 0.5).astype(int)
    cal = BetaCalibrator().fit(p, y)
    assert cal.a == 0.0 and cal.b == 0.0
    np.testing.assert_allclose(cal.transform(grid), 0.5, atol=1e-9)


def test_beta_drops_only_the_negative_feature():
    p = np.linspace(0.01, 0.99, 400)
    rng = np.random.default_rng(3)
    # logit(y) rises with -ln(1-p) only (b > 0, a = 0): plenty of mass at both ends
    z = 2.5 * (-np.log(1 - p)) - 2.5
    y = (rng.random(400) < 1 / (1 + np.exp(-z))).astype(int)
    cal = BetaCalibrator().fit(p, y)
    assert cal.a >= 0 and cal.b > 0
    assert np.all(np.diff(cal.transform(np.linspace(0, 1, 101))) >= -1e-12)


def test_beta_weights_respected():
    p, y = _toy()
    base = BetaCalibrator().fit(p, y)
    # integer weights == repeated rows
    w = np.random.default_rng(5).integers(1, 4, len(p))
    weighted = BetaCalibrator().fit(p, y, sample_weight=w)
    repeated = BetaCalibrator().fit(np.repeat(p, w), np.repeat(y, w))
    np.testing.assert_allclose(
        [weighted.a, weighted.b, weighted.c], [repeated.a, repeated.b, repeated.c], atol=1e-4)
    grid = np.linspace(0.01, 0.99, 50)
    np.testing.assert_allclose(weighted.transform(grid), repeated.transform(grid), atol=1e-4)
    # uniform weights == unweighted; up-weighting positives raises the map
    np.testing.assert_allclose(
        BetaCalibrator().fit(p, y, sample_weight=np.full(len(p), 2.0)).transform(grid),
        base.transform(grid), atol=1e-5)
    up = BetaCalibrator().fit(p, y, sample_weight=np.where(y == 1, 5.0, 1.0))
    assert np.all(up.transform(grid) > base.transform(grid))
    # through the factory a zero-weight row is dropped
    wz = np.ones(len(p))
    wz[:50] = 0.0
    a = _fit_binary_calibrator(p, y, n_min=10, sample_weight=wz, kind="beta")
    b = BetaCalibrator().fit(p[50:], y[50:])
    np.testing.assert_allclose(a.transform(grid), b.transform(grid), atol=1e-6)


def test_beta_clips_extremes_and_pickles():
    p, y = _toy()
    cal = BetaCalibrator().fit(p, y)
    out = cal.transform(np.array([0.0, 1e-9, 1.0, 1.0 - 1e-9, 0.5]))
    assert np.all(np.isfinite(out)) and np.all((out > 0) & (out < 1))
    back = pickle.loads(pickle.dumps(cal))
    grid = np.linspace(0, 1, 101)
    np.testing.assert_array_equal(back.transform(grid), cal.transform(grid))
    # an unfitted calibrator passes the input through; a single class gives identity
    np.testing.assert_array_equal(BetaCalibrator().transform(np.array([0.2, 0.7])), [0.2, 0.7])
    assert isinstance(_fit_binary_calibrator(p, np.ones(len(p), int), n_min=2, kind="beta"),
                      _IdentityBinaryCalibrator)


def test_factory_default_and_kinds_unchanged():
    p, y = _toy(n=120)
    assert type(_fit_binary_calibrator(p, y, n_min=40)) is IsotonicCalibrator
    assert type(_fit_binary_calibrator(p, y, n_min=500)) is PlattCalibrator
    assert type(_fit_binary_calibrator(p, y, n_min=500, kind="isotonic")) is IsotonicCalibrator
    assert type(_fit_binary_calibrator(p, y, n_min=10, kind="platt")) is PlattCalibrator
    assert type(_fit_binary_calibrator(p, y, n_min=10, kind="beta")) is BetaCalibrator
    assert _fit_binary_calibrator(p, y, n_min=10, kind="beta").name == "beta"
    with pytest.raises(ValueError):
        _fit_binary_calibrator(p, y, n_min=10, kind="spline")
    # kind=None is bit-identical to the pre-v13g call
    a = _fit_binary_calibrator(p, y, n_min=40, sample_weight=np.ones(len(p)))
    b = _fit_binary_calibrator(p, y, n_min=40, sample_weight=np.ones(len(p)), kind=None)
    np.testing.assert_array_equal(a.transform(p), b.transform(p))


def test_parse_specs():
    assert parse_head1_calibrator_specs([]) == {}
    assert parse_head1_calibrator_specs(["LSST:Beta", "ztf:platt"]) == {"lsst": "beta", "ztf": "platt"}
    for bad in ("lsst", "lsst:", ":beta", "lsst:spline"):
        with pytest.raises(ValueError):
            parse_head1_calibrator_specs([bad])


def _fit(**kw):
    df, tr, ca, te = _build_dataset()
    return HierarchicalFollowup(survey_cal_min=40, n_jobs=2, seed=SEED, **kw).fit(df, tr, ca, te), df


def test_head_default_unchanged_and_override_scoped_to_survey(tmp_path):
    default, df = _fit()
    explicit, _ = _fit(head1_calibrator={})
    np.testing.assert_array_equal(default.predict_proba(df), explicit.predict_proba(df))
    assert "head1_calibrator" not in default._v13_settings()
    beta, _ = _fit(head1_calibrator={"lsst": "beta"})
    assert type(beta.head1_calibrators["lsst"]) is BetaCalibrator
    assert beta.head1_calibrator_kinds["lsst"] == "beta"
    assert type(beta.head1_calibrators["ztf"]) is IsotonicCalibrator
    # heads and the other survey are untouched; only LSST P(SN) moves
    z = (df["survey"].astype(str).str.lower() == "ztf").to_numpy()
    np.testing.assert_array_equal(beta.predict_proba(df)[z], default.predict_proba(df)[z])
    np.testing.assert_array_equal(beta.predict_proba_raw(df), default.predict_proba_raw(df))
    assert not np.array_equal(beta.predict_proba(df)[~z], default.predict_proba(df)[~z])
    # persisted and used at scoring
    beta.save(str(tmp_path))
    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["v13"]["head1_calibrator"] == {"lsst": "beta"}
    loaded = HierarchicalFollowup.load(str(tmp_path))
    assert loaded.head1_calibrator == {"lsst": "beta"}
    np.testing.assert_array_equal(loaded.predict_proba(df), beta.predict_proba(df))
    # an artifact saved without the key loads with the default
    default.save(str(tmp_path / "d"))
    assert HierarchicalFollowup.load(str(tmp_path / "d")).head1_calibrator == {}
    with pytest.raises(ValueError):
        _fit(head1_calibrator={"lsst": "spline"})


def test_crossfit_weighted_path_uses_beta():
    kw = dict(head1_exclude_qualities=("lsst:weak",), equalize_lsst_provenance=False,
              context_mask_scope="survey", cross_fit_folds=3, head1_cal_weights="object",
              dropout=AvailabilityDropout(regime_weight=0.1, random_frac=0.25, seed=SEED))
    iso, df = _fit(**kw)
    # force both surveys to isotonic vs LSST beta: the weighted cross-fit calibrators
    beta, _ = _fit(head1_calibrator={"lsst": "beta", "ztf": "isotonic"}, **kw)
    assert type(beta.head1_calibrators["lsst"]) is BetaCalibrator
    assert beta.head1_calibrator_kinds["ztf"] == "isotonic"
    lst = (df["survey"].astype(str).str.lower() == "lsst").to_numpy()
    p = beta.predict_proba(df)
    assert np.all(np.isfinite(p)) and np.allclose(p.sum(axis=1), 1.0)
    # beta gives (nearly) all-distinct P(SN) on LSST rows where the isotonic map ties
    p1_beta = 1.0 - p[lst, 2]
    p1_iso = 1.0 - iso.predict_proba(df)[lst, 2]
    assert len(np.unique(np.round(p1_beta, 12))) >= len(np.unique(np.round(p1_iso, 12)))


def test_orchestrator_head1_calibrator_flag(tmp_path):
    from test_train_v11_smoke import _argv, _write_inputs

    from scripts.train_fusion_v11 import main

    snap, split_path, locked_path, truth_path = _write_inputs(tmp_path)
    rc = main(_argv(tmp_path, snap, split_path, locked_path, truth_path,
                    extra=("--head1-calibrator", "lsst:beta")))
    assert rc == 0
    report = json.loads((tmp_path / "fusion_v11_train.json").read_text())
    assert report["v13"]["head1_calibrator"] == {"lsst": "beta"}
    meta = json.loads((tmp_path / "followup_v11" / "metadata.json").read_text())
    assert meta["v13"]["head1_calibrator"] == {"lsst": "beta"}
    assert meta["head1_calibrator_kinds"]["lsst"] == "beta"
    # default run: nothing recorded
    (tmp_path / "d").mkdir()
    s2, sp2, lk2, tr2 = _write_inputs(tmp_path / "d")
    assert main(_argv(tmp_path / "d", s2, sp2, lk2, tr2)) == 0
    assert "head1_calibrator" not in json.loads((tmp_path / "d" / "fusion_v11_train.json").read_text())["v13"]
    with pytest.raises(SystemExit, match="head1-calibrator"):
        main(_argv(tmp_path / "d", s2, sp2, lk2, tr2, extra=("--head1-calibrator", "lsst:spline")))
