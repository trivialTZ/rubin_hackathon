"""Unit tests for the fusion_v11 hierarchical follow-up head (P3).

Self-contained synthetic data (no data/ access).  Covers the spec §3-P3 test
list: simplex sum-to-1 + clipping; P2-constant => p_snia rank == P1 rank
(graceful degradation); save/load roundtrip incl. predict_proba_raw; tiny-n
survey => Platt fallback; head-2 default excludes survey_is_lsst; LSST routing
default = base rate; plus composition coherence and weak-label routing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from debass_meta.models.calibrate import IsotonicCalibrator
from debass_meta.models.hierarchical_followup import (
    HierarchicalFollowup,
    _IdentityBinaryCalibrator,
    head2_training_mask,
    is_sn_target,
    survey_series,
)
from debass_meta.models.pooled_trust import PlattCalibrator

SEED = 42

# Real registry sanitized keys so provenance masking has valid targets.
_SANS = ("fink__snn", "alerce__stamp_classifier", "alerce_lc",
         "lasair__sherlock", "babamul")
# lcf feature centers per (is_sn, is_ia) regime — give head-1 and head-2 signal.
_SN_CENTER = np.array([2.0, -1.0, 0.5, 1.0])
_OTHER_CENTER = np.array([-2.0, 1.5, -1.0, -0.5])
_IA_SHIFT = np.array([1.5, -1.0, 0.0, 0.8])


def _class_for(is_sn: int, is_ia: int) -> str:
    if not is_sn:
        return "other"
    return "snia" if is_ia else "nonIa_snlike"


def _add_expert_block(row: dict, rng: np.random.Generator, p_snia: float) -> None:
    for san in _SANS:
        if rng.random() < 0.7:
            row[f"proj__{san}__p_snia"] = float(
                np.clip(p_snia + rng.normal(0, 0.25), 0, 1)
            )
            row[f"avail__{san}"] = 1.0
            row[f"exact__{san}"] = 1.0
            row[f"q__{san}"] = float(rng.uniform(0.3, 0.9))
        else:
            row[f"proj__{san}__p_snia"] = np.nan
            row[f"avail__{san}"] = 0.0
            row[f"exact__{san}"] = 0.0
            row[f"q__{san}"] = np.nan


def _build_dataset(seed: int = SEED):
    """Return (df, train_ids, cal_ids, test_ids).

    Split hand-crafted so ZTF cal is large (isotonic head-1 calibrator) and
    LSST cal is tiny with both is_sn classes (Platt fallback).  LSST objects
    carry a single epoch so LSST cal ROWS stay < survey_cal_min.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    split: dict[str, str] = {}

    def emit(object_id, survey, is_sn, is_ia, quality, n_epochs, phase):
        cls = _class_for(is_sn, is_ia)
        for n_det in range(1, n_epochs + 1):
            center = _SN_CENTER if is_sn else _OTHER_CENTER
            feats = center + (_IA_SHIFT if is_ia else 0.0) + rng.normal(0, 1.1, 4)
            # Spectroscopic rows carry concrete subtype provenance (guard G7);
            # weak/context rows are untyped (tns_type None, bts_type '-') but
            # never reach head-2.
            subtype = ("SN Ia" if is_ia else "SN II")
            row = {
                "object_id": object_id,
                "n_det": n_det,
                "survey": survey,
                "target_class": cls,
                "label_quality": quality,
                "label_source": "tns" if quality == "spectroscopic" else "weak_stamp",
                "tns_type": subtype if quality == "spectroscopic" else None,
                "bts_type": subtype if quality == "spectroscopic" else "-",
                "alert_jd": 2460000.0 + n_det,
                "survey_is_lsst": 1.0 if survey == "LSST" else 0.0,
                "traj_x__mean_slope": float(rng.normal()),
            }
            for j, v in enumerate(feats):
                row[f"lcf_{j}"] = float(v)
            _add_expert_block(row, rng, 0.8 if is_sn else 0.15)
            rows.append(row)
        split[object_id] = phase

    # ---- ZTF: 200 objects, multi-epoch, mostly spectroscopic ----
    zi = 0
    for phase, count in (("train", 120), ("cal", 60), ("test", 20)):
        for _ in range(count):
            is_sn = int(rng.random() < 0.6)
            is_ia = int(is_sn and rng.random() < 0.55)
            # a fraction of ZTF SN are weak (must NOT reach head-2)
            quality = "spectroscopic"
            if is_sn and phase == "train" and rng.random() < 0.2:
                quality = "weak"
            emit(f"ZTF{zi:05d}", "ZTF", is_sn, is_ia, quality,
                 int(rng.integers(4, 9)), phase)
            zi += 1

    # ---- LSST: single-epoch; tiny cal frame, both is_sn classes ----
    li = 0
    for phase, count in (("train", 40), ("cal", 18), ("test", 12)):
        for k in range(count):
            is_sn = int(k % 2 == 0)          # guarantee both classes in cal
            is_ia = int(is_sn and rng.random() < 0.5)
            quality = "spectroscopic" if (is_sn and phase != "train") else (
                "weak" if is_sn else "context"
            )
            emit(f"LSST{li:05d}", "LSST", is_sn, is_ia, quality, 1, phase)
            li += 1

    df = pd.DataFrame(rows)
    train_ids = {k for k, v in split.items() if v == "train"}
    cal_ids = {k for k, v in split.items() if v == "cal"}
    test_ids = {k for k, v in split.items() if v == "test"}
    return df, train_ids, cal_ids, test_ids


@pytest.fixture(scope="module")
def fitted():
    df, tr, ca, te = _build_dataset()
    model = HierarchicalFollowup(
        survey_cal_min=40, n_jobs=2, seed=SEED
    ).fit(df, tr, ca, te)
    return model, df, tr, ca, te


# --------------------------------------------------------------------------
# (a) composition coherence: simplex sum-to-1 + clipping bounds
# --------------------------------------------------------------------------

def test_simplex_sum_to_one(fitted):
    model, df, _, _, te = fitted
    eval_df = df[df["object_id"].isin(te)]
    for proba in (model.predict_proba(eval_df),
                  model.predict_proba_raw(eval_df)):
        assert proba.shape == (len(eval_df), 3)
        assert np.all(np.isfinite(proba))
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-9)


def test_compose_clips_extreme_heads():
    # P1=0,P2=1 and P1=1,P2=0 must stay strictly inside (0,1) -> finite log-loss
    out = HierarchicalFollowup._compose(
        np.array([0.0, 1.0]), np.array([1.0, 0.0])
    )
    assert np.all(out > 0.0) and np.all(out < 1.0)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-12)
    # p_other = 1 - clip(P1); at P1=0 -> ~1-1e-6, never exactly 1
    assert out[0, 2] == pytest.approx(1.0 - 1e-6, abs=1e-9)
    assert out[1, 2] == pytest.approx(1e-6, abs=1e-9)


# --------------------------------------------------------------------------
# (b) P2 constant => p_snia ranking equals P1 ranking (graceful degradation)
# --------------------------------------------------------------------------

def test_p2_constant_preserves_p1_rank(fitted):
    model, df, _, _, te = fitted
    assert not model.head2_lsst_enabled, "LSST head-2 must default OFF"
    lsst_eval = df[(df["object_id"].isin(te)) & (df["survey"] == "LSST")]
    assert len(lsst_eval) >= 5

    # LSST P2 is a single constant base rate -> p_snia is monotone in P1.
    p2 = model._head2_p(lsst_eval, calibrated=False)
    assert np.allclose(p2, p2[0]), "LSST P2 must be a single constant"

    proba = model.predict_proba_raw(lsst_eval)
    p_snia = proba[:, 0]
    p1 = 1.0 - proba[:, 2]            # P1 recovered from p_other
    assert np.array_equal(np.argsort(p_snia, kind="stable"),
                          np.argsort(p1, kind="stable"))


# --------------------------------------------------------------------------
# (c) save / load roundtrip incl. predict_proba_raw
# --------------------------------------------------------------------------

def test_save_load_roundtrip(fitted, tmp_path):
    model, df, _, _, te = fitted
    out_dir = tmp_path / "hier_v11"
    model.save(str(out_dir))
    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "model.pkl").exists()
    assert (out_dir / "calibrator.pkl").exists()

    loaded = HierarchicalFollowup.load(str(out_dir))
    assert loaded.head1_feature_cols == model.head1_feature_cols
    assert loaded.head2_feature_cols == model.head2_feature_cols
    assert loaded.head2_lsst_enabled == model.head2_lsst_enabled
    assert loaded.base_rate == model.base_rate

    eval_df = df[df["object_id"].isin(te)]
    assert np.allclose(loaded.predict_proba(eval_df),
                       model.predict_proba(eval_df), atol=1e-12)
    assert np.allclose(loaded.predict_proba_raw(eval_df),
                       model.predict_proba_raw(eval_df), atol=1e-12)


# --------------------------------------------------------------------------
# (d) tiny-n survey => Platt fallback; large survey => isotonic
# --------------------------------------------------------------------------

def test_tiny_survey_platt_fallback(fitted):
    model, _, _, ca, _ = fitted
    kinds = model.head1_calibrator_kinds
    # ZTF cal is large (many rows) -> isotonic; LSST cal tiny -> Platt.
    assert kinds.get("ztf") == "isotonic"
    assert kinds.get("lsst") == "platt"
    assert isinstance(model.head1_calibrators["ztf"], IsotonicCalibrator)
    assert isinstance(model.head1_calibrators["lsst"], PlattCalibrator)


def test_identity_calibrator_when_single_class():
    from debass_meta.models.hierarchical_followup import _fit_binary_calibrator

    cal = _fit_binary_calibrator(
        np.array([0.2, 0.3, 0.4]), np.array([1, 1, 1]), n_min=2
    )
    assert isinstance(cal, _IdentityBinaryCalibrator)
    assert np.allclose(cal.transform(np.array([0.5, 0.9])), [0.5, 0.9])


# --------------------------------------------------------------------------
# (e) head-2 default excludes survey_is_lsst; head-1 keeps it
# --------------------------------------------------------------------------

def test_head2_drops_survey_flag(fitted):
    model, _, _, _, _ = fitted
    assert "survey_is_lsst" not in model.head2_feature_cols
    # the pooled head-1 IS allowed to use the survey flag
    assert "survey_is_lsst" in model.head1_feature_cols
    assert model.head1_mode == "pooled"          # preregistered default


# --------------------------------------------------------------------------
# (f) LSST head-2 routing default = per-survey constant base rate
# --------------------------------------------------------------------------

def test_lsst_routing_default_base_rate(fitted):
    model, df, tr, _, _ = fitted
    assert model.head2_lsst_enabled is False
    lsst = df[(df["object_id"].isin(tr)) & (df["survey"] == "LSST")].head(20)
    p2_raw = model._head2_p(lsst, calibrated=False)
    p2_cal = model._head2_p(lsst, calibrated=True)
    expected = model.base_rate["lsst"]
    assert np.allclose(p2_raw, expected)
    assert np.allclose(p2_cal, expected)     # constant base rate not recalibrated
    lo, hi = model.base_rate_clip
    assert lo <= expected <= hi


# --------------------------------------------------------------------------
# (g) weak-label routing: weak SN rows never reach head-2
# --------------------------------------------------------------------------

def test_weak_sn_rows_excluded_from_head2():
    df, tr, ca, te = _build_dataset()
    train_df = df[df["object_id"].isin(tr)]
    mask = head2_training_mask(train_df, surveys=("ztf",))
    picked = train_df[mask]

    # every picked row is a ZTF spectroscopic SN row
    assert (picked["survey"] == "ZTF").all()
    assert (picked["label_quality"] == "spectroscopic").all()
    assert picked["target_class"].isin(("snia", "nonIa_snlike")).all()

    # no weak / context / LSST / 'other' row survived
    weak_sn = train_df[
        (train_df["label_quality"] == "weak")
        & train_df["target_class"].isin(("snia", "nonIa_snlike"))
    ]
    assert len(weak_sn) > 0, "fixture must contain weak SN rows to test routing"
    assert not mask[train_df["object_id"].isin(weak_sn["object_id"])].any()
    assert (picked["target_class"] != "other").all()


def test_head2_train_count_matches_mask(fitted):
    model, df, tr, _, _ = fitted
    train_df = df[df["object_id"].isin(tr)]
    n_expected = int(head2_training_mask(train_df, surveys=("ztf",)).sum())
    assert model.report_["head2_n_train_rows"] == n_expected
    assert n_expected > 0


# --------------------------------------------------------------------------
# (h) helper sanity
# --------------------------------------------------------------------------

def test_target_and_survey_helpers():
    df = pd.DataFrame({
        "target_class": ["snia", "nonIa_snlike", "other"],
        "survey": ["ZTF", "LSST", "ztf"],
    })
    assert list(is_sn_target(df)) == [1, 1, 0]
    assert list(survey_series(df)) == ["ztf", "lsst", "ztf"]


def test_report_gate_verdicts_present(fitted):
    model, _, _, _, _ = fitted
    gates = {g["gate"] for g in model.report_["gate_verdicts"]}
    assert "head1_per_survey_vs_pooled" in gates
    assert "head2_on_lsst" in gates
    assert "availability_audit" in model.report_
    assert "base_rate" in model.report_


# --------------------------------------------------------------------------
# (i) guard G7 — head-2 rows must carry concrete subtype provenance (B0)
# --------------------------------------------------------------------------

def test_g7_enforced_and_passes(fitted):
    """Fixture spec rows are typed => G7 enforced, zero untyped, fit succeeds."""
    model, _, _, _, _ = fitted
    assert model.report_["g7"]["status"] == "enforced"
    assert model.report_["g7"]["n_untyped_provenance"] == 0
    assert model.report_["g7"]["n_head2_rows"] > 0


def test_g7_fires_on_untyped_spec_row():
    """A spectroscopic SN row with untyped provenance (bts_type '-', tns_type
    None) — the exact B0 pre-fix bug — must trip the G7 hard assert in fit."""
    df, tr, ca, te = _build_dataset()
    train_df = df[df["object_id"].isin(tr)]
    spec_sn = train_df[
        (train_df["survey"] == "ZTF")
        & (train_df["label_quality"] == "spectroscopic")
        & train_df["target_class"].isin(("snia", "nonIa_snlike"))
    ]
    assert len(spec_sn) > 0, "fixture must contain a ZTF spec SN row"
    bad_oid = spec_sn["object_id"].iloc[0]
    m = df["object_id"] == bad_oid
    df.loc[m, "tns_type"] = None
    df.loc[m, "bts_type"] = "-"          # BTS filler force-mapped to a ternary

    with pytest.raises(AssertionError, match="G7"):
        HierarchicalFollowup(survey_cal_min=40, n_jobs=2, seed=SEED).fit(
            df, tr, ca, te
        )


def test_g7_not_evaluable_without_provenance_cols():
    """A frame lacking BOTH provenance columns cannot be checked — G7 records a
    not-evaluable note instead of force-failing (synthetic-frame path)."""
    df, tr, ca, te = _build_dataset()
    df = df.drop(columns=["tns_type", "bts_type"])
    model = HierarchicalFollowup(survey_cal_min=40, n_jobs=2, seed=SEED).fit(
        df, tr, ca, te
    )
    assert model.report_["g7"]["status"] == "not_evaluable_no_provenance_cols"
