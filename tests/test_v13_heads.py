"""fusion v13 follow-up-head tests (docs/fusion_v13_plan.md, B1-B5).

Synthetic frames only (no data/ access).  Covers: object-weight invariance
under availability dropout; regimes blank exactly the right columns; G8 fires
on class-pure masking and passes after the fix; OOF head predictions differ
from in-sample on train rows (and equal them on cal rows); defaults reproduce
the v11/v12 behaviour; survey-scoped quality exclusion; the anchor-blend
honesty/weight extensions; the orchestrator flags end to end.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_hierarchical_followup import _build_dataset  # noqa: E402

from debass_meta.features.availability import BROKER_EXPERTS, LOCAL_EXPERTS  # noqa: E402
from debass_meta.models import anchor_blend  # noqa: E402
from debass_meta.models.hierarchical_followup import (  # noqa: E402
    AvailabilityDropout,
    G8Error,
    HierarchicalFollowup,
    augment_availability,
    availability_label_corr,
    exclude_qualities,
    excluded_quality_mask,
)
from debass_meta.models.multiclass_followup import (  # noqa: E402
    blank_expert_blocks,
    compute_base_weights,
    expert_dropout_augment,
)
from debass_meta.projectors.base import sanitize_expert_key  # noqa: E402

SEED = 42
_BROKER = "fink_lsst/snn"          # in TRAJ_EXPERTS -> has traj__ columns
_BROKER2 = "babamul"
_LOCAL = "seq_v11"
_LOCAL2 = "alerce_lc"
_SANS = [sanitize_expert_key(k) for k in (_BROKER, _BROKER2, _LOCAL, _LOCAL2)]


def _gold(n_obj: int = 40, epochs: int = 3, seed: int = SEED) -> pd.DataFrame:
    """LSST+ZTF gold-like frame with every per-expert column family."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_obj):
        survey = "LSST" if i % 2 == 0 else "ZTF"
        is_sn = int(rng.random() < 0.5)
        for n_det in range(1, epochs + 1):
            r = {
                "object_id": f"O{i:04d}", "n_det": n_det, "alert_jd": 2460000.0 + n_det,
                "survey": survey, "survey_is_lsst": float(survey == "LSST"),
                "target_class": "snia" if is_sn else "other",
                "label_quality": "spectroscopic" if is_sn else "context",
                "label_source": "tns" if is_sn else "catalog:gaia_star",
                "mag_mean": float(rng.normal(20, 1)), "lcf_0": float(rng.normal()),
                "traj_x__mean_last": float(rng.random()), "traj_x__n_experts": 1.0,
                "traj_x__disagreement_last": np.nan,
            }
            for san in _SANS:
                r[f"proj__{san}__p_snia"] = float(rng.random())
                r[f"avail__{san}"] = 1.0
                r[f"exact__{san}"] = 1.0
                r[f"q__{san}"] = float(rng.random())
                r[f"q_prior__{san}"] = float(rng.random())
                r[f"mapped_pred_class__{san}"] = "snia"
                r[f"reason__{san}"] = "ok"
                r[f"traj__{san}__last"] = float(rng.random())
            rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# B3: augmentation — weight invariance + exact column blanking
# --------------------------------------------------------------------------

def test_augment_object_weight_invariance():
    df = _gold()
    w = compute_base_weights(df)
    spec = AvailabilityDropout(regime_weight=0.15, random_frac=0.5, seed=3)
    aug, w_aug, info = augment_availability(df, w, spec)
    before = pd.Series(w).groupby(df["object_id"].to_numpy()).sum()
    after = pd.Series(w_aug).groupby(aug["object_id"].to_numpy()).sum()
    pd.testing.assert_series_equal(before.sort_index(), after.sort_index(), check_names=False)
    assert info["object_weight_max_abs_delta"] < 1e-12
    assert len(aug) > len(df)
    assert (w_aug > 0).all()
    # copies keep object_id and carry is_aug / aug_regime / unique row_key
    assert set(aug["aug_regime"].unique()) == {"", "no_broker", "no_local", "no_expert", "random"}
    assert aug["row_key"].is_unique
    assert aug.loc[aug["is_aug"] == 0, "row_key"].str.endswith("|").all()


def test_regimes_blank_exactly_the_right_columns():
    df = _gold()
    w = compute_base_weights(df)
    spec = AvailabilityDropout(regime_weight=0.1, random_frac=0.0)
    aug, _, info = augment_availability(df, w, spec)
    broker_sans = [sanitize_expert_key(k) for k in (_BROKER, _BROKER2)]
    local_sans = [sanitize_expert_key(k) for k in (_LOCAL, _LOCAL2)]
    lsst_rows = aug["survey"].str.lower() == "lsst"

    def block(frame, san):
        return {
            "proj": frame[f"proj__{san}__p_snia"], "avail": frame[f"avail__{san}"],
            "exact": frame[f"exact__{san}"], "q": frame[f"q__{san}"],
            "mapped": frame[f"mapped_pred_class__{san}"], "traj": frame[f"traj__{san}__last"],
            "q_prior": frame[f"q_prior__{san}"], "reason": frame[f"reason__{san}"],
        }

    nb = aug[aug["aug_regime"] == "no_broker"]
    assert len(nb) == int(lsst_rows[aug["is_aug"] == 0].sum())     # every LSST row
    assert (nb["survey"] == "LSST").all()
    for san in broker_sans:
        b = block(nb, san)
        assert b["proj"].isna().all() and b["q"].isna().all() and b["traj"].isna().all()
        assert (b["avail"] == 0).all() and (b["exact"] == 0).all()
        assert b["mapped"].isna().all() and b["reason"].isna().all()
        assert b["q_prior"].notna().all()                              # q_prior stays
    for san in local_sans:
        b = block(nb, san)
        assert b["proj"].notna().all() and (b["avail"] == 1).all() and b["traj"].notna().all()
    # cross-expert trajectory aggregates go with the brokers: blanked here
    # (synthetic frame lacks the full per-expert traj block -> "blanked")
    assert nb["traj_x__mean_last"].isna().all()
    assert info["traj_x"]["no_broker"] == "blanked"
    # lightcurve features untouched
    src = aug[(aug["is_aug"] == 0) & lsst_rows].set_index("row_key")
    np.testing.assert_array_equal(nb["mag_mean"].to_numpy(),
                                  src["mag_mean"].to_numpy())

    nl = aug[aug["aug_regime"] == "no_local"]
    for san in local_sans:
        b = block(nl, san)
        assert b["proj"].isna().all() and (b["avail"] == 0).all() and b["q_prior"].notna().all()
    for san in broker_sans:
        assert block(nl, san)["proj"].notna().all()
    assert nl["traj_x__mean_last"].notna().all()                       # brokers survive

    ne = aug[aug["aug_regime"] == "no_expert"]
    for san in broker_sans + local_sans:
        b = block(ne, san)
        assert b["proj"].isna().all() and (b["avail"] == 0).all() and b["q_prior"].notna().all()
    # ZTF rows never get structured copies
    assert (aug.loc[aug["is_aug"] == 1, "survey"] == "LSST").all()


def test_groups_come_from_availability_module():
    assert _BROKER in BROKER_EXPERTS and _BROKER2 in BROKER_EXPERTS
    assert _LOCAL in LOCAL_EXPERTS and _LOCAL2 in LOCAL_EXPERTS


def test_traj_x_recomputed_when_full_traj_block_present():
    from debass_meta.features.trajectory import TRAJ_EXPERTS, TRAJ_STATS

    df = _gold(n_obj=6)
    rng = np.random.default_rng(0)
    for k in TRAJ_EXPERTS:
        san = sanitize_expert_key(k)
        for st in TRAJ_STATS:
            df[f"traj__{san}__{st}"] = rng.random(len(df))
    for c in ("n_experts", "mean_last", "max_last", "mean_slope",
              "disagreement_last", "disagreement_trend"):
        df[f"traj_x__{c}"] = 1.0
    cp = df.copy()
    dropped = [sanitize_expert_key(k) for k in TRAJ_EXPERTS[:2]]
    note = blank_expert_blocks(cp, np.ones(len(cp), bool), dropped)
    assert note["traj_x"] == "recomputed"
    assert (cp["traj_x__n_experts"] == len(TRAJ_EXPERTS) - 2).all()
    surviving = [f"traj__{sanitize_expert_key(k)}__last" for k in TRAJ_EXPERTS[2:]]
    np.testing.assert_allclose(cp["traj_x__mean_last"], df[surviving].mean(axis=1))
    # dropping every trajectory expert -> zero experts, NaN aggregates
    cp2 = df.copy()
    blank_expert_blocks(cp2, np.ones(len(cp2), bool),
                        [sanitize_expert_key(k) for k in TRAJ_EXPERTS])
    assert (cp2["traj_x__n_experts"] == 0).all() and cp2["traj_x__mean_last"].isna().all()


def test_expert_dropout_augment_old_path_unchanged_and_full_block():
    df = _gold(n_obj=10)
    w = compute_base_weights(df)
    aug, aw, info = expert_dropout_augment(df, w, aug_frac=0.5, aug_weight=0.3, seed=1)
    assert set(info) == {"n_source_rows", "n_eligible", "n_aug", "n_keep_one", "source_positions"}
    # old path: traj / mapped_pred_class untouched even where proj was dropped
    san = _SANS[0]
    dropped = aug[f"avail__{san}"] == 0
    assert dropped.any()
    assert aug.loc[dropped, f"traj__{san}__last"].notna().all()
    assert aug.loc[dropped, f"mapped_pred_class__{san}"].notna().all()
    np.testing.assert_allclose(aw, w[info["source_positions"]] * 0.3)

    aug2, _, info2 = expert_dropout_augment(df, w, aug_frac=0.5, aug_weight=0.3, seed=1,
                                            full_block=True)
    assert info2["full_block"] is True and "traj_x" in info2
    dropped2 = aug2[f"avail__{san}"] == 0
    pd.testing.assert_series_equal(dropped2.reset_index(drop=True), dropped.reset_index(drop=True))
    assert aug2.loc[dropped2, f"traj__{san}__last"].isna().all()
    assert aug2.loc[dropped2, f"mapped_pred_class__{san}"].isna().all()
    assert aug2.loc[dropped2, f"q_prior__{san}"].notna().all()


def test_dropout_spec_validation_and_roundtrip():
    with pytest.raises(ValueError):
        AvailabilityDropout(regime_weight=0.4)          # 3 x 0.4 + 0.4 >= 1
    with pytest.raises(ValueError):
        AvailabilityDropout(regimes=("no_such",))
    spec = AvailabilityDropout(regime_weight=0.2, random_frac=0.1, seed=9)
    assert AvailabilityDropout.from_dict(spec.to_dict()) == spec


# --------------------------------------------------------------------------
# G8: fires on class-pure masking, passes after the fix
# --------------------------------------------------------------------------

def test_g8_fires_on_class_pure_masking_and_passes_after_fix():
    df, tr, ca, te = _build_dataset()
    # Reproduce the v12 LSST situation: the LSST SN train rows are
    # spectroscopic (typed), the LSST 'other' rows catalogue-context.  The
    # LSST equalization then masks the ALeRCE family (incl. local alerce_lc)
    # on every 'other' row and on no SN row -> perfect label proxy.
    lsst_weak = (df["survey"] == "LSST") & (df["label_quality"] == "weak")
    df.loc[lsst_weak, "label_quality"] = "spectroscopic"
    df.loc[lsst_weak, "label_source"] = "tns"
    df.loc[lsst_weak, "tns_type"] = "SN Ia"
    df.loc[lsst_weak, "bts_type"] = "SN Ia"
    with pytest.raises(G8Error, match="G8 violation"):
        HierarchicalFollowup(survey_cal_min=40, n_jobs=2, seed=SEED,
                             g8_max_corr=0.2).fit(df, tr, ca, te)
    # override: recorded, not raised
    m = HierarchicalFollowup(survey_cal_min=40, n_jobs=2, seed=SEED,
                             g8_max_corr=0.2, g8_override=True).fit(df, tr, ca, te)
    assert m.report_["g8"]["status"] == "OVERRIDDEN"
    viol = {(v["survey"], v["expert"]) for v in m.report_["g8"]["violations"]}
    assert {("lsst", "avail__alerce_lc"), ("lsst", "avail__alerce__stamp_classifier"),
            ("lsst", "avail__babamul"), ("lsst", "avail__lasair__sherlock")} <= viol
    assert all(v["origin"] == "masking_or_augmentation" for v in m.report_["g8"]["violations"])
    # alerce_lc: masked on every LSST 'other' row, fires on ~70% of SN rows
    # (fixture availability) -> corr well above the threshold, ~0 in production
    lc = [v for v in m.report_["g8"]["violations"] if v["expert"] == "avail__alerce_lc"]
    assert lc and lc[0]["corr_final"] > 0.4 and lc[0]["corr_production"] < 0.1
    # the fix: equalization off (no ALeRCE-derived label is left on LSST),
    # context family masked on the whole survey -> no expert's availability
    # tracks the label
    m2 = HierarchicalFollowup(
        survey_cal_min=40, n_jobs=2, seed=SEED, head1_exclude_qualities=("lsst:weak",),
        equalize_lsst_provenance=False, context_mask_scope="survey", g8_max_corr=0.2,
    ).fit(df, tr, ca, te)
    assert m2.report_["g8"]["status"] == "PASS"
    assert m2.report_["g8"]["max_abs_corr"] <= 0.2
    assert m2.report_["head1_provenance_masking"]["n_lsst_equalized"] == 0
    assert m2.report_["head1_provenance_masking"]["survey_masks"]["lsst"]["experts"] == [
        "lasair/sherlock", "babamul"]


def test_g8_names_genuine_availability_gaps():
    """An expert only run on one class of objects (production corr already high)
    is reported with origin 'genuine_availability_gap_in_gold'."""
    df = _gold(n_obj=60)
    san = sanitize_expert_key(_LOCAL)
    other = df["target_class"] == "other"
    df.loc[other, f"avail__{san}"] = 0.0
    df.loc[other, f"proj__{san}__p_snia"] = np.nan
    y = df["target_class"].eq("snia").to_numpy(int)
    w = compute_base_weights(df)
    corr = availability_label_corr(df, y, w)
    assert corr["lsst"]["per_expert"][f"avail__{san}"]["corr"] > 0.9
    assert corr["lsst"]["per_expert"][f"avail__{san}"]["avail_other"] == 0.0
    ids = df["object_id"].unique()
    tr, ca = set(ids[:40]), set(ids[40:])
    with pytest.raises(G8Error, match="genuine_availability_gap_in_gold"):
        HierarchicalFollowup(survey_cal_min=5, n_jobs=2, seed=SEED, g8_max_corr=0.2,
                             equalize_lsst_provenance=False).fit(df, tr, ca, set())
    # remedy: mask the expert on the whole survey for head 1
    m = HierarchicalFollowup(survey_cal_min=5, n_jobs=2, seed=SEED, g8_max_corr=0.2,
                             equalize_lsst_provenance=False, context_mask_scope="survey",
                             head1_survey_masks={"lsst": (_LOCAL,), "ztf": (_LOCAL,)},
                             ).fit(df, tr, ca, set())
    assert m.report_["g8"]["status"] == "PASS"


# --------------------------------------------------------------------------
# B4: cross-fit — OOF differs from in-sample, cal rows unchanged
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted_v13():
    df, tr, ca, te = _build_dataset()
    model = HierarchicalFollowup(
        survey_cal_min=40, n_jobs=2, seed=SEED, head1_exclude_qualities=("lsst:weak",),
        equalize_lsst_provenance=False, context_mask_scope="survey",
        dropout=AvailabilityDropout(regime_weight=0.1, random_frac=0.25, seed=SEED),
        cross_fit_folds=3, g8_max_corr=0.2,
    ).fit(df, tr, ca, te)
    return model, df, tr, ca, te


def test_oof_differs_from_in_sample(fitted_v13):
    m, df, tr, ca, te = fitted_v13
    train = df[df["object_id"].isin(tr)]
    cov = m.oof_coverage(train)
    assert cov["n_oof_head1"] > 0 and cov["n_oof_head2"] > 0
    p_in = m.predict_proba(train)
    p_oof = m.predict_proba_oof(train)
    assert np.abs(p_in - p_oof).max() > 1e-3
    assert np.allclose(p_oof.sum(axis=1), 1.0)
    cal = df[df["object_id"].isin(ca)]
    assert m.oof_coverage(cal)["n_oof_head1"] == 0
    np.testing.assert_allclose(m.predict_proba(cal), m.predict_proba_oof(cal))
    # the head-1 calibrators saw OOF-train ∪ cal (same tiers as head-1 training)
    ledger = m.report_["head1_calibration_crossfit"]
    assert ledger["excluded_qualities"] == ["lsst:weak"]
    assert all(v["n_oof_train"] > 0 and v["n_cal"] > 0 for v in ledger["per_survey"].values())
    assert m.report_["cross_fit_head1"]["folds_used"] == 3
    # OOF frame carries the regime mixture + weights for the orchestrator
    assert {"p1_oof_raw", "sample_weight", "row_key", "aug_regime"} <= set(m.oof_frame_.columns)
    assert (m.oof_frame_["aug_regime"] == "random").any()


def test_object_cal_weights_drop_quality_factor(fitted_v13, tmp_path):
    """v13b: 'object' calibration weights remove the context factor, so the
    head-1 calibration frame is weighted by object mix (SN share falls when
    the non-SNe are context rows) and the setting survives save/load."""
    m, df, tr, ca, te = fitted_v13
    assert m.head1_cal_weights == "train"
    b = HierarchicalFollowup(
        survey_cal_min=40, n_jobs=2, seed=SEED, head1_exclude_qualities=("lsst:weak",),
        equalize_lsst_provenance=False, context_mask_scope="survey",
        dropout=AvailabilityDropout(regime_weight=0.1, random_frac=0.25, seed=SEED),
        cross_fit_folds=3, g8_max_corr=0.2, head1_cal_weights="object",
    ).fit(df, tr, ca, te)
    lt, lb = (x.report_["head1_calibration_crossfit"] for x in (m, b))
    assert lt["weights"] == "train" and lb["weights"] == "object"
    # LSST: SNe spectroscopic, non-SNe context -> the SN share falls (on ZTF it
    # can rise: some ZTF SNe carry weak / tns_untyped labels)
    assert lb["per_survey"]["lsst"]["weight_sn_frac"] < lt["per_survey"]["lsst"]["weight_sn_frac"] - 1e-6
    # the heads themselves are unchanged: only the calibrators differ
    np.testing.assert_allclose(m._head1_p(df, calibrated=False), b._head1_p(df, calibrated=False))
    q = b._quality_factor(pd.DataFrame({"label_quality": ["spectroscopic", "context", "weak", None]}))
    np.testing.assert_allclose(q, [1.0, b.context_weight, b.weak_weight, b.weak_weight])
    b.save(str(tmp_path))
    assert HierarchicalFollowup.load(str(tmp_path)).head1_cal_weights == "object"


def test_v13_save_load_roundtrip(fitted_v13, tmp_path):
    m, df, tr, _, _ = fitted_v13
    m.save(str(tmp_path))
    assert (tmp_path / "head1_oof.parquet").exists()
    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["v13"]["cross_fit_folds"] == 3
    assert meta["v13"]["dropout"]["regimes"] == ["no_broker", "no_local", "no_expert"]
    loaded = HierarchicalFollowup.load(str(tmp_path))
    assert loaded.dropout == m.dropout and loaded.head1_exclude_qualities == ("lsst:weak",)
    train = df[df["object_id"].isin(tr)]
    np.testing.assert_allclose(loaded.predict_proba(df), m.predict_proba(df))
    np.testing.assert_allclose(loaded.predict_proba_oof(train), m.predict_proba_oof(train))


# --------------------------------------------------------------------------
# defaults reproduce the old behaviour
# --------------------------------------------------------------------------

def test_defaults_reproduce_old_behaviour():
    df, tr, ca, te = _build_dataset()
    default = HierarchicalFollowup(survey_cal_min=40, n_jobs=2, seed=SEED).fit(df, tr, ca, te)
    explicit = HierarchicalFollowup(
        survey_cal_min=40, n_jobs=2, seed=SEED, head1_exclude_qualities=(),
        equalize_lsst_provenance=True, context_mask_scope="rows", head1_survey_masks={},
        dropout=None, cross_fit_folds=0, g8_max_corr=None,
    ).fit(df, tr, ca, te)
    np.testing.assert_array_equal(default.predict_proba(df), explicit.predict_proba(df))
    np.testing.assert_array_equal(default.predict_proba_raw(df), explicit.predict_proba_raw(df))
    # v12 masking ledger: equalization ON, no survey masks, no v13 ledgers
    pm = default.report_["head1_provenance_masking"]
    assert pm["n_lsst_equalized"] > 0 and pm["survey_masks"] == {}
    for key in ("g8", "head1_dropout", "cross_fit_head1", "head1_calibration_crossfit"):
        assert key not in default.report_
    assert default.oof_frame_ is None and default.head1_oof_ is None
    assert default.report_["head1_excluded_train_rows"] == 0
    # predict_proba_oof degenerates to predict_proba
    np.testing.assert_array_equal(default.predict_proba_oof(df), default.predict_proba(df))
    # the unweighted calibrator classes are the v11 ones
    from debass_meta.models.calibrate import IsotonicCalibrator
    from debass_meta.models.pooled_trust import PlattCalibrator
    assert type(default.head1_calibrators["ztf"]) is IsotonicCalibrator
    assert type(default.head1_calibrators["lsst"]) is PlattCalibrator


def test_load_pre_v13_metadata(tmp_path):
    """An artifact saved without the 'v13' block (v11/v12) loads with defaults."""
    df, tr, ca, te = _build_dataset()
    m = HierarchicalFollowup(survey_cal_min=40, n_jobs=2, seed=SEED).fit(df, tr, ca, te)
    m.save(str(tmp_path))
    meta = json.loads((tmp_path / "metadata.json").read_text())
    meta.pop("v13")
    meta["hyperparameters"].pop("equalize_lsst_provenance")
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    loaded = HierarchicalFollowup.load(str(tmp_path))
    assert loaded.dropout is None and loaded.cross_fit_folds == 0
    assert loaded.equalize_lsst_provenance is True and loaded.context_mask_scope == "rows"
    np.testing.assert_allclose(loaded.predict_proba(df), m.predict_proba(df))


# --------------------------------------------------------------------------
# B1: survey-scoped quality exclusion (head + anchor blend)
# --------------------------------------------------------------------------

def test_exclusion_is_survey_scoped():
    df = pd.DataFrame({
        "object_id": list("abcdef"), "n_det": [1] * 6,
        "survey": ["LSST", "LSST", "ZTF", "ZTF", "LSST", "ZTF"],
        "label_quality": ["weak", "spectroscopic", "weak", "spectroscopic", "context", "context"],
    })
    m = excluded_quality_mask(df, ("lsst:weak",))
    assert m.tolist() == [True, False, False, False, False, False]
    assert exclude_qualities(df, ("weak",))["object_id"].tolist() == list("bdef")
    assert exclude_qualities(df, ()) is df
    # same grammar in the anchor-blend honesty filter
    df["label_source"] = "tns"
    keep = anchor_blend._honesty_mask(df, exclude_qualities=("lsst:weak",))
    assert keep.tolist() == [False, True, True, True, True, True]
    keep = anchor_blend._honesty_mask(df, exclude_qualities=("weak",))
    assert keep.tolist() == [False, True, False, True, True, True]
    assert anchor_blend._honesty_mask(df).all()


def test_head_exclusion_keeps_ztf_weak_rows():
    df, tr, ca, te = _build_dataset()
    m = HierarchicalFollowup(survey_cal_min=40, n_jobs=2, seed=SEED,
                             head1_exclude_qualities=("lsst:weak",)).fit(df, tr, ca, te)
    train = df[df["object_id"].isin(tr)]
    n_lsst_weak = int(((train["survey"] == "LSST") & (train["label_quality"] == "weak")).sum())
    assert n_lsst_weak > 0
    assert m.report_["head1_excluded_train_rows"] == n_lsst_weak
    assert m.report_["v13"]["head1_exclude_qualities"] == ["lsst:weak"]


def test_fit_alpha_uniform_weights_match_unweighted():
    from test_anchor_blend import _cal_frame

    df = _cal_frame(240, anchor_good=True, model_good=False, seed=4)
    spec0 = anchor_blend.fit_alpha(df)
    df["sample_weight"] = 1.0
    spec1 = anchor_blend.fit_alpha(df, weight_col="sample_weight")
    assert spec0.alpha_cells.keys() == spec1.alpha_cells.keys()
    for ck in spec0.alpha_cells:
        assert spec0.alpha_cells[ck]["alpha"] == spec1.alpha_cells[ck]["alpha"]
    assert spec0.alpha_survey == spec1.alpha_survey
    assert spec1.g3["fit_frame"]["weight_col"] == "sample_weight"
    # zero-weight rows are dropped from the usable set
    df.loc[: len(df) // 2, "sample_weight"] = 0.0
    spec2 = anchor_blend.fit_alpha(df, weight_col="sample_weight")
    assert spec2.alpha_global["n"] < spec1.alpha_global["n"]


# --------------------------------------------------------------------------
# B5: orchestrator flags end to end (synthetic; --skip-stage-a)
# --------------------------------------------------------------------------

def test_orchestrator_v13_flags(tmp_path):
    from test_train_v11_smoke import _argv, _write_inputs

    from scripts.train_fusion_v11 import main

    snap, split_path, locked_path, truth_path = _write_inputs(tmp_path)
    extra = ("--head1-exclude-quality", "lsst:weak", "--no-lsst-equalization",
             "--head1-context-mask", "survey", "--head1-dropout",
             "--dropout-aug-weight", "0.1", "--dropout-random-frac", "0.25",
             "--cross-fit-folds", "3", "--g8-max-corr", "0.2",
             "--stage-a-weak-policy", "is_sn_only", "--stage-a-q-prior-experts", "trained")
    rc = main(_argv(tmp_path, snap, split_path, locked_path, truth_path, extra=extra))
    assert rc == 0
    report = json.loads((tmp_path / "fusion_v11_train.json").read_text())
    assert report["v13"]["head1_exclude_qualities"] == ["lsst:weak"]
    assert report["v13"]["equalize_lsst_provenance"] is False
    assert report["v13"]["context_mask_scope"] == "survey"
    assert report["v13"]["dropout"]["regime_weight"] == 0.1
    assert report["v13"]["cross_fit_folds"] == 3
    assert report["guards"]["G8"]["status"] == "PASS"
    assert report["guard_statuses"]["G8"] == "PASS"
    assert report["alpha_fit_frame"]["mode"] == "oof_train_mixture+cal_mixture"
    assert report["g2_frame"]["mode"] == "oof_train+cal"
    assert "head1_dropout" in report and "cross_fit_head1" in report
    meta = json.loads((tmp_path / "followup_v11" / "metadata.json").read_text())
    assert meta["v13"]["g8_max_corr"] == 0.2
    blend = json.loads((tmp_path / "blend_v11" / "blend.json").read_text())
    assert blend["g3"]["fit_frame"]["weight_col"] == "sample_weight"
    assert blend["g3"]["fit_frame"]["exclude_qualities"] == ["lsst:weak"]

    # G8 failure path: v12 equalization + weak rows -> SystemExit, report written
    (tmp_path / "fail").mkdir()
    snap2, split2, locked2, truth2 = _write_inputs(tmp_path / "fail")
    with pytest.raises(SystemExit, match="G8"):
        main(_argv(tmp_path / "fail", snap2, split2, locked2, truth2,
                   extra=("--g8-max-corr", "0.2")))
    rep = json.loads((tmp_path / "fail" / "fusion_v11_train.json").read_text())
    assert rep["guards"]["G8"]["status"] == "FAIL"
    # …and --acknowledge-g8 records OVERRIDDEN and exits 0
    rc = main(_argv(tmp_path / "fail", snap2, split2, locked2, truth2,
                    extra=("--g8-max-corr", "0.2", "--acknowledge-g8")))
    assert rc == 0
    rep = json.loads((tmp_path / "fail" / "fusion_v11_train.json").read_text())
    assert rep["guards"]["G8"]["status"] == "OVERRIDDEN"


def test_orchestrator_defaults_have_no_v13_effects(tmp_path):
    from test_train_v11_smoke import _argv, _write_inputs

    from scripts.train_fusion_v11 import main

    snap, split_path, locked_path, truth_path = _write_inputs(tmp_path)
    assert main(_argv(tmp_path, snap, split_path, locked_path, truth_path)) == 0
    report = json.loads((tmp_path / "fusion_v11_train.json").read_text())
    assert "G8" not in report["guards"]
    assert report["v13"]["dropout"] is None and report["v13"]["cross_fit_folds"] == 0
    assert report["alpha_fit_frame"]["mode"] == "cal"
    assert report["g2_frame"]["mode"] == "in_sample_train+cal"
    blend = json.loads((tmp_path / "blend_v11" / "blend.json").read_text())
    assert "fit_frame" not in blend["g3"]


# --------------------------------------------------------------------------
# follow-ups: serving parity of survey masks, global expert drop, dry run
# --------------------------------------------------------------------------

def _blank_on_frame(df: pd.DataFrame, experts, rows=None, q_prior=False) -> pd.DataFrame:
    out = df.copy()
    rows = np.ones(len(out), bool) if rows is None else np.asarray(rows, bool)
    blank_expert_blocks(out, rows, [sanitize_expert_key(k) for k in experts],
                        recompute_cross_traj=True, include_q_prior=q_prior)
    return out


def test_survey_masks_apply_identically_at_serve():
    """A fitted model with survey masks predicts the same on a frame where the
    masked experts are present and on the same frame with their blocks
    blanked — head 1 never sees Sherlock/Babamul on LSST, at fit or at serve."""
    df, tr, ca, te = _build_dataset()
    rng = np.random.default_rng(SEED)
    df["q_prior__fink__snn"] = np.where(df["avail__fink__snn"] == 1, rng.uniform(0, 1, len(df)), np.nan)
    m = HierarchicalFollowup(
        survey_cal_min=40, n_jobs=2, seed=SEED, head1_exclude_qualities=("lsst:weak",),
        equalize_lsst_provenance=False, context_mask_scope="survey",
        head1_survey_masks={"ztf": ("fink/snn",)},
    ).fit(df, tr, ca, te)
    assert m.head1_serving_masks == {"ztf": ("fink/snn",),
                                     "lsst": ("lasair/sherlock", "babamul")}
    lsst = df["survey"] == "LSST"
    blanked = _blank_on_frame(df, ["lasair/sherlock", "babamul"], rows=lsst)
    blanked = _blank_on_frame(blanked, ["fink/snn"], rows=~lsst)
    # the masked experts ARE present on the raw frame
    assert (df.loc[lsst, "avail__babamul"] == 1).any() and (df.loc[~lsst, "avail__fink__snn"] == 1).any()
    # head 1 (raw and calibrated) is identical with the experts present or blanked
    for calibrated in (False, True):
        np.testing.assert_array_equal(m._head1_p(df, calibrated=calibrated),
                                      m._head1_p(blanked, calibrated=calibrated))
    # survey masks are head-1 only: head 2 still uses fink/snn on ZTF, so the
    # composed probabilities agree on LSST rows (head 2 = base rate there)
    for fn in (m.predict_proba, m.predict_proba_raw, m.predict_proba_oof):
        np.testing.assert_array_equal(fn(df[lsst]), fn(blanked[lsst]))
    assert np.abs(m._head2_p(df, calibrated=False) - m._head2_p(blanked, calibrated=False)).max() > 0
    # an explicitly named mask also blanks the expert's q_prior (v13b)
    blanked_q = _blank_on_frame(blanked, ["fink/snn"], rows=~lsst, q_prior=True)
    assert blanked_q.loc[~lsst, "q_prior__fink__snn"].isna().all()
    assert df.loc[~lsst, "q_prior__fink__snn"].notna().any()
    np.testing.assert_array_equal(m._head1_p(df, calibrated=False),
                                  m._head1_p(blanked_q, calibrated=False))
    # …and head 1 genuinely depends on an unmasked expert
    other = _blank_on_frame(df, ["alerce_lc"])
    assert np.abs(m._head1_p(df, calibrated=False) - m._head1_p(other, calibrated=False)).max() > 0
    # persisted through save/load
    import tempfile
    d = tempfile.mkdtemp()
    m.save(d)
    loaded = HierarchicalFollowup.load(d)
    assert loaded.head1_serving_masks == m.head1_serving_masks
    np.testing.assert_array_equal(loaded._head1_p(df, calibrated=True),
                                  m._head1_p(blanked, calibrated=True))


def test_drop_experts_blank_both_heads_at_fit_and_serve():
    df, tr, ca, te = _build_dataset()
    m = HierarchicalFollowup(survey_cal_min=40, n_jobs=2, seed=SEED,
                             drop_experts=("fink/snn",)).fit(df, tr, ca, te)
    assert not any(c.startswith(("proj__fink__snn", "q__fink__snn", "avail__fink__snn"))
                   for c in m.head1_feature_cols + m.head2_feature_cols)
    blanked = _blank_on_frame(df, ["fink/snn"], q_prior=True)
    for fn in (m.predict_proba, m.predict_proba_raw, m.predict_proba_oof):
        np.testing.assert_array_equal(fn(df), fn(blanked))
    p2_raw = m._head2_p(df, calibrated=False)
    np.testing.assert_array_equal(p2_raw, m._head2_p(blanked, calibrated=False))
    meta_drop = m._metadata()["v13"]["drop_experts"]
    assert meta_drop == ["fink/snn"]
    import tempfile
    d = tempfile.mkdtemp()
    m.save(d)
    loaded = HierarchicalFollowup.load(d)
    assert loaded.drop_experts == ("fink/snn",)
    np.testing.assert_array_equal(loaded.predict_proba(df), m.predict_proba(df))


def test_blendspec_drop_experts_recomputes_anchor_at_apply():
    from test_anchor_blend import _cal_frame

    df = _cal_frame(120, anchor_good=True, model_good=False, seed=6)
    san = sanitize_expert_key("fink/snn")
    assert f"proj__{san}__p_snia" in df.columns
    spec = anchor_blend.fit_alpha(df, drop_experts=("fink/snn",))
    assert spec.drop_experts == ["fink/snn"]
    assert anchor_blend.BlendSpec.from_dict(spec.to_dict()).drop_experts == ["fink/snn"]
    # a caller-computed anchor WITH fink/snn is overridden by apply()
    with_snn = anchor_blend.compute_anchor(df)
    out = anchor_blend.apply(with_snn, spec)
    without = anchor_blend.compute_anchor(df, exclude_experts=("fink/snn",))
    np.testing.assert_allclose(out["p_snia_anchor"], without["p_snia_anchor"])
    np.testing.assert_array_equal(out["n_experts_fired"], without["n_experts_fired"])
    assert (with_snn["n_experts_fired"] > without["n_experts_fired"]).any()
    # a spec without drops keeps the caller's anchor (v11/v12 path)
    spec0 = anchor_blend.fit_alpha(df)
    assert "drop_experts" not in spec0.to_dict()
    np.testing.assert_allclose(anchor_blend.apply(with_snn, spec0)["p_snia_anchor"],
                               with_snn["p_snia_anchor"])


def test_g8_dry_run_matches_fit_and_never_raises():
    df, tr, ca, te = _build_dataset()
    lsst_weak = (df["survey"] == "LSST") & (df["label_quality"] == "weak")
    df.loc[lsst_weak, ["label_quality", "label_source"]] = ["spectroscopic", "tns"]
    df.loc[lsst_weak, ["tns_type", "bts_type"]] = "SN Ia"
    head = HierarchicalFollowup(survey_cal_min=40, n_jobs=2, seed=SEED, g8_max_corr=0.2)
    g8 = head.g8_dry_run(df, tr, ca)
    assert g8["dry_run"] is True and g8["status"] == "OVERRIDDEN"
    assert {v["expert"] for v in g8["violations"]} >= {"avail__alerce_lc", "avail__babamul"}
    e = g8["per_survey"]["lsst"]["per_expert"]["avail__alerce_lc"]
    assert e["avail_other"] == 0.0 and e["avail_sn"] > 0.5 and e["avail_other_production"] > 0.5
    # same numbers as the guard inside fit (override)
    m = HierarchicalFollowup(survey_cal_min=40, n_jobs=2, seed=SEED, g8_max_corr=0.2,
                             g8_override=True).fit(df, tr, ca, te)
    assert m.report_["g8"]["max_abs_corr"] == pytest.approx(g8["max_abs_corr"])
    # dry run with the v13 settings + drop: passes, drop blanks the expert
    head2 = HierarchicalFollowup(
        survey_cal_min=40, n_jobs=2, seed=SEED, g8_max_corr=0.2,
        head1_exclude_qualities=("lsst:weak",), equalize_lsst_provenance=False,
        context_mask_scope="survey", drop_experts=("fink/snn",),
        dropout=AvailabilityDropout(regime_weight=0.1, random_frac=0.25, seed=SEED))
    g8b = head2.g8_dry_run(df, tr, ca)
    assert g8b["status"] == "PASS"
    assert "avail__fink__snn" not in g8b["per_survey"]["lsst"]["per_expert"]
    assert g8b["head1_dropout"]["regimes"]["no_broker"]["n_rows"] > 0


def test_orchestrator_drop_expert_and_dry_run(tmp_path):
    from test_train_v11_smoke import _argv, _write_inputs

    from scripts.train_fusion_v11 import drop_expert_columns, drop_expert_helpfulness, main

    snap, split_path, locked_path, truth_path = _write_inputs(tmp_path)
    # helper semantics
    g = pd.read_parquet(snap)
    d = drop_expert_columns(g, ["fink/snn"])
    assert d["proj__fink__snn__p_snia"].isna().all() and (d["avail__fink__snn"] == 0).all()
    assert d["q__fink__snn"].isna().all()
    assert g["proj__fink__snn__p_snia"].notna().any()
    h = pd.DataFrame({"expert_key": ["fink/snn", "seq_v11"], "x": [1, 2]})
    assert drop_expert_helpfulness(h, ["fink/snn"])["expert_key"].tolist() == ["seq_v11"]
    # dry run: report written, no artifacts, exit 0
    rc = main(_argv(tmp_path, snap, split_path, locked_path, truth_path,
                    extra=("--g8-dry-run", "--g8-max-corr", "0.2", "--head1-dropout",
                           "--drop-expert", "fink/snn", "--head1-context-mask", "survey")))
    assert rc == 0
    rep = json.loads((tmp_path / "fusion_v11_train.json").read_text())
    assert rep["guards"]["G8"]["dry_run"] is True
    assert "g8_dry_run_table" in rep and rep["drop_experts"] == ["fink/snn"]
    assert not (tmp_path / "followup_v11").exists()
    # full run with the drop: head + blend carry it; scoring parity
    rc = main(_argv(tmp_path, snap, split_path, locked_path, truth_path,
                    extra=("--drop-expert", "fink/snn", "--stage-a-weak-policy", "lsst_is_sn_only")))
    assert rc == 0
    meta = json.loads((tmp_path / "followup_v11" / "metadata.json").read_text())
    assert meta["v13"]["drop_experts"] == ["fink/snn"]
    blend = json.loads((tmp_path / "blend_v11" / "blend.json").read_text())
    assert blend["drop_experts"] == ["fink/snn"]
    loaded = HierarchicalFollowup.load(str(tmp_path / "followup_v11"))
    np.testing.assert_array_equal(loaded.predict_proba(g),
                                  loaded.predict_proba(_blank_on_frame(g, ["fink/snn"], q_prior=True)))
