"""Unit tests for the fusion_v8 G4 modules (multiclass follow-up head,
Mondrian conformal sets, budget-aware selection).

All tests run on self-contained synthetic data (no data/ access).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from debass_meta.models.conformal import MondrianAPS
from debass_meta.models.multiclass_followup import (
    CLASSES,
    MulticlassFollowupArtifact,
    _fit_calibrator_ladder,
    _weighted_log_loss,
    apply_provenance_masking,
    compute_base_weights,
    expert_dropout_augment,
    train_multiclass_followup,
)
from debass_meta.models.selection import (
    UTILITY_PRESETS,
    fdr_controlled_threshold,
    rank_and_select,
)
from debass_meta.models.splitters import group_train_cal_test_split

SEED = 42

# Sanitized expert keys used in the synthetic expert blocks (real registry
# names so provenance masking has the right targets).
_SANS = (
    "fink__snn",
    "alerce__stamp_classifier",
    "alerce_lc",
    "lasair__sherlock",
    "babamul",
)
_CLASS_P_SNIA = {0: 0.8, 1: 0.3, 2: 0.1}


def _make_snapshots(n_objects: int = 600, seed: int = SEED) -> pd.DataFrame:
    """Blob-ish 3-class synthetic snapshot table (~n_objects × ~8 epochs)."""
    rng = np.random.default_rng(seed)
    centers = np.array([
        [0.0, 0.0, 2.0, -1.0, 0.5, 1.0],
        [3.0, -2.0, -1.0, 2.0, -0.5, -1.0],
        [-3.0, 3.0, 0.0, -2.0, 2.5, 0.0],
    ])
    rows: list[dict] = []
    for i in range(n_objects):
        cls = int(rng.choice(3, p=[0.4, 0.35, 0.25]))
        n_epochs = int(rng.integers(4, 9))
        survey = "LSST" if rng.random() < 0.2 else "ZTF"
        object_id = f"OBJ{i:06d}"
        for n_det in range(1, n_epochs + 1):
            row: dict = {
                "object_id": object_id,
                "n_det": n_det,
                "survey": survey,
                "target_class": CLASSES[cls],
                "label_quality": "spectroscopic",
                "label_source": "tns",
                "alert_jd": 2460000.0 + n_det,  # blocked from features
                "traj_x__mean_slope": float(rng.normal()),
            }
            feats = centers[cls] + rng.normal(0.0, 2.2, size=6)
            for j, value in enumerate(feats):
                row[f"lcf_{j}"] = float(value)
            for san in _SANS:
                if rng.random() < 0.7:
                    p_snia = float(np.clip(
                        _CLASS_P_SNIA[cls] + rng.normal(0.0, 0.3), 0.0, 1.0
                    ))
                    row[f"proj__{san}__p_snia"] = p_snia
                    row[f"proj__{san}__top1_prob"] = float(
                        rng.uniform(0.5, 1.0)
                    )
                    row[f"avail__{san}"] = 1.0
                    row[f"exact__{san}"] = 1.0
                    row[f"q__{san}"] = float(rng.uniform(0.3, 0.9))
                else:
                    row[f"proj__{san}__p_snia"] = np.nan
                    row[f"proj__{san}__top1_prob"] = np.nan
                    row[f"avail__{san}"] = 0.0
                    row[f"exact__{san}"] = 0.0
                    row[f"q__{san}"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Train the multiclass head once (grid_small) and share across tests."""
    snapshots = _make_snapshots()
    labels = (
        snapshots.groupby("object_id")["target_class"].first().to_dict()
    )
    split = group_train_cal_test_split(labels, seed=SEED)
    out_dir = tmp_path_factory.mktemp("followup_fusion_v8")
    metrics = train_multiclass_followup(
        snapshots,
        split.train_ids,
        split.cal_ids,
        split.test_ids,
        str(out_dir),
        grid_small=True,
        n_jobs=2,
        seed=SEED,
    )
    return snapshots, split, str(out_dir), metrics


# ---------------------------------------------------------------------------
# (a) training converges; calibration ladder guard holds
# ---------------------------------------------------------------------------

def test_training_converges(trained):
    _, _, _, metrics = trained
    assert metrics["grid"]["n_configs"] == 2  # grid_small
    assert metrics["grid"]["best_iteration"] >= 1
    test_cal = metrics["test_calibrated"]
    assert test_cal["n_rows"] > 0
    assert test_cal["macro_ovr_auc"] is not None
    assert test_cal["macro_ovr_auc"] > 0.80, (
        f"separable blobs should be learnable, got "
        f"macro AUC={test_cal['macro_ovr_auc']:.3f}"
    )


def test_calibration_improves_or_fallback_fires(trained):
    _, _, _, metrics = trained
    calibration = metrics["calibration"]
    if calibration["kind"] == "none":
        # fallback fired — the ladder refused every candidate
        assert calibration.get("skip_reason") is not None
    else:
        assert calibration["cal_cv_logloss_chosen"] is not None
        assert (
            calibration["cal_cv_logloss_chosen"]
            <= calibration["cal_cv_logloss_raw"] + 1e-9
        ), "chosen calibrator must not worsen cal-CV log-loss"


def test_calibrator_ladder_fixes_overconfidence():
    """On deliberately miscalibrated (overconfident) probabilities the
    ladder must pick a calibrator that beats raw cal-CV log-loss."""
    rng = np.random.default_rng(5)
    n = 1200
    p_true = rng.dirichlet((1.5, 1.2, 0.9), size=n)
    u = rng.random(n)
    y = (u[:, None] > p_true.cumsum(axis=1)).sum(axis=1).astype(int)
    # overconfident: sharpen p_true (p^3, renormalized)
    p_over = p_true ** 3
    p_over = p_over / p_over.sum(axis=1, keepdims=True)
    weights = np.ones(n)
    groups = np.array([f"G{i}" for i in range(n)])

    calibrator, kind, info = _fit_calibrator_ladder(
        p_over, y, weights, groups, seed=SEED
    )
    assert kind != "none"
    assert info["cal_cv_logloss_chosen"] < info["cal_cv_logloss_raw"]
    # calibrated probs recover most of the gap to the true probabilities
    loss_over = _weighted_log_loss(y, p_over)
    loss_cal = _weighted_log_loss(y, calibrator.transform(p_over))
    loss_true = _weighted_log_loss(y, p_true)
    assert loss_cal < loss_over
    assert loss_cal < loss_true + 0.05


def test_artifact_roundtrip(trained):
    snapshots, split, out_dir, metrics = trained
    artifact = MulticlassFollowupArtifact.load(out_dir)
    assert artifact.feature_cols == sorted(artifact.feature_cols)
    # label-derived columns must never be features
    for blocked in ("target_class", "label_quality", "label_source",
                    "alert_jd", "object_id"):
        assert blocked not in artifact.feature_cols
    test_df = snapshots[
        snapshots["object_id"].isin(split.test_ids)
    ].head(50)
    proba = artifact.predict_proba(test_df)
    assert proba.shape == (len(test_df), 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    raw = artifact.predict_proba_raw(test_df)
    assert raw.shape == (len(test_df), 3)


# ---------------------------------------------------------------------------
# (b) conformal empirical coverage on held-out synthetic data
# ---------------------------------------------------------------------------

def _simulated_probs(n: int, rng: np.random.Generator):
    """Perfectly calibrated probabilities: y ~ Categorical(p)."""
    proba = rng.dirichlet((1.5, 1.2, 0.9), size=n)
    u = rng.random(n)
    y = (u[:, None] > proba.cumsum(axis=1)).sum(axis=1)
    strata = pd.DataFrame({
        "survey": rng.choice(["ZTF", "LSST"], size=n, p=[0.7, 0.3]),
        "n_det": rng.integers(1, 21, size=n),
        "object_id": [f"C{i}" for i in range(n)],
    })
    return proba, y.astype(int), strata


def test_conformal_coverage(tmp_path):
    rng = np.random.default_rng(7)
    proba_cal, y_cal, strata_cal = _simulated_probs(3000, rng)
    proba_test, y_test, strata_test = _simulated_probs(3000, rng)

    aps = MondrianAPS()
    aps.fit(proba_cal, y_cal, strata_cal, alpha=0.1)
    sets = aps.predict_sets(proba_test, strata_test)
    assert sets.shape == (3000, 3)
    coverage = float(sets[np.arange(len(y_test)), y_test].mean())
    assert 0.85 <= coverage <= 0.95, f"coverage {coverage:.3f} outside [0.85, 0.95]"

    report = aps.coverage_report(proba_test, y_test, strata_test)
    global_row = report[report["level"] == "global"].iloc[0]
    assert abs(float(global_row["coverage"]) - coverage) < 1e-9
    assert 1.0 <= float(global_row["mean_set_size"]) <= 3.0

    # save/load roundtrip reproduces identical sets
    path = tmp_path / "conformal_fusion_v8.json"
    aps.save(path)
    loaded = MondrianAPS.load(path)
    sets2 = loaded.predict_sets(proba_test, strata_test)
    assert (sets == sets2).all()


# ---------------------------------------------------------------------------
# (c) hierarchy fallback fires for starved buckets
# ---------------------------------------------------------------------------

def test_conformal_hierarchy_fallback():
    rng = np.random.default_rng(11)
    blocks = [
        (0, "ZTF", 5, 300),    # rich: full class×survey×n_det stratum
        (1, "ZTF", 5, 300),
        (2, "ZTF", 5, 10),     # starved class everywhere -> global
        (0, "LSST", 15, 5),    # starved survey -> class level
    ]
    y_list, survey_list, ndet_list = [], [], []
    for cls, survey, n_det, count in blocks:
        y_list += [cls] * count
        survey_list += [survey] * count
        ndet_list += [n_det] * count
    y = np.array(y_list, dtype=int)
    n = len(y)
    proba = rng.dirichlet((1.0, 1.0, 1.0), size=n)
    strata = pd.DataFrame({"survey": survey_list, "n_det": ndet_list})

    aps = MondrianAPS(min_count=30)
    aps.fit(proba, y, strata, alpha=0.1)

    _, level_rich = aps.threshold_for(0, "ZTF", 5)
    assert level_rich == "class_survey_ndet"
    _, level_survey_starved = aps.threshold_for(0, "LSST", 15)
    assert level_survey_starved == "class"   # (0, LSST) has only 5 objects
    _, level_class_starved = aps.threshold_for(2, "LSST", 15)
    assert level_class_starved == "global"   # class 2 has only 10 objects


# ---------------------------------------------------------------------------
# (d) selection demotes full-set objects and respects budget
# ---------------------------------------------------------------------------

def _make_pred_frame() -> pd.DataFrame:
    # rows 0-4: decided Ia-eligible; row 5: ineligible for Ia; row 6: full set
    return pd.DataFrame({
        "object_id": [f"S{i}" for i in range(7)],
        "p_snia": [0.90, 0.85, 0.80, 0.75, 0.70, 0.20, 0.95],
        "p_nonia": [0.05, 0.10, 0.15, 0.20, 0.20, 0.50, 0.03],
        "p_other": [0.05, 0.05, 0.05, 0.05, 0.10, 0.30, 0.02],
        "set_snia": [1, 1, 1, 1, 1, 0, 1],
        "set_nonia": [0, 1, 0, 0, 1, 1, 1],
        "set_other": [0, 0, 0, 1, 0, 1, 1],
        "traj_x__mean_slope": [0.1, 0.2, 0.0, -0.1, 0.3, 0.0, 0.5],
    })


def test_selection_demotion_budget_eligibility():
    pred = _make_pred_frame()
    out = rank_and_select(pred, "ia", budget=3)

    # full-set row 6 has the highest p_snia but is demoted below ALL decided
    assert bool(out.loc[6, "demoted"])
    assert not bool(out.loc[6, "selected"])
    decided_ranks = out.loc[[0, 1, 2, 3, 4], "rank"]
    assert float(out.loc[6, "rank"]) > float(decided_ranks.max())

    # ineligible row 5 (set excludes snia) is never ranked or selected
    assert not bool(out.loc[5, "eligible"])
    assert not bool(out.loc[5, "selected"])
    assert np.isnan(out.loc[5, "rank"])

    # budget respected; top-3 decided by p_snia
    assert int(out["selected"].sum()) == 3
    assert list(out.index[out["selected"]]) == [0, 1, 2]
    assert out.attrs["expected_yield"] == pytest.approx(0.90 + 0.85 + 0.80)
    assert out.attrs["expected_purity"] == pytest.approx(
        out.attrs["expected_yield"] / 3.0
    )

    # with a budget exceeding the decided pool, the demoted row IS selected
    # (demotion, not exclusion — spec correction #4d)
    out_wide = rank_and_select(pred, "ia", budget=6)
    assert bool(out_wide.loc[6, "selected"])
    assert int(out_wide["selected"].sum()) == 6

    # preset and explicit vector agree
    out_vec = rank_and_select(pred, UTILITY_PRESETS["ia"], budget=3)
    assert np.allclose(
        out_vec["priority_score"].to_numpy(),
        out["priority_score"].to_numpy(),
    )


def test_fdr_threshold():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    is_target = np.array([1, 1, 0, 1])
    assert fdr_controlled_threshold(scores, is_target, gamma=0.2) == pytest.approx(0.8)
    assert fdr_controlled_threshold(scores, is_target, gamma=0.30) == pytest.approx(0.6)
    assert fdr_controlled_threshold(scores, 1 - is_target, gamma=0.0) == float("inf")

    # threshold plumbs through rank_and_select as an eligibility filter
    pred = _make_pred_frame()
    out = rank_and_select(pred, "ia", budget=5, score_threshold=0.78)
    assert not bool(out.loc[3, "eligible"])  # p_snia = 0.75 < 0.78
    assert bool(out.loc[0, "selected"])


# ---------------------------------------------------------------------------
# (e) expert-dropout augmentation: row counts, weights, labels
# ---------------------------------------------------------------------------

def test_expert_dropout_augmentation():
    df = _make_snapshots(n_objects=50, seed=1)
    base_w = np.full(len(df), 0.5)
    avail_cols = [c for c in df.columns if c.startswith("avail__")]
    n_eligible = int((df[avail_cols].fillna(0).sum(axis=1) > 0).sum())

    aug, weights, info = expert_dropout_augment(
        df, base_w, aug_frac=0.5, aug_weight=0.3, seed=SEED
    )

    assert info["n_eligible"] == n_eligible
    assert info["n_aug"] == round(0.5 * n_eligible)
    assert len(aug) == info["n_aug"]
    assert np.allclose(weights, 0.5 * 0.3)
    assert (aug["is_aug"] == 1.0).all()

    src = df.iloc[info["source_positions"]].reset_index(drop=True)
    # labels NEVER change
    assert (
        aug["target_class"].to_numpy() == src["target_class"].to_numpy()
    ).all()
    # every augmented copy dropped at least one available expert
    src_avail = src[avail_cols].fillna(0).to_numpy().sum(axis=1)
    aug_avail = aug[avail_cols].fillna(0).to_numpy().sum(axis=1)
    assert (aug_avail < src_avail).all()
    # dropped experts: proj -> NaN, avail/exact -> 0.0, q -> NaN
    for san in _SANS:
        dropped = (
            (src[f"avail__{san}"].to_numpy() > 0.5)
            & (aug[f"avail__{san}"].to_numpy() < 0.5)
        )
        if not dropped.any():
            continue
        assert aug.loc[dropped, f"proj__{san}__p_snia"].isna().all()
        assert aug.loc[dropped, f"q__{san}"].isna().all()
        assert (aug.loc[dropped, f"exact__{san}"] == 0.0).all()

    # determinism
    aug2, weights2, info2 = expert_dropout_augment(
        df, base_w, aug_frac=0.5, aug_weight=0.3, seed=SEED
    )
    pd.testing.assert_frame_equal(aug, aug2)
    assert np.array_equal(info["source_positions"], info2["source_positions"])

    # aug_frac=0 is a clean no-op
    aug0, w0, info0 = expert_dropout_augment(df, base_w, aug_frac=0.0)
    assert len(aug0) == 0 and len(w0) == 0 and info0["n_aug"] == 0


# ---------------------------------------------------------------------------
# (f) provenance masking NaNs exactly the right columns
# ---------------------------------------------------------------------------

def test_provenance_masking():
    cols = {}
    for san in _SANS:
        cols[f"proj__{san}__p_snia"] = [0.7, 0.7, 0.7]
        cols[f"avail__{san}"] = [1.0, 1.0, 1.0]
        cols[f"exact__{san}"] = [1.0, 1.0, 1.0]
        cols[f"q__{san}"] = [0.6, 0.6, 0.6]
    df = pd.DataFrame({
        "object_id": ["A", "B", "C"],
        "target_class": ["other", "other", "snia"],
        "label_source": ["alerce_self_label", "fink_xm_host_context", "tns"],
        "label_quality": ["weak", "context", "spectroscopic"],
        **cols,
    })
    masked = apply_provenance_masking(df)

    # row 0 (alerce self-label): ALL alerce-family blocks masked
    for san in ("alerce__stamp_classifier", "alerce_lc"):
        assert np.isnan(masked.loc[0, f"proj__{san}__p_snia"])
        assert np.isnan(masked.loc[0, f"q__{san}"])
        assert masked.loc[0, f"avail__{san}"] == 0.0
        assert masked.loc[0, f"exact__{san}"] == 0.0
    # ... but context experts and fink untouched on row 0
    for san in ("lasair__sherlock", "babamul", "fink__snn"):
        assert masked.loc[0, f"proj__{san}__p_snia"] == 0.7
        assert masked.loc[0, f"avail__{san}"] == 1.0

    # row 1 (context label): sherlock + babamul masked, alerce/fink untouched
    for san in ("lasair__sherlock", "babamul"):
        assert np.isnan(masked.loc[1, f"proj__{san}__p_snia"])
        assert np.isnan(masked.loc[1, f"q__{san}"])
        assert masked.loc[1, f"avail__{san}"] == 0.0
    for san in ("alerce__stamp_classifier", "alerce_lc", "fink__snn"):
        assert masked.loc[1, f"proj__{san}__p_snia"] == 0.7

    # row 2 (spectroscopic TNS): completely untouched
    for san in _SANS:
        assert masked.loc[2, f"proj__{san}__p_snia"] == 0.7
        assert masked.loc[2, f"avail__{san}"] == 1.0
        assert masked.loc[2, f"q__{san}"] == 0.6

    # labels never modified; input frame not mutated
    assert (masked["target_class"] == df["target_class"]).all()
    assert df.loc[0, "proj__alerce__stamp_classifier__p_snia"] == 0.7
    stats = masked.attrs["provenance_masking"]
    assert stats["n_alerce_masked"] == 1
    assert stats["n_context_masked"] == 1


# ---------------------------------------------------------------------------
# (g) object-weight normalization before class weighting
# ---------------------------------------------------------------------------

def test_object_weight_normalization():
    df = _make_snapshots(n_objects=40, seed=3)  # all spectroscopic, no BTS
    weights = compute_base_weights(
        df,
        weak_weight=0.1,
        context_weight=0.15,
        tns_untyped_weight=0.6,
        bts_weight=1.0,
    )
    sums = (
        pd.Series(weights, index=df.index)
        .groupby(df["object_id"].to_numpy())
        .sum()
    )
    # each object contributes total weight exactly 1.0 (= w_qual for spec)
    assert np.allclose(sums.to_numpy(), 1.0)

    # quality tiers scale the per-object total, not the shape
    df2 = df.copy()
    first_obj = df2["object_id"].iloc[0]
    df2.loc[df2["object_id"] == first_obj, "label_quality"] = "weak"
    weights2 = compute_base_weights(
        df2, weak_weight=0.1, context_weight=0.15,
        tns_untyped_weight=0.6, bts_weight=1.0,
    )
    sums2 = (
        pd.Series(weights2, index=df2.index)
        .groupby(df2["object_id"].to_numpy())
        .sum()
    )
    assert sums2[first_obj] == pytest.approx(0.1)
