"""fusion v13 Stage-A (pooled trust) corrections — docs/fusion_v13_plan.md.

1. The ALeRCE stamp family + BHRF top-level are SN filters (projection has no
   Ia/non-Ia distinction: p_snia == 0 exactly), so their trust target is
   ``is_sn``; the legacy 4-expert set is frozen for old artifacts.
2. The SN-filter set used at training is persisted in the pooled metadata and
   reproduced at inference (``is_sn_filter`` is a Stage-A feature); artifacts
   without the key fall back to the legacy set.
3. ``q_prior__`` on train rows is out-of-fold: it differs from the in-sample
   refit readout on train rows and is identical on cal/test rows; ``q__`` is
   unchanged by the refactor.
4. No ``q__``/``trust_source__`` for headless experts (Babamul, Lasair
   Sherlock) — the training emission matches ``attach_trust_columns``.
5. ``weak_policy`` gates weak rows per expert and is recorded in the metrics.
"""
from __future__ import annotations

import inspect
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import debass_meta.models.expert_trust as expert_trust  # noqa: E402
import debass_meta.models.pooled_trust as pooled_trust  # noqa: E402
from debass_meta.models.expert_trust import (  # noqa: E402
    LEGACY_SN_FILTER_EXPERTS,
    SN_FILTER_EXPERTS,
    is_alerce_family,
    trust_target_col,
)
from debass_meta.models.pooled_trust import (  # noqa: E402
    PooledTrustView,
    _assemble_features_for_expert,
    assemble_stage_a_long,
    train_pooled_trust,
    weak_rows_allowed,
)
from debass_meta.projectors import sanitize_expert_key  # noqa: E402
from debass_meta.projectors.alerce import project_events  # noqa: E402

V12_TRUST_DIR = REPO_ROOT / "models" / "trust_fusion_v12"

STAMP_FAMILY = (
    "alerce/stamp_classifier",
    "alerce/stamp_classifier_2025_beta",
    "alerce/stamp_classifier_rubin_beta",
)
NEW_SN_FILTERS = STAMP_FAMILY + ("alerce/lc_classifier_BHRF_forced_phot_top",)

TERNARY_EXPERTS = ("fink/snn", "supernnova")
SN_FILTER_SYNTH = ("fink/slsn", "alerce/stamp_classifier_rubin_beta")
HEADLESS_EXPERTS = ("babamul", "lasair/sherlock")
CLASSES = ["snia", "nonIa_snlike", "other"]
EPOCHS = (3, 5, 10)
N_OBJECTS = 240


def _entropy(probs: list[float]) -> float:
    return float(-sum(p * math.log(p) for p in probs if p > 0))


def _make_synthetic(seed: int = 7):
    """Wide snapshot + long helpfulness frames mirroring build_expert_helpfulness.

    - two ternary experts (target is_topclass_correct), one legacy SN filter
      (fink/slsn) and one ALeRCE stamp (p_snia == 0 by projection);
    - two headless context-only experts (Babamul, Sherlock) whose
      is_topclass_correct is None on every row — exactly the real table;
    - every 5th object (idx % 5 == 0) carries a WEAK LSST label (SN ->
      nonIa_snlike, never snia) and the next one (idx % 5 == 1) a WEAK ZTF
      label (which, as in the real ZTF gold, may be snia); label_source
      labels_csv so the anti-circularity filter does not already remove them
      for the ALeRCE family.
    """
    rng = np.random.default_rng(seed)
    snapshot_rows: list[dict] = []
    help_rows: list[dict] = []
    for idx in range(N_OBJECTS):
        object_id = f"OBJ{idx:04d}"
        if idx % 5 == 0:
            label = str(rng.choice(["nonIa_snlike", "other"], p=[0.6, 0.4]))
            label_quality, label_source, is_lsst = "weak", "labels_csv", 1.0
        elif idx % 5 == 1:
            label = str(rng.choice(CLASSES, p=[0.4, 0.3, 0.3]))
            label_quality, label_source, is_lsst = "weak", "labels_csv", 0.0
        else:
            label = str(rng.choice(CLASSES, p=[0.4, 0.3, 0.3]))
            label_quality, label_source, is_lsst = "spectroscopic", "tns_spectroscopic", 0.0
        is_sn = int(label != "other")
        for n_det in EPOCHS:
            base = {
                "object_id": object_id,
                "n_det": int(n_det),
                "alert_jd": 2460000.0 + idx * 0.41 + float(n_det),
                "target_class": label,
                "target_follow_proxy": float(label == "snia"),
                "label_source": label_source,
                "label_quality": label_quality,
                "t_since_first": float(n_det) * 2.0 + rng.uniform(0.0, 1.0),
                "mag_last": 19.0 + rng.normal(0.0, 0.6),
                "mag_range": (2.0 if is_sn else 0.5) + rng.normal(0.0, 0.2),
                "lc_noise": rng.normal(),
                "survey_is_lsst": is_lsst,
            }
            snap = dict(base)
            for expert_key in TERNARY_EXPERTS + SN_FILTER_SYNTH:
                san = sanitize_expert_key(expert_key)
                snap[f"avail__{san}"] = 1.0
                if expert_key in SN_FILTER_SYNTH:
                    score = float(np.clip(0.15 + 0.7 * is_sn + rng.uniform(0, 0.15), 0.01, 0.99))
                    p_vec = {"snia": 0.0, "nonIa_snlike": score, "other": 1.0 - score}
                else:
                    margin_draw = rng.uniform(0.0, 1.0)
                    correct = rng.random() < (0.1 + 0.8 * margin_draw)
                    pred = label if correct else str(rng.choice([c for c in CLASSES if c != label]))
                    p_top = (1.0 + 2.0 * margin_draw) / 3.0
                    rest = 1.0 - p_top
                    others = [c for c in CLASSES if c != pred]
                    p_vec = {pred: p_top, others[0]: rest * 0.6, others[1]: rest * 0.4}
                ranked = sorted(p_vec.items(), key=lambda kv: kv[1], reverse=True)
                mapped_pred = ranked[0][0]
                proj = {
                    f"proj__{san}__p_snia": p_vec["snia"],
                    f"proj__{san}__p_nonIa_snlike": p_vec["nonIa_snlike"],
                    f"proj__{san}__p_other": p_vec["other"],
                    f"proj__{san}__top1_prob": ranked[0][1],
                    f"proj__{san}__margin": ranked[0][1] - ranked[1][1],
                    f"proj__{san}__entropy": _entropy(list(p_vec.values())),
                }
                snap.update(proj)
                snap[f"exact__{san}"] = 1.0
                snap[f"temporal_exactness__{san}"] = "exact_alert"
                snap[f"mapped_pred_class__{san}"] = mapped_pred
                snap[f"prediction_type__{san}"] = "class_correctness"
                help_rows.append({
                    **base, **proj,
                    "expert_key": expert_key, "available": True,
                    "temporal_exactness": "exact_alert",
                    "mapped_pred_class": mapped_pred,
                    "prediction_type": "class_correctness",
                    f"avail__{san}": 1.0, f"exact__{san}": 1.0,
                    "is_topclass_correct": int(mapped_pred == label),
                    "is_sn": is_sn,
                })
            for expert_key in HEADLESS_EXPERTS:
                san = sanitize_expert_key(expert_key)
                snap[f"avail__{san}"] = 1.0
                snap[f"exact__{san}"] = 1.0
                snap[f"temporal_exactness__{san}"] = "static_safe"
                snap[f"prediction_type__{san}"] = "context_only"
                help_rows.append({
                    **base,
                    "expert_key": expert_key, "available": True,
                    "temporal_exactness": "static_safe",
                    "mapped_pred_class": None,
                    "prediction_type": "context_only",
                    f"avail__{san}": 1.0, f"exact__{san}": 1.0,
                    "is_topclass_correct": None,
                    "is_sn": is_sn,
                })
            snapshot_rows.append(snap)

    snapshots = pd.DataFrame(snapshot_rows)
    helpfulness = pd.DataFrame(help_rows)
    train_ids = {f"OBJ{i:04d}" for i in range(0, 144)}
    cal_ids = {f"OBJ{i:04d}" for i in range(144, 192)}
    test_ids = {f"OBJ{i:04d}" for i in range(192, 240)}
    return snapshots, helpfulness, train_ids, cal_ids, test_ids


def _train(synthetic, out_dir: Path, **kwargs):
    snapshots, helpfulness, train_ids, cal_ids, test_ids = synthetic
    return train_pooled_trust(
        helpfulness, snapshots, train_ids, cal_ids, test_ids, str(out_dir),
        n_jobs=2, seed=42, grid_small=True, **kwargs,
    )


@pytest.fixture(scope="module")
def synthetic():
    return _make_synthetic()


@pytest.fixture(scope="module")
def trained_default(synthetic, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("trust_v13")
    return out_dir, _train(synthetic, out_dir)


@pytest.fixture(scope="module")
def trained_legacy_prior(synthetic, tmp_path_factory):
    """Same seed/data, q_prior refit readout on train rows (legacy)."""
    out_dir = tmp_path_factory.mktemp("trust_v13_legacy_prior")
    return out_dir, _train(synthetic, out_dir, q_prior_oof=False)


# ---------------------------------------------------------------------------
# 1. stamp family / BHRF top are SN filters
# ---------------------------------------------------------------------------


def test_stamp_family_target_is_is_sn() -> None:
    for key in NEW_SN_FILTERS:
        assert key in SN_FILTER_EXPERTS
        assert trust_target_col(key) == "is_sn"
        assert key not in LEGACY_SN_FILTER_EXPERTS
    # legacy members untouched; frozen legacy set is exactly the v8-v12 four
    assert LEGACY_SN_FILTER_EXPERTS == frozenset(
        {"fink_lsst/snn", "fink_lsst/cats", "fink/slsn", "ampel/snguess"})
    assert LEGACY_SN_FILTER_EXPERTS <= SN_FILTER_EXPERTS
    assert isinstance(LEGACY_SN_FILTER_EXPERTS, frozenset)
    # ternary experts keep the top-class target
    for key in ("fink/snn", "fink/rf_ia", "alerce_lc", "supernnova", "seq_v11",
                "alerce/lc_classifier_transient",
                "alerce/lc_classifier_BHRF_forced_phot_transient"):
        assert trust_target_col(key) == "is_topclass_correct"
    # explicit override reproduces an older set
    assert trust_target_col("alerce/stamp_classifier", LEGACY_SN_FILTER_EXPERTS) == "is_topclass_correct"
    assert trust_target_col("fink/slsn", LEGACY_SN_FILTER_EXPERTS) == "is_sn"


def test_stamp_projection_has_no_ia_mass() -> None:
    """The projector routes ALL stamp 'SN' mass to nonIa_snlike: a correct SN
    call on a Type Ia can never be top-class-correct, which is why is_sn is
    the only honest target for these experts."""
    stamp_events = [
        {"classifier": "stamp_classifier", "class_name": "SN", "canonical_projection": 0.9},
        {"classifier": "stamp_classifier", "class_name": "AGN", "canonical_projection": 0.1},
    ]
    for key in STAMP_FAMILY:
        out = project_events(key, stamp_events)
        assert out["prediction_type"] == "class_correctness"
        assert out["p_snia"] == 0.0
        assert out["mapped_pred_class"] == "nonIa_snlike"
    top_events = [
        {"class_name": "Transient", "canonical_projection": 0.8},
        {"class_name": "Periodic", "canonical_projection": 0.2},
    ]
    out = project_events("alerce/lc_classifier_BHRF_forced_phot_top", top_events)
    assert out["p_snia"] == 0.0 and out["mapped_pred_class"] == "nonIa_snlike"
    # Negative control: a classifier WITH an SNIa class keeps p_snia > 0 and
    # is (correctly) NOT an SN filter.
    lc_events = [
        {"class_name": "SNIa", "canonical_projection": 0.7},
        {"class_name": "SNII", "canonical_projection": 0.3},
    ]
    out = project_events("alerce/lc_classifier_transient", lc_events)
    assert out["p_snia"] > 0.5
    assert "alerce/lc_classifier_transient" not in SN_FILTER_EXPERTS
    assert is_alerce_family("alerce/stamp_classifier") and is_alerce_family("alerce_lc")
    assert not is_alerce_family("fink/snn")


def test_assembly_uses_is_sn_for_stamp(synthetic) -> None:
    _, helpfulness, *_ = synthetic
    long_df, _, expert_frames = assemble_stage_a_long(helpfulness, apply_honesty_filters=True)
    stamp = "alerce/stamp_classifier_rubin_beta"
    assert expert_frames[stamp][1] == "is_sn"
    assert expert_frames["fink/snn"][1] == "is_topclass_correct"
    rows = long_df[long_df["expert_key"] == stamp]
    assert set(rows["is_sn_filter"].unique()) == {1.0}
    assert set(long_df.loc[long_df["expert_key"] == "fink/snn", "is_sn_filter"].unique()) == {0.0}
    # under the OLD target every correct 'SN' call on an Ia was graded wrong
    sub = helpfulness[helpfulness["expert_key"] == stamp]
    ia = sub[sub["target_class"] == "snia"]
    assert len(ia) > 0 and ia["is_topclass_correct"].max() == 0
    assert ia["is_sn"].min() == 1
    # headless context-only experts never get a head (no usable target)
    for key in HEADLESS_EXPERTS:
        assert key not in expert_frames


# ---------------------------------------------------------------------------
# 2. persisted SN-filter set at inference (+ legacy fallback)
# ---------------------------------------------------------------------------


def test_sn_filter_set_persisted_and_used_at_inference(
    trained_default, synthetic, tmp_path, monkeypatch
) -> None:
    out_dir, result = trained_default
    snapshots = synthetic[0]
    meta = json.loads((out_dir / "pooled" / "metadata.json").read_text())
    assert meta["sn_filter_experts"] == sorted(SN_FILTER_EXPERTS)
    assert result.metrics["_pooled"]["sn_filter_experts"] == sorted(SN_FILTER_EXPERTS)
    assert result.metrics["alerce/stamp_classifier_rubin_beta"]["target_col"] == "is_sn"
    assert result.metrics["alerce/stamp_classifier_rubin_beta"]["is_sn_filter"] is True

    stamp_dir = out_dir / sanitize_expert_key("alerce/stamp_classifier_rubin_beta")
    view = PooledTrustView.load(str(stamp_dir))
    assert view.sn_filter_experts == frozenset(SN_FILTER_EXPERTS)
    p_ref = view.predict_trust(snapshots)

    # Inference must NOT depend on the module-level set: shrink it to the
    # legacy four and re-score -> identical.
    monkeypatch.setattr(expert_trust, "SN_FILTER_EXPERTS", set(LEGACY_SN_FILTER_EXPERTS))
    monkeypatch.setattr(pooled_trust, "SN_FILTER_EXPERTS", set(LEGACY_SN_FILTER_EXPERTS))
    view2 = PooledTrustView.load(str(stamp_dir))
    assert view2.sn_filter_experts == frozenset(SN_FILTER_EXPERTS)
    np.testing.assert_array_equal(view2.predict_trust(snapshots), p_ref)

    # Legacy artifact (metadata lacks the key): fall back to the frozen
    # legacy set -> is_sn_filter feature is 0 for the stamp expert.
    legacy_dir = tmp_path / "legacy_artifact"
    shutil.copytree(out_dir, legacy_dir)
    legacy_meta = json.loads((legacy_dir / "pooled" / "metadata.json").read_text())
    del legacy_meta["sn_filter_experts"]
    (legacy_dir / "pooled" / "metadata.json").write_text(json.dumps(legacy_meta))
    legacy_view = PooledTrustView.load(str(legacy_dir / stamp_dir.name))
    assert legacy_view.sn_filter_experts == LEGACY_SN_FILTER_EXPERTS
    feats_legacy = _assemble_features_for_expert(
        snapshots, "alerce/stamp_classifier_rubin_beta", generic_cols=[],
        sn_filter_experts=legacy_view.sn_filter_experts)
    feats_new = _assemble_features_for_expert(
        snapshots, "alerce/stamp_classifier_rubin_beta", generic_cols=[],
        sn_filter_experts=view.sn_filter_experts)
    assert set(feats_legacy["is_sn_filter"].unique()) == {0.0}
    assert set(feats_new["is_sn_filter"].unique()) == {1.0}
    # and the legacy members are still flagged under the legacy set
    feats_slsn = _assemble_features_for_expert(
        snapshots, "fink/slsn", generic_cols=[], sn_filter_experts=legacy_view.sn_filter_experts)
    assert set(feats_slsn["is_sn_filter"].unique()) == {1.0}


def test_explicit_sn_filter_override_is_persisted(synthetic, tmp_path) -> None:
    """A run may pin its own set; what is persisted is what was used."""
    out_dir = tmp_path / "trust_override"
    result = _train(synthetic, out_dir, sn_filter_experts=LEGACY_SN_FILTER_EXPERTS)
    meta = json.loads((out_dir / "pooled" / "metadata.json").read_text())
    assert meta["sn_filter_experts"] == sorted(LEGACY_SN_FILTER_EXPERTS)
    assert result.metrics["alerce/stamp_classifier_rubin_beta"]["target_col"] == "is_topclass_correct"
    view = PooledTrustView.load(str(out_dir / "alerce__stamp_classifier_rubin_beta"))
    assert view.sn_filter_experts == LEGACY_SN_FILTER_EXPERTS


@pytest.mark.skipif(not (V12_TRUST_DIR / "pooled" / "metadata.json").exists(),
                    reason="models/trust_fusion_v12 not present")
def test_v12_artifact_loads_with_legacy_set() -> None:
    meta = json.loads((V12_TRUST_DIR / "pooled" / "metadata.json").read_text())
    assert "sn_filter_experts" not in meta  # pre-v13 artifact
    view = PooledTrustView.load(str(V12_TRUST_DIR / "alerce__stamp_classifier_rubin_beta"))
    assert view.sn_filter_experts == LEGACY_SN_FILTER_EXPERTS
    # documents the v12 defect: the Rubin stamp head was trained on is_topclass_correct
    stamp_meta = json.loads(
        (V12_TRUST_DIR / "alerce__stamp_classifier_rubin_beta" / "metadata.json").read_text())
    assert stamp_meta["target_col"] == "is_topclass_correct"


# ---------------------------------------------------------------------------
# 3. q_prior on train rows is OOF
# ---------------------------------------------------------------------------


def test_q_prior_train_rows_oof_differs_from_refit(
    trained_default, trained_legacy_prior, synthetic
) -> None:
    _, res_oof = trained_default
    _, res_ref = trained_legacy_prior
    _, _, train_ids, cal_ids, test_ids = synthetic
    out_oof, out_ref = res_oof.snapshots, res_ref.snapshots
    assert len(out_oof) == len(out_ref)
    oid = out_oof["object_id"].astype(str)
    is_train = oid.isin(train_ids).to_numpy()
    is_heldout = oid.isin(cal_ids | test_ids).to_numpy()
    assert is_train.sum() > 100 and is_heldout.sum() > 100

    emission = res_oof.metrics["_pooled"]["emission"]
    assert res_oof.metrics["_pooled"]["q_prior_oof"] is True
    assert res_oof.metrics["_pooled"]["q_prior_n_folds"] == 5
    assert emission["n_train_rows"] == int(is_train.sum())
    # every train object contributed labelled rows here -> all train rows OOF
    assert emission["n_train_rows_prior_oof"] == int(is_train.sum())
    assert res_ref.metrics["_pooled"]["q_prior_oof"] is False
    assert res_ref.metrics["_pooled"]["emission"]["n_train_rows_prior_oof"] == 0

    # trained expert AND a never-trained registered expert (prior emitted for
    # all): OOF differs from the in-sample refit on train rows, identical on
    # cal/test rows.
    # (the never-trained expert's prior passes through the GLOBAL isotonic
    # calibrator, whose plateaus absorb small raw differences -> lower bar)
    for expert_key, min_frac in (("fink/snn", 0.5),
                                 ("alerce/stamp_classifier_rubin_beta", 0.5),
                                 ("parsnip", 0.2)):
        col = f"q_prior__{sanitize_expert_key(expert_key)}"
        a = out_oof[col].to_numpy(float)
        b = out_ref[col].to_numpy(float)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        n_diff = int((np.abs(a[is_train] - b[is_train]) > 1e-9).sum())
        assert n_diff >= min_frac * is_train.sum(), (expert_key, n_diff)
        np.testing.assert_allclose(a[is_heldout], b[is_heldout], atol=1e-12)

    # q__ (already OOF on train rows) is untouched by the fold-model reuse.
    for expert_key in TERNARY_EXPERTS + SN_FILTER_SYNTH:
        col = f"q__{sanitize_expert_key(expert_key)}"
        np.testing.assert_allclose(
            out_oof[col].to_numpy(float), out_ref[col].to_numpy(float),
            atol=1e-12, equal_nan=True)
        src = f"trust_source__{sanitize_expert_key(expert_key)}"
        assert (out_oof.loc[is_train, src] == "oof").all()


# ---------------------------------------------------------------------------
# 4. no q__ for headless experts (training emission == scoring emission)
# ---------------------------------------------------------------------------


def test_no_q_for_headless_experts(trained_default, synthetic) -> None:
    out_dir, result = trained_default
    out = result.snapshots
    for key in HEADLESS_EXPERTS:
        san = sanitize_expert_key(key)
        assert f"avail__{san}" in out.columns          # the expert IS present...
        assert f"q__{san}" not in out.columns          # ...but has no head
        assert f"trust_source__{san}" not in out.columns
        assert f"q_prior__{san}" in out.columns        # prior still for all registered
        assert key not in result.metrics
        assert not (out_dir / san).exists()
    emitted_q = {c[len("q__"):] for c in out.columns if c.startswith("q__")}
    trained = {sanitize_expert_key(k) for k in result.metrics if not k.startswith("_")}
    assert emitted_q == trained
    assert set(result.metrics["_pooled"]["emission"]["q_experts"]) == set(
        k for k in result.metrics if not k.startswith("_"))
    assert set(result.metrics["_pooled"]["emission"]["headless_experts_with_avail"]) == set(HEADLESS_EXPERTS)

    # Exactly the columns the scorer produces from the trust dir.
    from scripts.score_fusion_v8 import _expert_dirs, attach_trust_columns

    scored = attach_trust_columns(synthetic[0].copy(), out_dir)
    scored_q = {c[len("q__"):] for c in scored.columns if c.startswith("q__")}
    assert scored_q == emitted_q == {d.name for d in _expert_dirs(out_dir)}


def test_legacy_headless_emission_and_trained_only_prior(synthetic, tmp_path) -> None:
    out_dir = tmp_path / "trust_legacy_headless"
    result = _train(synthetic, out_dir, emit_headless_q=True, q_prior_experts="trained")
    out = result.snapshots
    assert "q__babamul" in out.columns and "trust_source__babamul" in out.columns
    assert "q_prior__babamul" not in out.columns
    assert "q_prior__parsnip" not in out.columns
    assert "q_prior__fink__snn" in out.columns
    with pytest.raises(ValueError, match="q_prior_experts"):
        _train(synthetic, tmp_path / "bad", q_prior_experts="nope")


# ---------------------------------------------------------------------------
# 5. weak_policy
# ---------------------------------------------------------------------------


def test_weak_rows_allowed_table() -> None:
    assert weak_rows_allowed("all", "fink/snn", "is_topclass_correct") is True
    assert weak_rows_allowed("all", "alerce/stamp_classifier_rubin_beta", "is_sn") is True
    assert weak_rows_allowed("none", "fink_lsst/snn", "is_sn") is False
    assert weak_rows_allowed("none", "fink/snn", "is_topclass_correct", is_lsst=False) is False
    assert weak_rows_allowed("is_sn_only", "fink_lsst/snn", "is_sn") is True
    assert weak_rows_allowed("is_sn_only", "babamul", "is_sn") is True
    assert weak_rows_allowed("is_sn_only", "fink/snn", "is_topclass_correct") is False
    assert weak_rows_allowed("is_sn_only", "fink/snn", "is_topclass_correct", is_lsst=False) is False
    assert weak_rows_allowed("is_sn_only", "seq_v11", "is_topclass_correct") is False
    # ALeRCE family never, even with an is_sn target
    assert weak_rows_allowed("is_sn_only", "alerce/stamp_classifier_rubin_beta", "is_sn") is False
    assert weak_rows_allowed("is_sn_only", "alerce_lc", "is_topclass_correct") is False
    # lsst_is_sn_only: LSST rows follow is_sn_only, ZTF rows follow "all"
    assert weak_rows_allowed("lsst_is_sn_only", "fink_lsst/snn", "is_sn", is_lsst=True) is True
    assert weak_rows_allowed("lsst_is_sn_only", "fink/snn", "is_topclass_correct", is_lsst=True) is False
    assert weak_rows_allowed("lsst_is_sn_only", "fink/snn", "is_topclass_correct", is_lsst=False) is True
    assert weak_rows_allowed("lsst_is_sn_only", "alerce/stamp_classifier_rubin_beta", "is_sn", is_lsst=True) is False
    assert weak_rows_allowed("lsst_is_sn_only", "alerce/stamp_classifier", "is_sn", is_lsst=False) is True
    assert weak_rows_allowed("lsst_is_sn_only", "alerce_lc", "is_topclass_correct", is_lsst=False) is True
    with pytest.raises(ValueError, match="weak_policy"):
        weak_rows_allowed("sometimes", "fink/snn", "is_sn")


def test_row_is_lsst_fallbacks() -> None:
    from debass_meta.models.pooled_trust import _row_is_lsst

    frame = pd.DataFrame({
        "object_id": ["ZTF25abc", "170028485647532096", "ZTF25abd", "170028485647532097", "ZTF25abe"],
        "survey_is_lsst": [0.0, 1.0, np.nan, np.nan, np.nan],
        "survey": ["ZTF", "LSST", "lsst", None, "ZTF"],
    })
    # column -> survey string -> id shape
    np.testing.assert_array_equal(_row_is_lsst(frame), [False, True, True, True, False])
    np.testing.assert_array_equal(
        _row_is_lsst(frame[["object_id"]]), [False, True, False, True, False])
    np.testing.assert_array_equal(
        _row_is_lsst(frame[["object_id", "survey"]]), [False, True, True, True, False])


def _weak_counts(helpfulness: pd.DataFrame):
    weak = helpfulness[helpfulness["label_quality"] == "weak"]
    lsst = weak["survey_is_lsst"] > 0.5
    return (weak.groupby("expert_key").size(),
            weak[lsst].groupby("expert_key").size(),
            weak[~lsst].groupby("expert_key").size())


def test_weak_policy_filters_rows_and_records_ledger(synthetic) -> None:
    _, helpfulness, *_ = synthetic
    weak_all, weak_lsst, weak_ztf = _weak_counts(helpfulness)
    stamp, slsn, snn = "alerce/stamp_classifier_rubin_beta", "fink/slsn", "fink/snn"
    assert weak_lsst[stamp] > 0 and weak_ztf[stamp] > 0
    n_strong = (helpfulness["label_quality"] != "weak").groupby(helpfulness["expert_key"]).sum()

    by_policy = {}
    for policy in ("all", "is_sn_only", "lsst_is_sn_only", "none"):
        long_df, _, frames = assemble_stage_a_long(helpfulness, weak_policy=policy)
        by_policy[policy] = (long_df, frames, long_df.attrs["weak_ledger"])
        assert long_df.attrs["weak_policy"] == policy
        # ledger arithmetic + frame/long consistency for every expert
        for key, entry in long_df.attrs["weak_ledger"].items():
            assert entry["kept"] + entry["dropped"] == weak_all[key]
            assert entry["kept_lsst"] + entry["dropped_lsst"] == weak_lsst[key]
            assert entry["allowed"] == (entry["allowed_lsst"] and entry["allowed_ztf"])
            sub = frames[key][0]
            assert (sub["label_quality"] == "weak").sum() == entry["kept"]
            rows = long_df[long_df["expert_key"] == key]
            assert (rows["label_quality"] == "weak").sum() == entry["kept"]
            assert len(rows) == n_strong[key] + entry["kept"]     # non-weak rows untouched

    _, _, ledger_all = by_policy["all"]
    for key in (stamp, slsn, snn, "supernnova"):
        assert ledger_all[key]["kept"] == weak_all[key] and ledger_all[key]["dropped"] == 0
        assert ledger_all[key]["kept_lsst"] == weak_lsst[key]

    _, _, ledger_sn = by_policy["is_sn_only"]
    assert ledger_sn[slsn]["allowed"] is True and ledger_sn[slsn]["kept"] == weak_all[slsn]
    for key in (stamp, snn, "supernnova"):
        assert ledger_sn[key]["allowed"] is False
        assert ledger_sn[key]["dropped"] == weak_all[key]
        assert ledger_sn[key]["dropped_lsst"] == weak_lsst[key]

    # lsst_is_sn_only: ZTF weak rows always kept (legacy), LSST weak rows
    # only for is_sn-target experts outside the ALeRCE family.
    _, _, ledger_l = by_policy["lsst_is_sn_only"]
    assert ledger_l[slsn]["allowed_lsst"] is True and ledger_l[slsn]["kept"] == weak_all[slsn]
    for key in (stamp, snn, "supernnova"):
        assert ledger_l[key]["allowed_lsst"] is False and ledger_l[key]["allowed_ztf"] is True
        assert ledger_l[key]["allowed"] is False
        assert ledger_l[key]["dropped"] == ledger_l[key]["dropped_lsst"] == weak_lsst[key]
        assert ledger_l[key]["kept"] == weak_ztf[key] and ledger_l[key]["kept_lsst"] == 0
    # ZTF rows of every expert are exactly the "all" rows
    long_all_df = by_policy["all"][0]
    long_l_df = by_policy["lsst_is_sn_only"][0]
    for key in (stamp, snn, slsn, "supernnova"):
        a = long_all_df[(long_all_df["expert_key"] == key) & (long_all_df["survey_is_lsst"] < 0.5)]
        b = long_l_df[(long_l_df["expert_key"] == key) & (long_l_df["survey_is_lsst"] < 0.5)]
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))

    long_none, _, ledger_none = by_policy["none"]
    assert not (long_none["label_quality"] == "weak").any()
    assert all(v["dropped"] == weak_all[k] for k, v in ledger_none.items())

    with pytest.raises(ValueError, match="weak_policy"):
        assemble_stage_a_long(helpfulness, weak_policy="maybe")


def test_train_pooled_trust_weak_policy_metrics(synthetic, tmp_path) -> None:
    assert "weak_policy" in inspect.signature(train_pooled_trust).parameters  # orchestrator contract
    assert "q_prior_experts" in inspect.signature(train_pooled_trust).parameters
    helpfulness = synthetic[1]
    weak_all, weak_lsst, weak_ztf = _weak_counts(helpfulness)
    stamp = "alerce/stamp_classifier_rubin_beta"

    out_dir = tmp_path / "trust_is_sn_only"
    m = _train(synthetic, out_dir, weak_policy="is_sn_only").metrics
    assert m["_pooled"]["weak_policy"] == "is_sn_only"
    assert m["fink/slsn"]["n_weak_rows_kept"] == weak_all["fink/slsn"]
    assert m["fink/slsn"]["n_weak_rows_dropped"] == 0
    assert m["fink/snn"]["n_weak_rows_dropped"] == weak_all["fink/snn"]
    assert m["fink/snn"]["n_weak_rows_kept"] == 0
    assert m[stamp]["n_weak_rows_dropped"] == weak_all[stamp]
    assert m[stamp]["n_weak_rows_dropped_lsst"] == weak_lsst[stamp]
    for key in ("fink/snn", "fink/slsn"):
        assert m[key]["weak_policy"] == "is_sn_only"
        assert "cal_set_raw_auc" in m[key]
    meta = json.loads((out_dir / "pooled" / "metadata.json").read_text())
    assert meta["weak_policy"] == "is_sn_only"
    assert meta["weak_ledger"]["fink/snn"]["dropped"] == weak_all["fink/snn"]
    assert meta["q_prior_oof"] is True and meta["emit_headless_q"] is False
    assert meta["prior_mode_version"] == 2 and meta["q_prior_experts"] == "all"

    out_dir_l = tmp_path / "trust_lsst_is_sn_only"
    m = _train(synthetic, out_dir_l, weak_policy="lsst_is_sn_only").metrics
    assert m["_pooled"]["weak_policy"] == "lsst_is_sn_only"
    assert m["fink/snn"]["n_weak_rows_dropped"] == weak_lsst["fink/snn"]
    assert m["fink/snn"]["n_weak_rows_kept"] == weak_ztf["fink/snn"]
    assert m["fink/snn"]["n_weak_rows_kept_lsst"] == 0
    assert m[stamp]["n_weak_rows_dropped_lsst"] == weak_lsst[stamp]
    assert m[stamp]["n_weak_rows_kept"] == weak_ztf[stamp]
    assert m["fink/slsn"]["n_weak_rows_kept"] == weak_all["fink/slsn"]
    # vs the legacy policy the ONLY rows missing are fink/snn's LSST weak train rows
    m_all = _train(synthetic, tmp_path / "trust_all", weak_policy="all").metrics
    train_ids = synthetic[2]
    h = helpfulness
    n_lsst_weak_train = int((
        (h["expert_key"] == "fink/snn") & (h["label_quality"] == "weak")
        & (h["survey_is_lsst"] > 0.5) & h["object_id"].astype(str).isin(train_ids)
    ).sum())
    assert n_lsst_weak_train > 0
    assert m["fink/snn"]["n_train_rows"] == m_all["fink/snn"]["n_train_rows"] - n_lsst_weak_train
    with pytest.raises(ValueError, match="weak_policy"):
        _train(synthetic, tmp_path / "bad", weak_policy="maybe")


# ---------------------------------------------------------------------------
# 6. train/serve parity: training snapshot == attach_trust_columns on cal/test
# ---------------------------------------------------------------------------


def _heldout_mask(frame: pd.DataFrame, synthetic) -> np.ndarray:
    _, _, _, cal_ids, test_ids = synthetic
    return frame["object_id"].astype(str).isin(cal_ids | test_ids).to_numpy()


def test_training_snapshot_matches_scoring_on_heldout_rows(trained_default, synthetic) -> None:
    """For CAL/TEST rows every q__* and q_prior__* value train_pooled_trust
    wrote must equal what attach_trust_columns produces from the saved
    artifact (train rows are OOF in the snapshot, so they differ by design)."""
    from scripts.score_fusion_v8 import attach_trust_columns

    out_dir, result = trained_default
    snapshots = synthetic[0]
    trained = result.snapshots
    scored = attach_trust_columns(snapshots.copy(), out_dir)
    assert (scored["object_id"].to_numpy() == trained["object_id"].to_numpy()).all()
    held = _heldout_mask(trained, synthetic)
    assert held.sum() > 200

    q_cols_train = sorted(c for c in trained.columns if c.startswith("q__"))
    q_cols_score = sorted(c for c in scored.columns if c.startswith("q__"))
    qp_cols_train = sorted(c for c in trained.columns if c.startswith("q_prior__"))
    qp_cols_score = sorted(c for c in scored.columns if c.startswith("q_prior__"))
    assert q_cols_train == q_cols_score
    assert qp_cols_train == qp_cols_score and len(qp_cols_train) == 30  # q_prior_experts="all"
    assert "q_prior__babamul" in qp_cols_score  # headless expert served via the pooled model
    for col in q_cols_train + qp_cols_train:
        a = trained[col].to_numpy(float)[held]
        b = scored[col].to_numpy(float)[held]
        np.testing.assert_allclose(a, b, atol=1e-9, rtol=0, equal_nan=True, err_msg=col)
        assert np.isfinite(b).any()
    # trust_source__ legitimately differs ('train_model' vs 'score_time').
    for col in q_cols_train:
        src = f"trust_source__{col[len('q__'):]}"
        assert set(trained.loc[held, src]) <= {"train_model", "unavailable"}
        assert set(scored.loc[held, src]) <= {"score_time", "unavailable"}
    # train rows: q_prior differs (OOF in the snapshot, refit at scoring)
    col = "q_prior__fink__snn"
    assert (np.abs(trained[col].to_numpy(float)[~held] - scored[col].to_numpy(float)[~held]) > 1e-9).mean() > 0.5


def test_scoring_honours_persisted_q_prior_experts(synthetic, tmp_path) -> None:
    from scripts.score_fusion_v8 import attach_trust_columns

    out_dir = tmp_path / "trust_trained_only"
    result = _train(synthetic, out_dir, q_prior_experts="trained")
    scored = attach_trust_columns(synthetic[0].copy(), out_dir)
    qp_train = sorted(c for c in result.snapshots.columns if c.startswith("q_prior__"))
    qp_score = sorted(c for c in scored.columns if c.startswith("q_prior__"))
    assert qp_train == qp_score
    assert "q_prior__babamul" not in qp_score and "q_prior__parsnip" not in qp_score
    held = _heldout_mask(scored, synthetic)
    for col in qp_train:
        np.testing.assert_allclose(
            result.snapshots[col].to_numpy(float)[held], scored[col].to_numpy(float)[held],
            atol=1e-9, rtol=0, equal_nan=True, err_msg=col)


def test_legacy_artifact_uses_legacy_prior_path(trained_default, synthetic, tmp_path) -> None:
    """Strip the v13 keys from a copied artifact: the scorer must take the
    v8-v12 path (proj__ masked through predict_trust, trust dirs only)."""
    from scripts.score_fusion_v8 import attach_trust_columns

    out_dir, _ = trained_default
    legacy_dir = tmp_path / "legacy"
    shutil.copytree(out_dir, legacy_dir)
    meta_path = legacy_dir / "pooled" / "metadata.json"
    meta = json.loads(meta_path.read_text())
    for key in ("prior_mode_version", "q_prior_experts", "sn_filter_experts"):
        meta.pop(key, None)
    meta_path.write_text(json.dumps(meta))

    snapshots = synthetic[0]
    scored = attach_trust_columns(snapshots.copy(), legacy_dir)
    qp_cols = sorted(c for c in scored.columns if c.startswith("q_prior__"))
    assert qp_cols == sorted(f"q_prior__{sanitize_expert_key(k)}" for k in TERNARY_EXPERTS + SN_FILTER_SYNTH)
    san = sanitize_expert_key("fink/snn")
    view = PooledTrustView.load(str(legacy_dir / san))
    assert view.prior_mode_version == 1 and view.sn_filter_experts == LEGACY_SN_FILTER_EXPERTS
    masked = snapshots.copy()
    for c in [c for c in masked.columns if c.startswith(f"proj__{san}__")]:
        masked[c] = np.nan
    np.testing.assert_array_equal(scored[f"q_prior__{san}"].to_numpy(float), view.predict_trust(masked))
    # (on this traj-less synthetic the two readouts coincide numerically; the
    # protocols differ on gold with traj__ columns / unmapped exactness codes)
    v2_view = PooledTrustView.load(str(out_dir / san))
    assert v2_view.prior_mode_version == 2 and v2_view.has_head
    assert np.isfinite(v2_view.predict_prior(snapshots)).all()
