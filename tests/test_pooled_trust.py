"""Unit tests for the fusion_v8 pooled Stage-A trust model.

Synthetic multi-expert helpfulness + snapshot frames (3 experts x 300
objects x 3 epochs) engineered so that:

- trust-target parity is verifiable (SN-filter expert -> is_sn target);
- the three calibration tiers fire by construction of cal-row counts
  (isotonic >= 200, platt 30..199, global < 30);
- q is NaN exactly where avail == 0;
- OOF train-row emission differs from refit-model (view) predictions;
- q_prior differs from q and is defined where the expert is absent;
- split-integrity asserts fire on overlapping IDs;
- the spec-correction-#5 dedicated-head fallback can be forced via
  monkeypatched module thresholds.

Plus the spec-correction-#4e parity test against the real
data/gold/expert_helpfulness.parquet base rates (skipped if absent).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import debass_meta.models.pooled_trust as pooled_trust
from debass_meta.models.pooled_trust import (
    PooledTrustResult,
    PooledTrustView,
    _should_fallback,
    assemble_stage_a_long,
    train_pooled_trust,
)
from debass_meta.projectors import sanitize_expert_key

REAL_HELPFULNESS = REPO_ROOT / "data" / "gold" / "expert_helpfulness.parquet"

EXPERTS = ["fink/snn", "fink/slsn", "supernnova"]
EPOCHS = (3, 5, 10)
N_OBJECTS = 300
CLASSES = ["snia", "nonIa_snlike", "other"]


def _avail(expert_key: str, idx: int) -> bool:
    if expert_key == "fink/snn":
        return True  # full coverage -> >=200 cal rows -> isotonic tier
    if expert_key == "fink/slsn":
        return idx % 2 == 0  # ~150 cal rows -> platt tier
    return idx % 15 == 0  # ~21 cal rows -> global tier


def _entropy(probs: list[float]) -> float:
    return float(-sum(p * math.log(p) for p in probs if p > 0))


def _make_synthetic(seed: int = 42):
    """Snapshot (wide) + helpfulness (long) frames mirroring the real
    build_expert_helpfulness.py schema, with learnable trust signal:
    correctness ~ Bernoulli(0.1 + 0.8 * own_margin) for ternary experts and
    an LC feature (mag_range) correlated with is_sn for the SN filter."""
    rng = np.random.default_rng(seed)
    target_class = rng.choice(CLASSES, size=N_OBJECTS, p=[0.4, 0.3, 0.3])

    snapshot_rows: list[dict] = []
    helpfulness_rows: list[dict] = []
    for idx in range(N_OBJECTS):
        object_id = f"OBJ{idx:04d}"
        label = str(target_class[idx])
        is_sn = int(label != "other")
        for n_det in EPOCHS:
            alert_jd = 2460000.0 + idx * 0.37 + float(n_det)
            base = {
                "object_id": object_id,
                "n_det": int(n_det),
                "alert_jd": alert_jd,
                "target_class": label,
                "target_follow_proxy": float(label == "snia"),
                "label_source": "tns_spectroscopic",
                "label_quality": "spectroscopic",
                "t_since_first": float(n_det) * 2.0 + rng.uniform(0.0, 1.0),
                "mag_last": 19.0 + rng.normal(0.0, 0.6),
                # class-informative LC feature -> q_prior carries signal
                "mag_range": (2.0 if is_sn else 0.5) + rng.normal(0.0, 0.2),
                "lc_noise": rng.normal(),
                "survey_is_lsst": 0.0,
            }
            snapshot_row = dict(base)
            for expert_key in EXPERTS:
                san = sanitize_expert_key(expert_key)
                available = _avail(expert_key, idx)
                snapshot_row[f"avail__{san}"] = 1.0 if available else 0.0
                if not available:
                    continue
                if expert_key == "fink/slsn":
                    # SN-filter projection: p_snia capped at 0.
                    score = float(np.clip(0.15 + 0.7 * is_sn + rng.uniform(0, 0.15), 0.01, 0.99))
                    p_vec = {"snia": 0.0, "nonIa_snlike": score, "other": 1.0 - score}
                else:
                    margin_draw = rng.uniform(0.0, 1.0)
                    correct = rng.random() < (0.1 + 0.8 * margin_draw)
                    if correct:
                        pred = label
                    else:
                        pred = str(rng.choice([c for c in CLASSES if c != label]))
                    p_top = (1.0 + 2.0 * margin_draw) / 3.0
                    rest = 1.0 - p_top
                    others = [c for c in CLASSES if c != pred]
                    p_vec = {pred: p_top, others[0]: rest * 0.6, others[1]: rest * 0.4}
                ranked = sorted(p_vec.items(), key=lambda item: item[1], reverse=True)
                mapped_pred = ranked[0][0]
                top1 = ranked[0][1]
                margin = ranked[0][1] - ranked[1][1]
                proj = {
                    f"proj__{san}__p_snia": p_vec["snia"],
                    f"proj__{san}__p_nonIa_snlike": p_vec["nonIa_snlike"],
                    f"proj__{san}__p_other": p_vec["other"],
                    f"proj__{san}__top1_prob": top1,
                    f"proj__{san}__margin": margin,
                    f"proj__{san}__entropy": _entropy(list(p_vec.values())),
                }
                snapshot_row.update(proj)
                snapshot_row[f"exact__{san}"] = 1.0
                snapshot_row[f"temporal_exactness__{san}"] = "exact_alert"
                snapshot_row[f"mapped_pred_class__{san}"] = mapped_pred
                helpfulness_rows.append(
                    {
                        **base,
                        **proj,
                        "expert_key": expert_key,
                        "available": True,
                        "temporal_exactness": "exact_alert",
                        "mapped_pred_class": mapped_pred,
                        f"avail__{san}": 1.0,
                        f"exact__{san}": 1.0,
                        "is_topclass_correct": int(mapped_pred == label),
                        "is_sn": is_sn,
                    }
                )
            snapshot_rows.append(snapshot_row)

    snapshots = pd.DataFrame(snapshot_rows)
    helpfulness = pd.DataFrame(helpfulness_rows)
    train_ids = {f"OBJ{i:04d}" for i in range(0, 150)}
    cal_ids = {f"OBJ{i:04d}" for i in range(150, 250)}
    test_ids = {f"OBJ{i:04d}" for i in range(250, 300)}
    return snapshots, helpfulness, train_ids, cal_ids, test_ids


@pytest.fixture(scope="module")
def synthetic():
    return _make_synthetic()


@pytest.fixture(scope="module")
def trained(synthetic, tmp_path_factory):
    snapshots, helpfulness, train_ids, cal_ids, test_ids = synthetic
    helpfulness = helpfulness.copy()
    # Stacking-leak probes: informative columns that MUST stay out of X_A.
    rng = np.random.default_rng(0)
    helpfulness["q__hack"] = rng.random(len(helpfulness))
    helpfulness["trust_source__hack"] = "oof"
    out_dir = tmp_path_factory.mktemp("trust_fusion_v8")
    result = train_pooled_trust(
        helpfulness,
        snapshots,
        train_ids,
        cal_ids,
        test_ids,
        str(out_dir),
        n_jobs=2,
        seed=42,
        grid_small=True,
    )
    return snapshots, helpfulness, (train_ids, cal_ids, test_ids), out_dir, result


# ---------------------------------------------------------------------------
# (a) trust-target parity
# ---------------------------------------------------------------------------


def test_trust_target_parity_synthetic(synthetic) -> None:
    _, helpfulness, *_ = synthetic
    long_df, _, expert_frames = assemble_stage_a_long(
        helpfulness, apply_honesty_filters=False
    )
    for expert_key in EXPERTS:
        sub = helpfulness[helpfulness["expert_key"] == expert_key]
        rows = long_df[long_df["expert_key"] == expert_key]
        expected_col = "is_sn" if expert_key == "fink/slsn" else "is_topclass_correct"
        assert rows["target_col"].unique().tolist() == [expected_col]
        assert rows["y"].mean() == pytest.approx(sub[expected_col].mean(), abs=1e-12)
    # The SN-filter target genuinely differs from top-class correctness here
    # (slsn never predicts snia), so parity is a real check, not a tautology.
    slsn = helpfulness[helpfulness["expert_key"] == "fink/slsn"]
    assert abs(slsn["is_sn"].mean() - slsn["is_topclass_correct"].mean()) > 0.05


def test_honesty_filters_verbatim() -> None:
    rows = []

    def _row(expert_key, object_id, label_source, exactness):
        return {
            "object_id": object_id,
            "n_det": 3,
            "alert_jd": 2460000.5,
            "expert_key": expert_key,
            "temporal_exactness": exactness,
            "label_source": label_source,
            "label_quality": "spectroscopic",
            "is_topclass_correct": 1,
            "is_sn": 1,
            "t_since_first": 1.0,
        }

    for i in range(3):
        rows.append(_row("alerce_lc", f"A{i}", "tns_spectroscopic", "exact_alert"))
    for i in range(2):
        rows.append(_row("alerce_lc", f"AC{i}", "alerce_self_label", "exact_alert"))
    for i in range(3):
        rows.append(_row("fink/snn", f"F{i}", "tns_spectroscopic", "exact_alert"))
    for i in range(2):
        rows.append(_row("fink/snn", f"FC{i}", "broker_consensus", "exact_alert"))
    rows.append(_row("fink/snn", "FU0", "tns_spectroscopic", "latest_object_unsafe"))
    # alerce_self_label rows are circular ONLY for alerce experts:
    rows.append(_row("supernnova", "S0", "alerce_self_label", "exact_alert"))
    rows.append(_row("supernnova", "S1", "tns_spectroscopic", "exact_alert"))

    frame = pd.DataFrame(rows)
    long_filtered, _, _ = assemble_stage_a_long(frame, apply_honesty_filters=True)
    counts = long_filtered.groupby("expert_key").size().to_dict()
    assert counts == {"alerce_lc": 3, "fink/snn": 3, "supernnova": 2}

    long_raw, _, _ = assemble_stage_a_long(frame, apply_honesty_filters=False)
    assert len(long_raw) == len(frame)
    # temporal exactness encoded numerically
    assert set(long_filtered["temporal_exactness_code"].unique()) == {2.0}


# ---------------------------------------------------------------------------
# (f) split integrity
# ---------------------------------------------------------------------------


def test_split_integrity_asserts_fire(synthetic, tmp_path) -> None:
    snapshots, helpfulness, train_ids, cal_ids, test_ids = synthetic
    overlapping_cal = set(cal_ids) | {next(iter(train_ids))}
    with pytest.raises(AssertionError, match="train/cal overlap"):
        train_pooled_trust(
            helpfulness, snapshots, train_ids, overlapping_cal, test_ids,
            str(tmp_path / "m"), n_jobs=2, seed=42, grid_small=True,
        )


# ---------------------------------------------------------------------------
# Trained-model assertions (single shared training run)
# ---------------------------------------------------------------------------


def test_result_shape_and_metric_keys(trained) -> None:
    snapshots, _, _, out_dir, result = trained
    assert isinstance(result, PooledTrustResult)
    assert result.artifact_dir == str(out_dir)
    assert len(result.snapshots) == len(snapshots)
    pinned = {"raw_auc", "cal_auc", "brier", "ece", "n_test", "calibrator_kind", "fallback_used"}
    for expert_key in EXPERTS:
        assert expert_key in result.metrics
        assert pinned <= set(result.metrics[expert_key])
        assert result.metrics[expert_key]["fallback_used"] is False
    assert "_pooled" in result.metrics
    # pooled model genuinely learned the planted margin->correctness signal
    assert result.metrics["fink/snn"]["raw_auc"] > 0.6


def test_calibration_tiers_fire_per_guards(trained) -> None:
    _, _, _, out_dir, result = trained
    # fink/snn: 100 cal objects x 3 epochs = 300 valid cal rows -> isotonic.
    assert result.metrics["fink/snn"]["calibrator_kind"] == "isotonic"
    # fink/slsn: 50 cal objects x 3 = 150 rows (30 <= n < 200) -> platt.
    assert result.metrics["fink/slsn"]["calibrator_kind"] == "platt"
    # supernnova: 7 cal objects x 3 = 21 rows (< 30) -> global fallback.
    assert result.metrics["supernnova"]["calibrator_kind"] == "global"
    # artifact layout matches the tiers
    assert (out_dir / "pooled" / "model.pkl").exists()
    assert (out_dir / "pooled" / "metadata.json").exists()
    assert (out_dir / "pooled" / "global_calibrator.pkl").exists()
    assert (out_dir / "fink__snn" / "calibrator.pkl").exists()
    assert (out_dir / "fink__slsn" / "calibrator.pkl").exists()
    assert not (out_dir / "supernnova" / "calibrator.pkl").exists()
    assert (out_dir / "supernnova" / "metadata.json").exists()


def test_q_nan_exactly_where_absent(trained) -> None:
    _, _, _, _, result = trained
    out = result.snapshots
    for expert_key in ("fink/slsn", "supernnova"):
        san = sanitize_expert_key(expert_key)
        absent = out[f"avail__{san}"].to_numpy(dtype=float) <= 0
        assert absent.any() and (~absent).any()
        q = out[f"q__{san}"].to_numpy(dtype=float)
        assert np.isnan(q[absent]).all()
        assert np.isfinite(q[~absent]).all()
        source = out[f"trust_source__{san}"].to_numpy()
        assert (source[absent] == "unavailable").all()
        assert set(source[~absent]) <= {"oof", "train_model"}


def test_oof_rows_differ_from_refit_rows(trained) -> None:
    _, _, (train_ids, _, _), out_dir, result = trained
    out = result.snapshots
    san = sanitize_expert_key("fink/slsn")
    mask = (
        out["object_id"].astype(str).isin(train_ids)
        & (out[f"avail__{san}"].to_numpy(dtype=float) > 0)
    ).to_numpy()
    assert mask.sum() > 50
    assert (out.loc[mask, f"trust_source__{san}"] == "oof").all()
    cal_test_mask = (
        ~out["object_id"].astype(str).isin(train_ids)
        & (out[f"avail__{san}"].to_numpy(dtype=float) > 0)
    ).to_numpy()
    assert (out.loc[cal_test_mask, f"trust_source__{san}"] == "train_model").all()

    # The view scores with the refit-on-train model; snapshot train rows hold
    # OOF values. The platt calibrator is strictly monotone, so any raw
    # difference survives calibration.
    view = PooledTrustView.load(str(out_dir / san))
    refit_q = view.predict_trust(out.loc[mask])
    oof_q = out.loc[mask, f"q__{san}"].to_numpy(dtype=float)
    assert refit_q.shape == oof_q.shape
    assert (np.abs(refit_q - oof_q) > 1e-9).sum() >= 5


def test_q_prior_defined_where_absent_and_differs(trained) -> None:
    _, _, _, _, result = trained
    out = result.snapshots
    san = sanitize_expert_key("fink/slsn")
    q = out[f"q__{san}"].to_numpy(dtype=float)
    q_prior = out[f"q_prior__{san}"].to_numpy(dtype=float)
    absent = out[f"avail__{san}"].to_numpy(dtype=float) <= 0
    # prior is defined everywhere, including where the expert never fired
    assert np.isfinite(q_prior).all()
    assert np.isnan(q[absent]).all()
    # with own_* slots NaN'd the prediction must change on available rows
    avail = ~absent
    assert (np.abs(q_prior[avail] - q[avail]) > 1e-9).sum() >= 5
    # emitted for ALL registered experts, even ones never seen in training
    assert "q_prior__parsnip" in out.columns
    assert np.isfinite(out["q_prior__parsnip"].to_numpy(dtype=float)).all()


def test_stage_a_leakage_guard(trained) -> None:
    import json

    _, _, _, out_dir, _ = trained
    with open(out_dir / "pooled" / "metadata.json") as fh:
        metadata = json.load(fh)
    feature_cols = metadata["feature_cols"]
    bad = [c for c in feature_cols if c.startswith(("q__", "q_prior__", "trust_source__"))]
    assert bad == []
    assert "q__hack" not in feature_cols
    assert "alert_jd" not in feature_cols
    # pinned generic slots present
    for slot in ("own_p_snia", "own_margin", "is_sn_filter", "survey_match",
                 "temporal_exactness_code", "log_expert_train_rows"):
        assert slot in feature_cols


def test_view_predictions_well_formed(trained) -> None:
    snapshots, _, _, out_dir, _ = trained
    view = PooledTrustView.load(str(out_dir / "fink__snn"))
    assert view.fallback_used is False
    preds = view.predict_trust(snapshots)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(snapshots),)
    assert np.isfinite(preds).all()
    assert ((preds >= 0.0) & (preds <= 1.0)).all()


# ---------------------------------------------------------------------------
# Spec correction #5: dedicated-head fallback gate
# ---------------------------------------------------------------------------


def test_should_fallback_gate_logic() -> None:
    assert _should_fallback(0.80, 0.75) is True       # regression > 0.02
    assert _should_fallback(0.76, 0.75) is False      # within tolerance
    assert _should_fallback(0.77001, 0.75) is True
    assert _should_fallback(None, 0.75) is False      # not evaluable
    assert _should_fallback(0.80, None) is False
    assert _should_fallback(0.80, 0.75, tol=0.10) is False


def test_fallback_integration_forced(synthetic, tmp_path, monkeypatch) -> None:
    snapshots, helpfulness, train_ids, cal_ids, test_ids = synthetic
    # Force the gate: every expert counts as data-rich and any computable
    # dedicated-vs-pooled comparison triggers the fallback.
    monkeypatch.setattr(pooled_trust, "DATA_RICH_MIN_ROWS", 1)
    monkeypatch.setattr(pooled_trust, "FALLBACK_AUC_TOL", -10.0)
    out_dir = tmp_path / "trust_fb"
    result = train_pooled_trust(
        helpfulness, snapshots, train_ids, cal_ids, test_ids,
        str(out_dir), n_jobs=2, seed=42, grid_small=True,
    )
    metrics = result.metrics["fink/snn"]
    assert metrics["fallback_used"] is True
    assert metrics["dedicated_auc"] is not None
    assert metrics["pooled_raw_auc"] is not None
    assert (out_dir / "fink__snn" / "model.pkl").exists()

    view = PooledTrustView.load(str(out_dir / "fink__snn"))
    assert view.fallback_used is True
    preds = view.predict_trust(snapshots)
    assert preds.shape == (len(snapshots),)
    assert np.isfinite(preds).all()
    assert ((preds >= 0.0) & (preds <= 1.0)).all()
    # q emission still NaN-absent under the fallback path
    san = sanitize_expert_key("fink/slsn")
    absent = result.snapshots[f"avail__{san}"].to_numpy(dtype=float) <= 0
    assert np.isnan(result.snapshots.loc[absent, f"q__{san}"].to_numpy(dtype=float)).all()


# ---------------------------------------------------------------------------
# Spec correction #4e: parity against the real helpfulness table
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not REAL_HELPFULNESS.exists(), reason="expert_helpfulness.parquet not present")
def test_trust_target_parity_real_base_rates() -> None:
    helpfulness = pd.read_parquet(REAL_HELPFULNESS)
    long_df, _, _ = assemble_stage_a_long(helpfulness, apply_honesty_filters=False)
    expected = {"fink/snn": 0.290, "fink/rf_ia": 0.652, "supernnova": 0.478}
    for expert_key, base_rate in expected.items():
        rows = long_df[long_df["expert_key"] == expert_key]
        assert len(rows) > 0, f"no Stage-A rows assembled for {expert_key}"
        assert rows["y"].mean() == pytest.approx(base_rate, abs=0.005), expert_key


def test_dedicated_head_handles_empty_test_split(synthetic):
    """Regression (SCC v9c crash): a data-rich expert whose rows all come from
    train/cal-only objects (the LSST weak-label cohort) has ZERO rows in the
    locked test set — the dedicated-head comparison must degrade gracefully,
    never feed LightGBM an empty frame."""
    from debass_meta.models.pooled_trust import _should_fallback, _train_dedicated_head

    snapshots, helpfulness, train_ids, cal_ids, test_ids = synthetic
    expert_key = sorted(helpfulness["expert_key"].unique())[0]
    rows = helpfulness[helpfulness["expert_key"] == expert_key].copy()
    # Re-split so NO object of this expert is in test: would-be-test → train.
    all_ids = set(rows["object_id"].astype(str))
    new_train = (set(train_ids) | set(test_ids)) & all_ids
    new_cal = set(cal_ids) & all_ids
    result = _train_dedicated_head(
        rows, expert_key, "is_topclass_correct",
        new_train, new_cal, {"NO_SUCH_OBJECT"},
        seed=42, n_jobs=2,
    )
    assert result is not None
    assert len(result["q_test"]) == 0 and len(result["y_test"]) == 0
    assert result["test_auc"] is None
    assert len(result["q_cal"]) == len(result["y_cal"]) > 0
    assert _should_fallback(result["test_auc"], 0.9) is False
