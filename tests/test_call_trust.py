"""v13g call trust: one meaning of trust (``--stage-a-call-trust``).

Synthetic Stage-A frames of ``test_v13_stage_a`` (two ternary experts, two SN
filters, two headless context experts).  Covers: the main Stage-A outputs and
the artifact layout are unchanged by default and when the heads are added;
heads for exactly the non-SN-filter experts with a trust head, SN-filter experts
reused; same folds as the main model and out-of-fold ``q_sn`` on train rows;
cal/test parity between the training snapshot and the scorer; the
``call_trust`` / ``sn_call`` definition and NaN rules; the new columns never
become head inputs; the scorer end to end; old artifacts untouched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_v13_stage_a import (  # noqa: E402
    HEADLESS_EXPERTS,
    SN_FILTER_SYNTH,
    TERNARY_EXPERTS,
    _make_synthetic,
    _train,
)

from debass_meta.models import pooled_trust  # noqa: E402
from debass_meta.models.call_trust import (  # noqa: E402
    CALL_TRUST_SUBDIR,
    attach_call_trust,
    call_trust_from_q_sn,
    expert_p_sn,
    load_call_trust_assets,
)
from debass_meta.models.multiclass_followup import _numeric_feature_cols  # noqa: E402
from debass_meta.projectors import sanitize_expert_key  # noqa: E402

SAN = sanitize_expert_key
HEADED = TERNARY_EXPERTS + SN_FILTER_SYNTH


@pytest.fixture(scope="module")
def synthetic():
    return _make_synthetic()


@pytest.fixture(scope="module")
def plain(synthetic, tmp_path_factory):
    out = tmp_path_factory.mktemp("trust_plain")
    return out, _train(synthetic, out)


@pytest.fixture(scope="module")
def with_ct(synthetic, tmp_path_factory):
    out = tmp_path_factory.mktemp("trust_ct")
    return out, _train(synthetic, out, call_trust=True)


def _scored(synthetic, trust_dir):
    from scripts.score_fusion_v8 import attach_trust_columns

    return attach_call_trust(attach_trust_columns(synthetic[0].copy(), trust_dir), trust_dir)


# ---------------------------------------------------------------- defaults

def test_default_has_no_call_trust(plain, synthetic):
    out, result = plain
    assert not (out / "pooled" / CALL_TRUST_SUBDIR).exists()
    assert load_call_trust_assets(out) is None
    assert "call_trust" not in json.loads((out / "pooled" / "metadata.json").read_text())
    assert "call_trust" not in result.metrics["_pooled"]
    assert not [c for c in result.snapshots.columns if c.startswith(("q_sn__", "call_trust__", "sn_call__"))]
    # the scorer step is a no-op for an artifact without the heads
    scored = _scored(synthetic, out)
    assert not [c for c in scored.columns if c.startswith(("q_sn__", "call_trust__", "sn_call__"))]


def test_adding_the_heads_leaves_stage_a_unchanged(plain, with_ct):
    (_, a), (_, b) = plain, with_ct
    new = [c for c in b.snapshots.columns if c.startswith("q_sn__")]
    assert new and set(b.snapshots.columns) - set(a.snapshots.columns) == set(new)
    pd.testing.assert_frame_equal(a.snapshots, b.snapshots[a.snapshots.columns], check_exact=True)
    for key, m in a.metrics.items():
        if key == "_pooled":
            continue
        assert b.metrics[key] == m
    pa, pb = dict(a.metrics["_pooled"]), dict(b.metrics["_pooled"])
    pb.pop("call_trust")
    assert pb["emission"].pop("q_sn_experts")
    assert pa == pb


# ------------------------------------------------------------- the heads

def test_heads_for_non_sn_filter_experts_and_sn_filter_reused(with_ct):
    out, result = with_ct
    meta = json.loads((out / "pooled" / CALL_TRUST_SUBDIR / "metadata.json").read_text())
    assert sorted(meta["experts"]) == sorted(TERNARY_EXPERTS)          # no duplicate head for SN filters
    assert sorted(meta["reused_sn_filter_experts"]) == sorted(SN_FILTER_SYNTH)
    assert meta["target"] == "is_sn" and meta["same_folds_as_pooled"] and meta["n_folds"] == 5
    assert meta["sn_call_threshold"] == 0.5
    snap = result.snapshots
    for key in HEADED:
        assert f"q_sn__{SAN(key)}" in snap.columns
    for key in HEADLESS_EXPERTS:                                        # no trust head -> no call trust
        assert f"q_sn__{SAN(key)}" not in snap.columns
    for key in SN_FILTER_SYNTH:                                         # their q__ already is P(SN)
        np.testing.assert_array_equal(snap[f"q_sn__{SAN(key)}"].to_numpy(float),
                                      snap[f"q__{SAN(key)}"].to_numpy(float))
    s = result.metrics["_pooled"]["call_trust"]
    assert s["experts_fitted"] == sorted(TERNARY_EXPERTS) and s["same_folds_as_pooled"]
    for key in TERNARY_EXPERTS:
        assert s["per_expert"][key]["n_train_rows"] > 0
        assert s["per_expert"][key]["raw_auc"] is not None
    # the is_sn head is not the top-class-correct head: different target, different values
    for key in TERNARY_EXPERTS:
        q, qs = (result.snapshots[f"{p}__{SAN(key)}"].to_numpy(float) for p in ("q", "q_sn"))
        assert np.nanmax(np.abs(q - qs)) > 0.05


def test_same_rows_same_folds(with_ct, synthetic):
    """The is_sn heads' held-out fold of every train object is the main model's."""
    out, result = with_ct
    snapshots, helpfulness, train_ids, cal_ids, test_ids = synthetic
    long_df, _, _ = pooled_trust.assemble_stage_a_long(
        helpfulness, weak_policy="all", with_sn_target=True)
    tr = long_df[long_df["object_id"].isin(train_ids) & long_df["y"].notna()]
    groups = tr["object_id"].astype(str).to_numpy()
    from debass_meta.models.folds import StableGroupKFold

    main_fold = StableGroupKFold(5).fold_of_rows(groups)
    fixed = dict(zip(groups.tolist(), main_fold.tolist()))
    sub = tr[tr["expert_key"].isin(TERNARY_EXPERTS)]
    X = pooled_trust._prepare_pooled_matrix(sub, ["mag_last"], sorted(set(tr["expert_key"])))
    y = sub["y_sn"].to_numpy(int)
    w = np.ones(len(sub))
    oof, bundles, fold_of_group = pooled_trust._oof_pooled_fits(
        X, y, w, sub["object_id"].astype(str).to_numpy(), params={}, n_estimators=5, seed=1,
        n_jobs=1, fixed_fold_of_group=fixed)
    assert len(bundles) == 5 and np.isfinite(oof).all()
    assert all(fold_of_group[g] == fixed[g] for g in fold_of_group)
    # default path unchanged: without the map the folds are the stable balanced ones
    _, _, default_folds = pooled_trust._oof_pooled_fits(
        X, y, w, sub["object_id"].astype(str).to_numpy(), params={}, n_estimators=5, seed=1, n_jobs=1)
    assert len(default_folds) == len(fold_of_group)


def test_train_rows_out_of_fold_and_heldout_rows_match_scorer(with_ct, synthetic):
    out, result = with_ct
    trained = result.snapshots
    scored = _scored(synthetic, out)                    # scorer from the saved artifact only
    train_ids = synthetic[2]
    is_train = trained["object_id"].astype(str).isin(train_ids).to_numpy()
    for key in TERNARY_EXPERTS:
        col = f"q_sn__{SAN(key)}"
        a, b = trained[col].to_numpy(float), scored[col].to_numpy(float)
        np.testing.assert_allclose(a[~is_train], b[~is_train], atol=1e-9, equal_nan=True, err_msg=col)
        assert (a[is_train] != b[is_train]).mean() > 0.2       # OOF in the snapshot, refit at scoring (the synthetic is near-separable: tiny gaps)
    for key in SN_FILTER_SYNTH:
        col = f"q_sn__{SAN(key)}"
        np.testing.assert_allclose(trained[col].to_numpy(float)[~is_train],
                                   scored[col].to_numpy(float)[~is_train], atol=1e-9, equal_nan=True)
    # a q_sn column already present (the training snapshot) is kept by the scorer, train rows included
    from scripts.score_fusion_v8 import attach_trust_columns

    again = attach_call_trust(attach_trust_columns(trained.copy(), out), out)
    col = f"q_sn__{SAN(TERNARY_EXPERTS[0])}"
    np.testing.assert_array_equal(again[col].to_numpy(float), trained[col].to_numpy(float))


# ------------------------------------------------------ the definition

def test_call_trust_definition(with_ct, synthetic):
    out, _ = with_ct
    snaps = synthetic[0].copy()
    key = TERNARY_EXPERTS[0]
    san = SAN(key)
    snaps.loc[snaps.index[:7], f"avail__{san}"] = 0.0       # expert unavailable on some rows
    from scripts.score_fusion_v8 import attach_trust_columns

    scored = attach_call_trust(attach_trust_columns(snaps, out), out)
    for k in HEADED:
        s = SAN(k)
        p_sn = (snaps[f"proj__{s}__p_snia"] + snaps[f"proj__{s}__p_nonIa_snlike"]).to_numpy(float)
        q_sn = scored[f"q_sn__{s}"].to_numpy(float)
        call = scored[f"sn_call__{s}"].to_numpy(float)
        ct = scored[f"call_trust__{s}"].to_numpy(float)
        avail = snaps[f"avail__{s}"].to_numpy(float) > 0
        assert set(np.unique(call[~np.isnan(call)])) <= {0.0, 1.0}
        np.testing.assert_array_equal(np.isnan(ct), ~avail)
        np.testing.assert_array_equal(np.isnan(call), ~avail)
        np.testing.assert_array_equal(call[avail], (p_sn[avail] >= 0.5).astype(float))
        exp = np.where(p_sn >= 0.5, q_sn, 1.0 - q_sn)
        np.testing.assert_allclose(ct[avail], exp[avail])
        assert ((ct[avail] >= 0) & (ct[avail] <= 1)).all()
    assert scored[f"call_trust__{san}"].isna().iloc[:7].all()
    # expert-by-expert q__ and the other columns are untouched
    base = attach_trust_columns(snaps.copy(), out)
    for c in base.columns:
        np.testing.assert_array_equal(base[c].to_numpy(), scored[c].to_numpy())
    assert not [c for c in scored.columns if c.startswith("call_trust__") and SAN("babamul") in c]


def test_call_trust_helpers_edge_cases():
    q = np.array([0.9, 0.9, 0.2, np.nan, 0.6])
    p = np.array([0.5, 0.49, 0.8, 0.9, np.nan])
    call, trust = call_trust_from_q_sn(q, p, np.array([True, True, True, True, True]))
    np.testing.assert_array_equal(call, [1.0, 0.0, 1.0, np.nan, np.nan])
    np.testing.assert_allclose(trust, [0.9, 0.1, 0.2, np.nan, np.nan])
    call, trust = call_trust_from_q_sn(q, p, np.array([False, True, True, True, True]))
    assert np.isnan(call[0]) and np.isnan(trust[0])
    df = pd.DataFrame({"proj__x__p_snia": [0.2, np.nan, np.nan], "proj__x__p_nonIa_snlike": [0.3, 0.4, np.nan]})
    np.testing.assert_allclose(expert_p_sn(df, "x"), [0.5, 0.4, np.nan])
    assert np.isnan(expert_p_sn(df, "missing")).all()


# ------------------------------------------------- not a head input

def test_new_columns_never_head_inputs(with_ct, plain):
    cols_with = _numeric_feature_cols(with_ct[1].snapshots)
    cols_plain = _numeric_feature_cols(plain[1].snapshots)
    assert cols_with == cols_plain
    assert not [c for c in cols_with if c.startswith(("q_sn__", "call_trust__", "sn_call__"))]
    scored = _scored(_make_synthetic(), with_ct[0]).assign(target_class="snia")
    assert [c for c in scored.columns if c.startswith(("q_sn__", "call_trust__", "sn_call__"))]
    assert not [c for c in _numeric_feature_cols(scored) if c.startswith(("q_sn__", "call_trust__", "sn_call__"))]


# ------------------------------------------------- scorer, end to end

def test_scorer_emits_columns_and_leaves_probabilities_unchanged(with_ct, plain, synthetic, tmp_path, monkeypatch):
    import importlib

    from debass_meta.models.anchor_blend import BlendSpec

    score_v11 = importlib.import_module("scripts.score_fusion_v11")

    class _Stub:
        def _p(self, df):
            x = np.linspace(0.1, 0.9, len(df))
            return np.column_stack([x * 0.5, x * 0.5, 1 - x])

        predict_proba = predict_proba_raw = _p

    monkeypatch.setattr(score_v11, "load_followup_artifact", lambda d: _Stub())
    snap = synthetic[0].copy()
    snap["survey"] = np.where(snap["survey_is_lsst"] > 0.5, "lsst", "ztf")
    path = tmp_path / "snap.parquet"
    snap.to_parquet(path, index=False)
    blend = tmp_path / "blend"
    BlendSpec(alpha_global={"alpha": 0.5, "n": 100}, base_rates={"ztf": 0.4}).save(blend)
    outs = {}
    for name, (trust, _) in {"ct": with_ct, "plain": plain}.items():
        out = tmp_path / f"{name}.parquet"
        score_v11.main(["--snapshots", str(path), "--blend-dir", str(blend),
                        "--followup-dir", str(tmp_path / "nofollowup"), "--trust-dir", str(trust),
                        "--conformal", str(tmp_path / "noconf.pkl"), "--out", str(out), "--no-priority"])
        outs[name] = pd.read_parquet(out)
    a, b = outs["ct"], outs["plain"]
    extra = [c for c in a.columns if c not in b.columns]
    assert extra and all(c.startswith(("q_sn__", "call_trust__", "sn_call__")) for c in extra)
    for k in HEADED:
        assert {f"q_sn__{SAN(k)}", f"call_trust__{SAN(k)}", f"sn_call__{SAN(k)}"} <= set(extra)
    pd.testing.assert_frame_equal(a[b.columns], b, check_exact=True)   # every existing column identical


# ------------------------------------------------------ orchestrator

def test_orchestrator_flag(tmp_path):
    from test_train_v11_smoke import _argv, _write_inputs

    from scripts.train_fusion_v11 import _build_arg_parser, main

    assert _build_arg_parser().parse_args([]).stage_a_call_trust is False
    snap, split_path, locked_path, truth_path = _write_inputs(tmp_path)
    with pytest.raises(SystemExit, match="call-trust"):
        main(_argv(tmp_path, snap, split_path, locked_path, truth_path, extra=("--stage-a-call-trust",)))
    # default: nothing recorded
    assert main(_argv(tmp_path, snap, split_path, locked_path, truth_path)) == 0
    rep = json.loads((tmp_path / "fusion_v11_train.json").read_text())
    assert "stage_a_call_trust" not in rep["v13"]
    # a reused Stage A that has the heads is accepted and recorded
    d = tmp_path / "trust_v11" / "pooled" / CALL_TRUST_SUBDIR
    d.mkdir(parents=True)
    (d / "metadata.json").write_text("{}")
    assert main(_argv(tmp_path, snap, split_path, locked_path, truth_path, extra=("--stage-a-call-trust",))) == 0
    rep = json.loads((tmp_path / "fusion_v11_train.json").read_text())
    assert rep["v13"]["stage_a_call_trust"] is True


def test_orchestrator_passes_call_trust_to_stage_a(tmp_path, monkeypatch):
    from test_train_v11_smoke import _argv, _write_inputs

    import scripts.train_fusion_v11 as tv

    snap, split_path, locked_path, truth_path = _write_inputs(tmp_path)
    help_path = tmp_path / "help.parquet"
    pd.DataFrame({"object_id": ["a"], "expert_key": ["fink/snn"], "n_det": [1]}).to_parquet(help_path)
    monkeypatch.setattr(tv, "_smoke_subset", lambda snaps, helps, *a, **k: (snaps, helps))   # keep the fixture's rows
    seen = {}

    def fake_train(helpfulness, snapshots, train_ids, cal_ids, test_ids, out_dir, *, n_jobs=1, seed=0,
                   grid_small=False, call_trust=False):
        seen["call_trust"] = call_trust
        return pooled_trust.PooledTrustResult(snapshots=snapshots, metrics={}, artifact_dir=out_dir)

    monkeypatch.setattr(pooled_trust, "train_pooled_trust", fake_train)
    argv = [a for a in _argv(tmp_path, snap, split_path, locked_path, truth_path) if a != "--skip-stage-a"]
    argv += ["--snapshots", str(snap), "--helpfulness", str(help_path)]   # never the default data/ gold
    assert tv.main(argv + ["--stage-a-call-trust"]) == 0
    assert seen["call_trust"] is True
    rep = json.loads((tmp_path / "fusion_v11_train.json").read_text())
    assert rep["stage_a"]["v13_passthrough"]["call_trust"] is True
    seen.clear()
    assert tv.main(argv) == 0
    assert seen["call_trust"] is False
