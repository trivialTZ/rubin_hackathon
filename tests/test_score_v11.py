"""fusion_v11 scorer — CLI mirror of v8 + anchored-blend columns (smoke)."""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from debass_meta.models.anchor_blend import BlendSpec
from debass_meta.projectors.base import sanitize_expert_key

score_v11 = importlib.import_module("scripts.score_fusion_v11")


class _StubFollowup:
    """Minimal followup artifact: random simplex from predict_proba{,_raw}."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def _simplex(self, df):
        x = self.rng.uniform(0.05, 1.0, size=(len(df), 3))
        return x / x.sum(axis=1, keepdims=True)

    def predict_proba_raw(self, df):
        return self._simplex(df)

    def predict_proba(self, df):
        return self._simplex(df)


def _san(key: str) -> str:
    return sanitize_expert_key(key)


def _snapshot(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    is_ia = rng.random(n) < 0.5
    rows = []
    for i in range(n):
        s = rng.uniform(0.1, 0.9)
        row = {
            "object_id": f"OBJ{i:05d}",
            "n_det": int(rng.integers(3, 8)),
            "survey": "ztf",
            "target_class": "snia" if is_ia[i] else "nonIa_snlike",
            "label_quality": "spectroscopic",
        }
        for key, ratio in (("fink/snn", 0.8), ("pittgoogle/supernnova_ztf", 0.6)):
            k = _san(key)
            row[f"proj__{k}__p_snia"] = s * ratio * 0.9
            row[f"proj__{k}__p_nonIa_snlike"] = s * (1 - ratio) * 0.9
            row[f"proj__{k}__p_other"] = 1 - s * 0.9
            row[f"avail__{k}"] = True
            row[f"q__{k}"] = 0.7
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture()
def scored(tmp_path, monkeypatch):
    snap = tmp_path / "snap.parquet"
    _snapshot().to_parquet(snap, index=False)
    blend_dir = tmp_path / "blend"
    BlendSpec(alpha_global={"alpha": 0.5, "n": 100},
              base_rates={"ztf": 0.4}).save(blend_dir)
    out = tmp_path / "pred.parquet"

    monkeypatch.setattr(score_v11, "load_followup_artifact",
                        lambda d: _StubFollowup())
    score_v11.main([
        "--snapshots", str(snap),
        "--blend-dir", str(blend_dir),
        "--followup-dir", str(tmp_path / "nofollowup"),
        "--trust-dir", str(tmp_path / "notrust"),
        "--conformal", str(tmp_path / "noconf.pkl"),
        "--out", str(out),
        "--no-priority",
    ])
    return pd.read_parquet(out)


def test_predictions_have_blend_columns(scored):
    pinned = {
        "object_id", "n_det", "survey", "target_class",
        "p_snia_raw", "p_nonia_raw", "p_other_raw",
        "p_snia_model", "p_nonia_model", "p_other_model",
        "p_snia_anchor", "p_nonia_anchor", "p_other_anchor",
        "n_experts_fired", "alpha", "alpha_fallback_level",
        "p_snia", "p_nonia", "p_other",
        "set_snia", "set_nonia", "set_other", "set_size",
        "s_ia", "s_nonia", "s_other", "p_follow_proxy", "ensemble_p_snia",
    }
    assert pinned <= set(scored.columns)


def test_deployed_probs_are_blend(scored):
    dep = scored[["p_snia", "p_nonia", "p_other"]].to_numpy(float)
    assert np.allclose(dep.sum(axis=1), 1.0, atol=1e-6)
    a = scored["alpha"].to_numpy(float)[:, None]
    model = scored[["p_snia_model", "p_nonia_model", "p_other_model"]].to_numpy(float)
    anchor = scored[["p_snia_anchor", "p_nonia_anchor", "p_other_anchor"]].to_numpy(float)
    expected = a * model + (1 - a) * np.where(np.isfinite(anchor), anchor, model)
    assert np.allclose(dep, expected, atol=1e-9)


def test_two_experts_fired(scored):
    assert (scored["n_experts_fired"] == 2).all()
    # anchor fallback level: the stub spec has empty cell/survey tables, so no
    # survey is "fitted" and the survey-guarded ladder must terminate at the
    # anchor (never the cross-survey global rung — 2026-07-05 smoke defect)
    assert set(scored["alpha_fallback_level"]) <= {"anchor_default", "no_anchor"}


def test_cli_mirrors_v8_plus_blend_dir():
    parser = score_v11._build_arg_parser()
    opts = {a.dest for a in parser._actions}
    for dest in ("snapshots", "dp1", "followup_dir", "trust_dir", "conformal",
                 "budgets", "fdr_gamma", "fdr_n_det_max", "split", "no_priority",
                 "smoke", "tag", "blend_dir"):
        assert dest in opts


def test_load_followup_falls_back_to_multiclass(tmp_path):
    """With no hierarchical head available, loader falls back to the v8 artifact
    (raising cleanly on a bogus dir rather than importing the wrong class)."""
    with pytest.raises(Exception):
        score_v11.load_followup_artifact(str(tmp_path / "does_not_exist"))
