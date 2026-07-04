"""--fdr-gamma wiring in scripts/score_fusion_v8.py (fusion_v10 fix #3).

Tests the cal-fit FDR threshold helper directly (the CLI plumbs it into a
``selected_fdr`` column per goal via selection.rank_and_select).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from debass_meta.models.selection import fdr_controlled_threshold, rank_and_select

_ROOT = Path(__file__).resolve().parents[1]


def _load_score_module():
    spec = importlib.util.spec_from_file_location(
        "score_fusion_v8_script", _ROOT / "scripts" / "score_fusion_v8.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def score_mod():
    return _load_score_module()


def _latest_frame(n: int = 120, seed: int = 3) -> pd.DataFrame:
    """One-row-per-object frame: high s_ia rows are mostly true snia."""
    rng = np.random.default_rng(seed)
    is_ia = np.arange(n) < n // 2
    s_ia = np.where(is_ia, rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.5, n))
    # plant impurity: 5 non-Ia objects above every Ia score
    s_ia[-5:] = 1.5
    return pd.DataFrame({
        "object_id": [f"OBJ{i:05d}" for i in range(n)],
        "n_det": np.full(n, 20),
        "target_class": np.where(is_ia, "snia", "nonIa_snlike"),
        "label_quality": "spectroscopic",
        "s_ia": s_ia,
        "s_nonia": 1.0 - s_ia,
        "s_other": np.full(n, 0.01),
    })


def _write_split(tmp_path: Path, cal_ids: list[str]) -> Path:
    path = tmp_path / "split.json"
    path.write_text(json.dumps({
        "split": {"train_ids": [], "cal_ids": cal_ids, "test_ids": []},
    }))
    return path


def test_cli_exposes_fdr_gamma(score_mod, monkeypatch):
    """--fdr-gamma parses as an optional float (default None)."""
    import argparse

    captured = {}
    orig = argparse.ArgumentParser.parse_args

    def fake_parse(self, args=None, namespace=None):
        ns = orig(self, ["--fdr-gamma", "0.2", "--no-priority", "--smoke"])
        captured.update(vars(ns))
        raise SystemExit(0)  # stop before any I/O

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", fake_parse)
    with pytest.raises(SystemExit):
        score_mod.main()
    assert captured["fdr_gamma"] == pytest.approx(0.2)
    assert "split" in captured


def test_fdr_thresholds_match_direct_call(score_mod, tmp_path):
    latest = _latest_frame()
    split = _write_split(tmp_path, latest["object_id"].tolist())
    gamma = 0.2

    thresholds = score_mod.compute_fdr_thresholds(latest, split, gamma)
    assert set(thresholds) == {"ia", "nonia", "other"}

    s = latest["s_ia"].to_numpy(float)
    y = (latest["target_class"] == "snia").to_numpy()
    assert thresholds["ia"]["tau"] == pytest.approx(
        fdr_controlled_threshold(s, y, gamma=gamma))

    # the fit n_det regime is recorded per goal (finding: a tau fit at
    # late epochs does not transfer to 3-5-det selections)
    assert thresholds["ia"]["fit_n_det_median"] == pytest.approx(20.0)
    assert thresholds["ia"]["fit_n_det_max"] == pytest.approx(20.0)
    assert thresholds["ia"]["fit_n_det_cap"] is None
    assert thresholds["ia"]["n_fit"] == len(latest)

    # goal=other has zero cal targets -> tau = inf (select nothing)
    assert not np.isfinite(thresholds["other"]["tau"])

    # empirical cal FDR of {s >= tau} respects gamma for the ia goal
    tau = thresholds["ia"]["tau"]
    picked = s >= tau
    assert picked.any()
    assert (1.0 - y[picked].mean()) <= gamma + 1e-12


def test_fdr_n_det_max_fits_on_early_epochs(score_mod, tmp_path):
    """--fdr-n-det-max caps the cal fit rows at each object's latest epoch
    with n_det <= cap; late epochs (better separated) never leak in."""
    latest = _latest_frame()
    early = latest.copy()
    early["n_det"] = 4
    # early scores are noisier: shrink separation toward 0.5
    early["s_ia"] = 0.5 + 0.4 * (early["s_ia"] - 0.5)
    frame = pd.concat([latest, early], ignore_index=True)
    split = _write_split(tmp_path, latest["object_id"].tolist())

    capped = score_mod.compute_fdr_thresholds(frame, split, 0.2, n_det_max=5)
    assert capped["ia"]["fit_n_det_max"] == pytest.approx(4.0)
    assert capped["ia"]["fit_n_det_cap"] == 5

    s = early["s_ia"].to_numpy(float)
    y = (early["target_class"] == "snia").to_numpy()
    assert capped["ia"]["tau"] == pytest.approx(
        fdr_controlled_threshold(s, y, gamma=0.2))

    # without the cap the latest (n_det=20) rows are used instead
    uncapped = score_mod.compute_fdr_thresholds(frame, split, 0.2)
    assert uncapped["ia"]["fit_n_det_max"] == pytest.approx(20.0)


def test_fdr_selected_column_via_rank_and_select(score_mod, tmp_path):
    """The threshold plumbs through rank_and_select exactly as the CLI does:
    budget-free selection of everything at or above tau."""
    latest = _latest_frame()
    split = _write_split(tmp_path, latest["object_id"].tolist())
    tau = score_mod.compute_fdr_thresholds(latest, split, 0.2)["ia"]["tau"]

    base = latest.copy()
    for col in ("set_snia", "set_nonia", "set_other"):
        base[col] = True
    base = base.rename(columns={"s_ia": "p_snia", "s_nonia": "p_nonia",
                                "s_other": "p_other"})
    res = rank_and_select(base, "ia", budget=len(base), score_threshold=tau)
    sel = res["selected"].to_numpy(bool)
    s = base["p_snia"].to_numpy(float)
    assert (sel == (s >= tau)).all()


def test_fdr_thresholds_graceful_degradation(score_mod, tmp_path):
    latest = _latest_frame()

    # missing split manifest -> {}
    assert score_mod.compute_fdr_thresholds(
        latest, tmp_path / "nope.json", 0.2) == {}

    # no labeled cal rows (DP1-like frame) -> {}
    unlabeled = latest.drop(columns=["target_class"])
    split = _write_split(tmp_path, latest["object_id"].tolist())
    assert score_mod.compute_fdr_thresholds(unlabeled, split, 0.2) == {}

    # too few cal objects -> {}
    tiny_split = _write_split(tmp_path, latest["object_id"].tolist()[:5])
    assert score_mod.compute_fdr_thresholds(latest, tiny_split, 0.2) == {}
