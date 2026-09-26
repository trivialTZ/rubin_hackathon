"""fusion v13d anchored-blend options: call-trust anchor weights, SN-vs-other α
objective, object-clustered SE, object-level base rate, head feature-prefix drop.
Defaults must reproduce the v13c behaviour exactly."""
from __future__ import annotations

import numpy as np
import pandas as pd

from debass_meta.models.anchor_blend import (
    BlendSpec,
    _fit_alpha_1se,
    apply,
    compute_anchor,
    fit_alpha,
)
from debass_meta.models.hierarchical_followup import HierarchicalFollowup

STAMP = "alerce__stamp_classifier_rubin_beta"   # SN filter, trust target is_sn
CATS = "fink_lsst__cats"


def _row(stamp_sn: float, stamp_q: float, cats_sn: float, cats_q: float) -> dict:
    return {
        "survey": "lsst",
        f"proj__{STAMP}__p_snia": 0.0, f"proj__{STAMP}__p_nonIa_snlike": stamp_sn,
        f"proj__{STAMP}__p_other": 1.0 - stamp_sn, f"avail__{STAMP}": 1, f"q__{STAMP}": stamp_q,
        f"proj__{CATS}__p_snia": cats_sn / 2, f"proj__{CATS}__p_nonIa_snlike": cats_sn / 2,
        f"proj__{CATS}__p_other": 1.0 - cats_sn, f"avail__{CATS}": 1, f"q__{CATS}": cats_q,
    }


def _p_sn(df: pd.DataFrame) -> np.ndarray:
    return (df["p_snia_anchor"] + df["p_nonia_anchor"]).to_numpy()


def test_call_weights_keep_a_correct_not_sn_call():
    # The stamp says "not SN" (0.03) and its is_sn trust head agrees (q = P(SN) = 0.02);
    # CATS says "SN" (0.86) with q 0.2.  With q weights the stamp barely counts and the
    # anchor follows CATS; with call weights the stamp carries 0.98 and CATS 0.2.
    df = pd.DataFrame([_row(0.03, 0.02, 0.86, 0.2)])
    q_w = _p_sn(compute_anchor(df))[0]
    call_w = _p_sn(compute_anchor(df, call_weighted_experts=("alerce/stamp_classifier_rubin_beta",
                                                              "fink_lsst/cats")))[0]
    assert q_w > 0.7
    assert abs(call_w - (0.98 * 0.03 + 0.2 * 0.86) / (0.98 + 0.2)) < 1e-6
    # an "SN" call keeps q
    df2 = pd.DataFrame([_row(0.9, 0.7, 0.9, 0.7)])
    assert np.allclose(_p_sn(compute_anchor(df2)),
                       _p_sn(compute_anchor(df2, call_weighted_experts=("fink_lsst/cats",))))


def test_default_spec_serializes_as_before_and_v13d_round_trips():
    assert "v13d" not in BlendSpec().to_dict()
    spec = BlendSpec(anchor_call_experts=["fink_lsst/cats"], alpha_objective="sn_binary",
                     alpha_se="object", base_rate_unit="object")
    back = BlendSpec.from_dict(spec.to_dict())
    assert back.anchor_call_experts == ["fink_lsst/cats"]
    assert (back.alpha_objective, back.alpha_se, back.base_rate_unit) == ("sn_binary", "object", "object")
    assert back.anchor_kwargs()["call_weighted_experts"] == ("fink_lsst/cats",)


def test_sn_binary_objective_ignores_the_ia_split():
    # Model: right on SN vs other, wrong on Ia vs non-Ia.  Anchor: hedged everywhere.
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 3, n)                      # 0 snia, 1 nonIa, 2 other
    model = np.where((y < 2)[:, None], [[0.05, 0.9, 0.05]], [[0.02, 0.02, 0.96]])
    model[y == 1] = [0.9, 0.05, 0.05]              # Ia/non-Ia swapped
    anchor = np.tile([0.3, 0.3, 0.4], (n, 1))
    a_multi, _ = _fit_alpha_1se(model, anchor, y)
    a_sn, info = _fit_alpha_1se(model, anchor, y, objective="sn_binary")
    assert a_sn == 1.0 and info["best_grid"] == 1.0
    assert a_multi < 1.0


def test_object_clustered_se_counts_copies_once():
    rng = np.random.default_rng(1)
    n_obj, copies = 60, 5
    y = np.repeat(rng.integers(0, 3, n_obj), copies)
    model = rng.dirichlet([1, 1, 1], n_obj).repeat(copies, axis=0)
    anchor = np.tile([1 / 3, 1 / 3, 1 / 3], (n_obj * copies, 1))
    groups = np.repeat(np.arange(n_obj), copies)
    _, row = _fit_alpha_1se(model, anchor, y)
    _, obj = _fit_alpha_1se(model, anchor, y, groups=groups)
    assert obj["means"] == row["means"]
    assert obj["se"] > 1.5 * row["se"]          # ~sqrt(5) for exact duplicates


def _alpha_frame() -> pd.DataFrame:
    rows = []
    # object A: an Ia with 30 epochs; objects B..E: non-Ia with 1 epoch each
    for oid, cls, n in [("A", "snia", 30)] + [(k, "nonIa_snlike", 1) for k in "BCDE"]:
        for _ in range(n):
            r = _row(0.9, 0.8, 0.9, 0.8)
            r.update(object_id=oid, target_class=cls, n_det=5,
                     p_snia=0.45, p_nonia=0.45, p_other=0.1)
            rows.append(r)
    return pd.DataFrame(rows)


def test_object_level_base_rate():
    df = _alpha_frame()
    assert abs(fit_alpha(df).base_rates["lsst"] - 30 / 34) < 1e-9
    spec = fit_alpha(df, base_rate_unit="object", alpha_objective="sn_binary", alpha_se="object",
                     anchor_call_experts=("fink_lsst/cats",))
    assert abs(spec.base_rates["lsst"] - 1 / 5) < 1e-9
    assert spec.to_dict()["v13d"]["base_rate_unit"] == "object"


def test_apply_recomputes_a_call_weighted_anchor():
    df = pd.DataFrame([_row(0.03, 0.02, 0.86, 0.2)]).assign(p_snia=0.0, p_nonia=0.05, p_other=0.95)
    stale = compute_anchor(df)                     # a caller's q-weighted anchor
    spec = BlendSpec(anchor_call_experts=["alerce/stamp_classifier_rubin_beta", "fink_lsst/cats"],
                     alpha_cells={"lsst|2+|<0.25": {"alpha": 0.0, "n": 99}})
    out = apply(stale, spec)
    assert np.allclose(_p_sn(out), _p_sn(compute_anchor(df, **spec.anchor_kwargs())))
    assert _p_sn(out)[0] < 0.3


def test_head_feature_prefix_drop_is_persisted():
    head = HierarchicalFollowup(feature_drop_prefixes=("event_count__", "exact__"))
    cols = ["n_det", "event_count__alerce_lc", "exact__fink__snn", "q__alerce_lc"]
    assert head._drop_prefixed(cols) == ["n_det", "q__alerce_lc"]
    assert head._v13_settings()["feature_drop_prefixes"] == ["event_count__", "exact__"]
    assert "feature_drop_prefixes" not in HierarchicalFollowup()._v13_settings()


def test_alpha_rule_best_takes_the_grid_minimum():
    rng = np.random.default_rng(2)
    n_obj, copies = 20, 10
    y = np.repeat(rng.integers(0, 3, n_obj), copies)
    model = np.full((n_obj * copies, 3), 0.05)
    model[np.arange(len(y)), y] = 0.9
    model[::7] = [1 / 3, 1 / 3, 1 / 3]
    anchor = np.tile([0.3, 0.3, 0.4], (len(y), 1))
    groups = np.repeat(np.arange(n_obj), copies)
    a_1se, info = _fit_alpha_1se(model, anchor, y, groups=groups, objective="sn_binary")
    a_best, info_b = _fit_alpha_1se(model, anchor, y, groups=groups, objective="sn_binary", rule="best")
    assert a_best == info_b["best_grid"] and a_1se <= a_best
    assert BlendSpec.from_dict(BlendSpec(alpha_rule="best").to_dict()).alpha_rule == "best"
    assert "v13d" not in BlendSpec(alpha_rule="1se").to_dict()
