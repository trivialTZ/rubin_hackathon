"""Tests for fusion_v8 broker score trajectory features (design §1.3).

Covers:
  (a) hand-built trajectory → known n/last/mean/std/min/max/slope/delta/
      volatility/ewm values,
  (b) strict as-of leakage: events after alert_jd (or non-exact_alert events)
      never affect the output,
  (c) payload_hash dedupe: duplicated silver rows do not double-count,
  (d) empty / missing expert → NaN columns,
  (e) golden check against the real gold table: traj__fink__snn__last must
      equal proj__fink__snn__p_snia at the same (object_id, n_det),
  plus projector parity: the vectorized per-event p_snia must equal the
  output of the experts' existing projectors (never invent mappings).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from debass_meta.features.trajectory import (  # noqa: E402
    TRAJ_EXPERTS,
    TRAJ_FEATURE_NAMES,
    TRAJ_STATS,
    build_trajectory_features,
)
from debass_meta.projectors.base import _dispatch_projector, sanitize_expert_key  # noqa: E402

_GOLD_PATH = _ROOT / "data" / "gold" / "object_epoch_snapshots.parquet"
_SILVER_PATH = _ROOT / "data" / "silver" / "broker_events.parquet"

_T0 = 2_460_000.0  # realistic JD origin for synthetic events


def _ev(
    object_id: str,
    expert_key: str,
    t: float,
    score: float | None,
    *,
    field: str | None = None,
    class_name: str | None = None,
    alert_id: float | None = None,
    payload_hash: str = "hash0",
    temporal_exactness: str = "exact_alert",
    availability: bool = True,
) -> dict:
    """One synthetic silver broker_events row."""
    return {
        "object_id": object_id,
        "expert_key": expert_key,
        "field": field,
        "class_name": class_name,
        "event_time_jd": t,
        "alert_id": alert_id if alert_id is not None else t,
        "canonical_projection": score,
        "temporal_exactness": temporal_exactness,
        "payload_hash": payload_hash,
        "availability": availability,
    }


def _rf_events(object_id: str, times: list[float], scores: list[float]) -> list[dict]:
    """fink/rf_ia events — projector maps p_snia = score directly."""
    return [
        _ev(object_id, "fink/rf_ia", t, s, field="rf_snia_vs_nonia")
        for t, s in zip(times, scores)
    ]


def _gold_keys(object_id: str, alert_jds: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "object_id": object_id,
        "n_det": [float(i + 1) for i in range(len(alert_jds))],
        "alert_jd": alert_jds,
    })


def _col(expert: str, stat: str) -> str:
    return f"traj__{sanitize_expert_key(expert)}__{stat}"


def _reference_stats(times: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Straightforward loop/closed-form reference for the 10 trajectory stats."""
    n = len(scores)
    out = {
        "n": float(n),
        "last": scores[-1],
        "mean": scores.mean(),
        "std": scores.std(ddof=0),
        "min": scores.min(),
        "max": scores.max(),
        "delta": scores[-1] - scores[0],
    }
    if n >= 2:
        out["volatility"] = float(np.mean(np.abs(np.diff(scores))))
        tc = times - times.mean()
        denom = float((tc * tc).sum())
        out["slope"] = float((tc * (scores - scores.mean())).sum() / denom) if denom > 0 else np.nan
    else:
        out["volatility"] = np.nan
        out["slope"] = np.nan
    weights = 0.5 ** ((times[-1] - times) / 5.0)
    out["ewm"] = float((weights * scores).sum() / weights.sum())
    return out


# ---------------------------------------------------------------------------
# Shape / naming
# ---------------------------------------------------------------------------

def test_feature_names_shape_and_order() -> None:
    assert len(TRAJ_FEATURE_NAMES) == 66
    assert len(set(TRAJ_FEATURE_NAMES)) == 66
    assert len(TRAJ_EXPERTS) == 6
    assert TRAJ_FEATURE_NAMES[0] == "traj__fink__snn__n"
    assert TRAJ_FEATURE_NAMES[1] == "traj__fink__snn__last"
    assert TRAJ_FEATURE_NAMES[60:] == [
        "traj_x__n_experts", "traj_x__mean_last", "traj_x__max_last",
        "traj_x__mean_slope", "traj_x__disagreement_last", "traj_x__disagreement_trend",
    ]


# ---------------------------------------------------------------------------
# (a) Hand-built trajectory with known statistics
# ---------------------------------------------------------------------------

def test_hand_built_trajectory_known_stats() -> None:
    times = np.array([_T0, _T0 + 1.0, _T0 + 3.0, _T0 + 10.0])
    scores = np.array([0.1, 0.3, 0.2, 0.8])
    events = pd.DataFrame(_rf_events("obj1", list(times), list(scores)))
    # Epoch 1 sees the first 3 events; epoch 2 sees all 4.
    keys = _gold_keys("obj1", [_T0 + 3.0, _T0 + 10.5])
    out = build_trajectory_features(events, keys)

    for row, k in ((0, 3), (1, 4)):
        expected = _reference_stats(times[:k], scores[:k])
        for stat, value in expected.items():
            got = out.loc[row, _col("fink/rf_ia", stat)]
            assert got == pytest.approx(value, abs=1e-12), (row, stat)

    # Spot-check closed forms for the 3-event prefix.
    assert out.loc[0, _col("fink/rf_ia", "slope")] == pytest.approx(0.1 / (14.0 / 3.0), abs=1e-12)
    assert out.loc[0, _col("fink/rf_ia", "delta")] == pytest.approx(0.1)
    assert out.loc[0, _col("fink/rf_ia", "volatility")] == pytest.approx(0.15)
    assert out.loc[0, _col("fink/rf_ia", "std")] == pytest.approx(np.sqrt(0.02 / 3.0), abs=1e-12)

    # Single available expert → cross aggregates degenerate correctly.
    assert out.loc[0, "traj_x__n_experts"] == 1.0
    assert out.loc[0, "traj_x__mean_last"] == pytest.approx(0.2)
    assert np.isnan(out.loc[0, "traj_x__disagreement_last"])


def test_single_event_prefix() -> None:
    events = pd.DataFrame(_rf_events("obj1", [_T0], [0.4]))
    out = build_trajectory_features(events, _gold_keys("obj1", [_T0]))
    assert out.loc[0, _col("fink/rf_ia", "n")] == 1.0
    assert out.loc[0, _col("fink/rf_ia", "last")] == pytest.approx(0.4)
    assert out.loc[0, _col("fink/rf_ia", "std")] == 0.0  # ddof=0 convention
    assert out.loc[0, _col("fink/rf_ia", "delta")] == 0.0
    assert out.loc[0, _col("fink/rf_ia", "ewm")] == pytest.approx(0.4)
    assert np.isnan(out.loc[0, _col("fink/rf_ia", "slope")])
    assert np.isnan(out.loc[0, _col("fink/rf_ia", "volatility")])


# ---------------------------------------------------------------------------
# (b) Strict as-of: future / non-exact events never affect the output
# ---------------------------------------------------------------------------

def test_future_events_never_leak() -> None:
    past = _rf_events("obj1", [_T0, _T0 + 2.0], [0.2, 0.4])
    keys = _gold_keys("obj1", [_T0 + 2.0, _T0 + 5.0])
    baseline = build_trajectory_features(pd.DataFrame(past), keys)

    future = _rf_events("obj1", [_T0 + 5.1, _T0 + 30.0], [0.999, 0.001])
    with_future = build_trajectory_features(pd.DataFrame(past + future), keys)

    pd.testing.assert_frame_equal(baseline, with_future)


def test_event_exactly_at_alert_jd_is_included() -> None:
    events = _rf_events("obj1", [_T0, _T0 + 2.0], [0.2, 0.4])
    out = build_trajectory_features(pd.DataFrame(events), _gold_keys("obj1", [_T0 + 2.0]))
    # <= semantics (allow_exact_matches), same as select_events_asof.
    assert out.loc[0, _col("fink/rf_ia", "n")] == 2.0
    assert out.loc[0, _col("fink/rf_ia", "last")] == pytest.approx(0.4)


def test_non_exact_alert_events_are_ignored() -> None:
    past = _rf_events("obj1", [_T0], [0.2])
    keys = _gold_keys("obj1", [_T0 + 5.0])
    baseline = build_trajectory_features(pd.DataFrame(past), keys)

    tainted = past + [
        _ev("obj1", "fink/rf_ia", _T0 + 1.0, 0.95, field="rf_snia_vs_nonia",
            temporal_exactness="latest_object_unsafe"),
        _ev("obj1", "fink/rf_ia", _T0 + 1.5, 0.95, field="rf_snia_vs_nonia",
            temporal_exactness="rerun_exact"),
        _ev("obj1", "fink/rf_ia", _T0 + 2.0, 0.95, field="rf_snia_vs_nonia",
            availability=False),
    ]
    with_tainted = build_trajectory_features(pd.DataFrame(tainted), keys)
    pd.testing.assert_frame_equal(baseline, with_tainted)


# ---------------------------------------------------------------------------
# (c) payload_hash dedupe
# ---------------------------------------------------------------------------

def test_payload_hash_dedupe_does_not_double_count() -> None:
    events = _rf_events("obj1", [_T0, _T0 + 2.0, _T0 + 4.0], [0.2, 0.4, 0.6])
    keys = _gold_keys("obj1", [_T0 + 4.0])
    baseline = build_trajectory_features(pd.DataFrame(events), keys)

    # The same payload normalized into silver twice → identical rows
    # (same payload_hash).  Counts and stats must not change.
    doubled = build_trajectory_features(pd.DataFrame(events + events), keys)
    pd.testing.assert_frame_equal(baseline, doubled)
    assert doubled.loc[0, _col("fink/rf_ia", "n")] == 3.0

    # A re-fetch of the same alerts under a different payload_hash must also
    # collapse to one row per (object, alert, event_time).
    refetched = [dict(e, payload_hash="hash1") for e in events]
    deduped = build_trajectory_features(pd.DataFrame(events + refetched), keys)
    assert deduped.loc[0, _col("fink/rf_ia", "n")] == 3.0


# ---------------------------------------------------------------------------
# (d) Empty / missing expert → NaN columns
# ---------------------------------------------------------------------------

def test_missing_expert_gives_nan_columns() -> None:
    events = pd.DataFrame(_rf_events("obj1", [_T0], [0.4]))
    out = build_trajectory_features(events, _gold_keys("obj1", [_T0 + 1.0]))
    for expert in TRAJ_EXPERTS:
        if expert == "fink/rf_ia":
            continue
        for stat in TRAJ_STATS:
            assert np.isnan(out.loc[0, _col(expert, stat)]), (expert, stat)
    assert out.loc[0, "traj_x__n_experts"] == 1.0


def test_empty_events_frame_all_nan() -> None:
    empty = pd.DataFrame(columns=[
        "object_id", "expert_key", "field", "class_name", "event_time_jd",
        "alert_id", "canonical_projection", "temporal_exactness",
        "payload_hash", "availability",
    ])
    out = build_trajectory_features(empty, _gold_keys("obj1", [_T0, _T0 + 1.0]))
    assert list(out.columns[-66:]) == TRAJ_FEATURE_NAMES
    assert (out["traj_x__n_experts"] == 0.0).all()
    per_expert = [c for c in TRAJ_FEATURE_NAMES if c.startswith("traj__")]
    assert out[per_expert].isna().all().all()
    assert out[["traj_x__mean_last", "traj_x__max_last", "traj_x__mean_slope",
                "traj_x__disagreement_last", "traj_x__disagreement_trend"]].isna().all().all()


def test_object_without_events_is_nan_even_when_others_have_events() -> None:
    events = pd.DataFrame(_rf_events("obj1", [_T0], [0.4]))
    keys = pd.concat([
        _gold_keys("obj1", [_T0 + 1.0]),
        _gold_keys("obj2", [_T0 + 1.0]),
    ], ignore_index=True)
    out = build_trajectory_features(events, keys)
    assert out.loc[0, _col("fink/rf_ia", "n")] == 1.0
    assert np.isnan(out.loc[1, _col("fink/rf_ia", "n")])
    assert out.loc[1, "traj_x__n_experts"] == 0.0


# ---------------------------------------------------------------------------
# Projector parity — vectorized per-event p_snia equals the EXISTING projectors
# ---------------------------------------------------------------------------

def _last_p_snia(events: list[dict], expert: str, alert_jd: float) -> float:
    out = build_trajectory_features(pd.DataFrame(events), _gold_keys("obj1", [alert_jd]))
    return float(out.loc[0, _col(expert, "last")])


def _projector_p_snia(expert: str, events: list[dict]) -> float:
    projected = _dispatch_projector(expert, events)
    assert projected.get("p_snia") is not None, projected
    return float(projected["p_snia"])


def test_parity_fink_snn() -> None:
    events = [
        _ev("obj1", "fink/snn", _T0, 0.7, field="snn_snia_vs_nonia", alert_id=1.0),
        _ev("obj1", "fink/snn", _T0, 0.8, field="snn_sn_vs_all", alert_id=1.0),
    ]
    assert _last_p_snia(events, "fink/snn", _T0) == pytest.approx(
        _projector_p_snia("fink/snn", events), abs=1e-12)
    assert _last_p_snia(events, "fink/snn", _T0) == pytest.approx(0.56)


def test_parity_fink_rf_ia() -> None:
    events = [_ev("obj1", "fink/rf_ia", _T0, 0.3, field="rf_snia_vs_nonia")]
    assert _last_p_snia(events, "fink/rf_ia", _T0) == pytest.approx(
        _projector_p_snia("fink/rf_ia", events), abs=1e-12)


def test_parity_fink_lsst_snn() -> None:
    events = [_ev("obj1", "fink_lsst/snn", _T0, 0.8, field="clf_snnSnVsOthers_score")]
    assert _last_p_snia(events, "fink_lsst/snn", _T0) == pytest.approx(
        _projector_p_snia("fink_lsst/snn", events), abs=1e-12)
    assert _last_p_snia(events, "fink_lsst/snn", _T0) == pytest.approx(0.4)


def test_parity_fink_lsst_cats() -> None:
    for class_name, score in (("11", 0.9), ("21", 0.99), ("22", 0.5), ("41", 0.8)):
        events = [_ev("obj1", "fink_lsst/cats", _T0, score,
                      field="clf_cats_class", class_name=class_name)]
        assert _last_p_snia(events, "fink_lsst/cats", _T0) == pytest.approx(
            _projector_p_snia("fink_lsst/cats", events), abs=1e-12), class_name


def test_parity_fink_lsst_early_snia() -> None:
    events = [_ev("obj1", "fink_lsst/early_snia", _T0, 0.6, field="clf_earlySNIa_score")]
    assert _last_p_snia(events, "fink_lsst/early_snia", _T0) == pytest.approx(
        _projector_p_snia("fink_lsst/early_snia", events), abs=1e-12)

    # Sentinel -1 means "not triggered" — no event, all stats NaN.
    sentinel = [_ev("obj1", "fink_lsst/early_snia", _T0, -1.0, field="clf_earlySNIa_score")]
    out = build_trajectory_features(pd.DataFrame(sentinel), _gold_keys("obj1", [_T0]))
    assert np.isnan(out.loc[0, _col("fink_lsst/early_snia", "n")])


def test_parity_pittgoogle_supernnova_lsst() -> None:
    events = [_ev("obj1", "pittgoogle/supernnova_lsst", _T0, 0.55,
                  field="supernnova_prob_class0")]
    assert _last_p_snia(events, "pittgoogle/supernnova_lsst", _T0) == pytest.approx(
        _projector_p_snia("pittgoogle/supernnova_lsst", events), abs=1e-12)


# ---------------------------------------------------------------------------
# Cross-expert aggregates
# ---------------------------------------------------------------------------

def test_cross_expert_aggregates() -> None:
    events = (
        _rf_events("obj1", [_T0, _T0 + 1.0, _T0 + 2.0], [0.1, 0.2, 0.3])
        + [_ev("obj1", "fink_lsst/early_snia", t, s, field="clf_earlySNIa_score")
           for t, s in zip([_T0, _T0 + 1.0, _T0 + 2.0], [0.9, 0.8, 0.7])]
    )
    keys = _gold_keys("obj1", [_T0, _T0 + 1.0, _T0 + 2.0])
    out = build_trajectory_features(pd.DataFrame(events), keys)

    # Epoch 3: rf last=0.3, early last=0.7
    assert out.loc[2, "traj_x__n_experts"] == 2.0
    assert out.loc[2, "traj_x__mean_last"] == pytest.approx(0.5)
    assert out.loc[2, "traj_x__max_last"] == pytest.approx(0.7)
    assert out.loc[2, "traj_x__mean_slope"] == pytest.approx((0.1 + (-0.1)) / 2.0, abs=1e-12)
    assert out.loc[2, "traj_x__disagreement_last"] == pytest.approx(0.2)  # ddof=0

    # Disagreement series over alert_jd: 0.4, 0.3, 0.2 → slope -0.1/day.
    assert np.isnan(out.loc[0, "traj_x__disagreement_trend"])  # 1 point
    assert out.loc[1, "traj_x__disagreement_trend"] == pytest.approx(-0.1, abs=1e-12)
    assert out.loc[2, "traj_x__disagreement_trend"] == pytest.approx(-0.1, abs=1e-12)


# ---------------------------------------------------------------------------
# Row-order preservation
# ---------------------------------------------------------------------------

def test_gold_keys_row_order_preserved() -> None:
    events = pd.DataFrame(_rf_events("obj1", [_T0], [0.4]))
    keys = pd.DataFrame({
        "object_id": ["obj2", "obj1", "obj1"],
        "n_det": [1.0, 2.0, 1.0],
        "alert_jd": [_T0 + 9.0, _T0 + 5.0, _T0 + 1.0],
    })
    out = build_trajectory_features(events, keys)
    pd.testing.assert_frame_equal(out[["object_id", "n_det", "alert_jd"]],
                                  keys.reset_index(drop=True))


# ---------------------------------------------------------------------------
# (e) Golden check against the real gold table (skipped without data/)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (_GOLD_PATH.exists() and _SILVER_PATH.exists()),
    reason="requires local data/gold and data/silver parquet files",
)
def test_golden_fink_snn_last_matches_gold_projection() -> None:
    """traj__fink__snn__last at (object_id, n_det) must equal the gold table's
    proj__fink__snn__p_snia: both select the latest exact_alert event at or
    before alert_jd and apply the same fink/snn projection."""
    gold = pd.read_parquet(
        _GOLD_PATH,
        columns=["object_id", "n_det", "alert_jd", "proj__fink__snn__p_snia"],
    )
    have = gold[gold["proj__fink__snn__p_snia"].notna()]
    if have.empty:
        pytest.skip("gold table has no fink/snn projections")
    objects = sorted(have["object_id"].astype(str).unique())[:3]

    events = pd.read_parquet(
        _SILVER_PATH,
        filters=[("expert_key", "==", "fink/snn"), ("object_id", "in", objects)],
    )
    keys = (
        gold[gold["object_id"].astype(str).isin(objects)]
        [["object_id", "n_det", "alert_jd"]]
        .reset_index(drop=True)
    )
    out = build_trajectory_features(events, keys)

    merged = out.merge(
        have[["object_id", "n_det", "proj__fink__snn__p_snia"]],
        on=["object_id", "n_det"],
        how="inner",
    )
    assert len(merged) > 0, "no comparable rows for the 3 golden objects"
    diff = (merged["traj__fink__snn__last"] - merged["proj__fink__snn__p_snia"]).abs()
    assert diff.notna().all(), "traj last missing where gold has a projection"
    assert float(diff.max()) < 1e-6, f"max |traj_last - gold_proj| = {diff.max()}"
