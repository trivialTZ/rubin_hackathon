"""n_det-sliced trust table in scripts/eval_fusion_v8.py (fusion_v10 fix #2).

The PI contract is per-expert trust at 3-5 detections; Table 5b reports
calibrated trust AUC/ECE at n_det buckets 3/4/5 and pooled 3-5 on the locked
test rows.  Tested against a synthetic frame with a known-informative q.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load_eval_module():
    spec = importlib.util.spec_from_file_location(
        "eval_fusion_v8_script", _ROOT / "scripts" / "eval_fusion_v8.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def eval_mod():
    return _load_eval_module()


def _make_frame(n_objects: int = 80, seed: int = 5) -> pd.DataFrame:
    """Objects × n_det 3..6 with two experts:

    * fink__snn (regular): q is informative about topclass correctness
      (q=0.9 when mapped_pred_class == target_class, else 0.1);
    * fink__slsn (SN-filter): q is informative about is_sn.
    """
    rng = np.random.default_rng(seed)
    rows = []
    classes = ("snia", "nonIa_snlike", "other")
    for i in range(n_objects):
        oid = f"T{i:05d}"
        cls = classes[i % 3]
        for n_det in (3, 4, 5, 6):
            correct = rng.random() < 0.5
            pred = cls if correct else classes[(classes.index(cls) + 1) % 3]
            absent = rng.random() < 0.15
            rows.append({
                "object_id": oid,
                "n_det": n_det,
                "survey": "ZTF",
                "target_class": cls,
                "label_quality": "spectroscopic",
                "q__fink__snn": np.nan if absent else (0.9 if correct else 0.1),
                "mapped_pred_class__fink__snn": None if absent else pred,
                "prediction_type__fink__snn": "class_correctness",
                "q__fink__slsn": 0.85 if cls != "other" else 0.15,
                "mapped_pred_class__fink__slsn": "nonIa_snlike",
            })
    return pd.DataFrame(rows)


def test_trust_ndet_rows_buckets_and_auc(eval_mod):
    frame = _make_frame()
    test_ids = set(frame["object_id"].astype(str))
    rows = eval_mod.trust_ndet_rows(frame, test_ids)
    assert rows, "expected per-expert rows"

    by_key = {(r["expert"], r.get("n_det")): r for r in rows if "n_det" in r}
    # every expert × bucket combination is reported
    for expert in ("fink/snn", "fink/slsn"):
        for bucket in ("3", "4", "5", "3-5"):
            assert (expert, bucket) in by_key, f"missing {expert} @ {bucket}"

    # n_det=6 rows are excluded; pooled 3-5 is the sum of the three buckets
    snn = {b: by_key[("fink/snn", b)] for b in ("3", "4", "5", "3-5")}
    assert snn["3-5"]["n_rows"] == sum(snn[b]["n_rows"] for b in ("3", "4", "5"))
    # absent-expert rows (NaN q) are excluded from the claim counts
    assert snn["3-5"]["n_rows"] < 3 * 80

    # informative q -> near-perfect AUC; ECE small for a well-placed q
    assert snn["3-5"]["auc"] is not None and snn["3-5"]["auc"] > 0.95
    for b in ("3", "4", "5"):
        assert snn[b]["auc"] is not None and snn[b]["auc"] > 0.95
        assert snn[b]["ece15"] is not None
    assert snn["3-5"]["target"] == "is_topclass_correct"

    # SN-filter expert scored against is_sn, not topclass correctness
    slsn = by_key[("fink/slsn", "3-5")]
    assert slsn["target"] == "is_sn"
    assert slsn["auc"] == pytest.approx(1.0)  # q separates SN from other exactly


def test_trust_ndet_rows_respects_protocol_filters(eval_mod):
    frame = _make_frame(n_objects=40, seed=8)
    all_ids = set(frame["object_id"].astype(str))

    # locked-test filter: objects outside test_ids never contribute
    half = set(sorted(all_ids)[:20])
    rows_half = eval_mod.trust_ndet_rows(frame, half)
    rows_all = eval_mod.trust_ndet_rows(frame, all_ids)
    n_half = next(r["n_rows"] for r in rows_half
                  if r["expert"] == "fink/slsn" and r.get("n_det") == "3-5")
    n_all = next(r["n_rows"] for r in rows_all
                 if r["expert"] == "fink/slsn" and r.get("n_det") == "3-5")
    assert n_half == n_all / 2

    # spec-only labels: weak rows are excluded
    weak = frame.copy()
    weak["label_quality"] = "weak"
    assert eval_mod.trust_ndet_rows(weak, all_ids) == []

    # context_only expert claims never count as topclass correctness rows
    ctx = frame.copy()
    ctx["prediction_type__fink__snn"] = "context_only"
    rows_ctx = eval_mod.trust_ndet_rows(ctx, all_ids)
    n_ctx = next(r["n_rows"] for r in rows_ctx
                 if r["expert"] == "fink/snn" and r.get("n_det") == "3-5")
    assert n_ctx == 0


def test_build_table_5_includes_ndet_section(eval_mod):
    frame = _make_frame(n_objects=30, seed=11)
    test_ids = set(frame["object_id"].astype(str))
    train_report = {"stage_a": {"metrics": {
        "fink/snn": {"raw_auc": 0.9, "cal_auc": 0.92, "brier": 0.1,
                     "ece": 0.02, "n_test": 100, "calibrator_kind": "isotonic",
                     "fallback_used": False},
    }}}
    payload, md = eval_mod.build_table_5(
        train_report, Path("/nonexistent.json"), Path("/nonexistent_dir"),
        frame=frame, test_ids=test_ids,
    )
    section = payload["ndet_sliced"]
    assert section["rows"], "n_det-sliced rows missing from Table 5 payload"
    buckets = {r.get("n_det") for r in section["rows"] if "n_det" in r}
    assert buckets == {"3", "4", "5", "3-5"}
    assert "Table 5b" in md and "3-5" in md

    # graceful degradation without a frame
    payload2, md2 = eval_mod.build_table_5(
        train_report, Path("/nonexistent.json"), Path("/nonexistent_dir"))
    assert payload2["ndet_sliced"]["rows"] == []
    assert payload2["ndet_sliced"]["reason_unavailable"]
    assert "UNAVAILABLE" in md2
