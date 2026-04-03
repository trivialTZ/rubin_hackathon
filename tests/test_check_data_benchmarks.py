from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.check_data_benchmarks import DEFAULT_BENCHMARKS, evaluate_run


def _build_run(root: Path, *, include_all_expert_blocks: bool = True) -> tuple[Path, Path]:
    bronze_dir = root / "bronze"
    silver_dir = root / "silver"
    gold_dir = root / "gold"
    truth_dir = root / "truth"
    trust_dir = root / "models/trust"
    followup_dir = root / "models/followup"
    reports_metrics = root / "reports/metrics"
    for path in [bronze_dir, silver_dir, gold_dir, truth_dir, trust_dir, followup_dir, reports_metrics]:
        path.mkdir(parents=True, exist_ok=True)

    labels_path = root.parent / "labels.csv"
    labels_path.write_text("object_id,label\nZTF1,snia\n")
    lc_dir = root.parent / "lightcurves"
    lc_dir.mkdir(exist_ok=True)
    (lc_dir / "ZTF1.json").write_text("[]")

    pd.DataFrame(
        [
            {"broker": "alerce", "object_id": "ZTF1", "availability": True, "fixture_used": False},
            {"broker": "fink", "object_id": "ZTF1", "availability": True, "fixture_used": False},
        ]
    ).to_parquet(bronze_dir / "part-000.parquet", index=False)

    pd.DataFrame(
        [
            {
                "broker": "fink",
                "expert_key": "fink/snn",
                "object_id": "ZTF1",
                "alert_id": 111,
                "n_det": 1,
                "event_time_jd": 2460001.0,
                "availability": True,
                "fixture_used": False,
                "temporal_exactness": "exact_alert",
            },
            {
                "broker": "fink",
                "expert_key": "fink/rf_ia",
                "object_id": "ZTF1",
                "alert_id": 111,
                "n_det": 1,
                "event_time_jd": 2460001.0,
                "availability": True,
                "fixture_used": False,
                "temporal_exactness": "exact_alert",
            },
        ]
    ).to_parquet(silver_dir / "broker_events.parquet", index=False)

    pd.DataFrame(
        [
            {
                "object_id": "ZTF1",
                "n_det": 1,
                "alert_jd": 2460001.0,
                "avail__fink__snn": 1.0,
                "avail__fink__rf_ia": 1.0,
                "avail__parsnip": 0.0,
                "avail__supernnova": 0.0,
                "avail__alerce__lc_classifier_transient": 0.0,
                "avail__alerce__stamp_classifier": 0.0,
                "avail__lasair__sherlock": 0.0,
            }
        ]
    ).to_parquet(gold_dir / "object_epoch_snapshots.parquet", index=False)

    pd.DataFrame(
        [
            {
                "object_id": "ZTF1",
                "final_class_raw": "snia",
                "final_class_ternary": "snia",
                "follow_proxy": 1,
                "label_source": "external_curated",
                "label_quality": "strong",
            }
        ]
    ).to_parquet(truth_dir / "object_truth.parquet", index=False)

    (trust_dir / "metadata.json").write_text(
        json.dumps({"experts": ["fink/snn", "fink/rf_ia"], "contains_weak_labels": False})
    )
    (followup_dir / "metadata.json").write_text(json.dumps({"feature_cols": ["q__fink__snn", "q__fink__rf_ia"]}))

    (reports_metrics / "expert_trust_metrics.json").write_text(
        json.dumps(
            {
                "fink/snn": {"n_rows": 4, "roc_auc": 0.9, "brier": 0.1},
                "fink/rf_ia": {"n_rows": 4, "roc_auc": 0.8, "brier": 0.2},
            }
        )
    )
    (reports_metrics / "followup_metrics.json").write_text(
        json.dumps({"n_rows": 4, "roc_auc": 0.85, "brier": 0.12})
    )

    expert_confidence = {
        "fink/snn": {"available": True, "trust": 0.8, "prediction_type": "class_correctness", "exactness": "exact_alert"},
        "fink/rf_ia": {"available": True, "trust": 0.7, "prediction_type": "class_correctness", "exactness": "exact_alert"},
        "parsnip": {"available": False, "trust": None, "prediction_type": None, "exactness": None},
        "supernnova": {"available": False, "trust": None, "prediction_type": None, "exactness": None},
        "alerce_lc": {"available": False, "trust": None, "prediction_type": None, "exactness": None},
        "alerce/lc_classifier_transient": {"available": False, "trust": None, "prediction_type": None, "exactness": None},
        "alerce/stamp_classifier": {"available": False, "trust": None, "prediction_type": None, "exactness": None},
        "lasair/sherlock": {"available": False, "trust": None, "prediction_type": None, "exactness": None},
    }
    if not include_all_expert_blocks:
        expert_confidence.pop("parsnip")

    (root / "reports").mkdir(exist_ok=True)
    (root / "reports/scores_ndet4.jsonl").write_text(
        json.dumps(
            {
                "object_id": "ZTF1",
                "n_det": 1,
                "alert_jd": 2460001.0,
                "expert_confidence": expert_confidence,
                "ensemble": {"p_follow_proxy": 0.4, "recommended": False},
            }
        )
        + "\n"
    )

    return labels_path, lc_dir


def test_evaluate_run_passes_for_valid_contract_run(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    labels_path, lc_dir = _build_run(run_root)

    result = evaluate_run(
        root=run_root,
        benchmarks=deepcopy(DEFAULT_BENCHMARKS),
        labels_path=labels_path,
        lightcurve_dir=lc_dir,
    )

    assert result["passed"] is True
    assert result["failures"] == []
    assert result["score_rows"] == 1


def test_evaluate_run_fails_when_score_payload_drops_phase1_expert_block(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    labels_path, lc_dir = _build_run(run_root, include_all_expert_blocks=False)

    result = evaluate_run(
        root=run_root,
        benchmarks=deepcopy(DEFAULT_BENCHMARKS),
        labels_path=labels_path,
        lightcurve_dir=lc_dir,
    )

    assert result["passed"] is False
    assert any("missing expert blocks" in failure for failure in result["failures"])
