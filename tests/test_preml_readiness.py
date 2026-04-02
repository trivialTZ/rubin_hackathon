from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import scripts.check_preml_readiness as readiness_module


def test_evaluate_preml_readiness_surfaces_external_blockers(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    bronze_dir = run_root / "bronze"
    silver_dir = run_root / "silver"
    gold_dir = run_root / "gold"
    truth_dir = run_root / "truth"
    local_dir = silver_dir / "local_expert_outputs/parsnip"
    for path in [bronze_dir, silver_dir, gold_dir, truth_dir, local_dir]:
        path.mkdir(parents=True, exist_ok=True)

    labels_path = tmp_path / "labels.csv"
    labels_path.write_text("object_id,label\nZTF1,snia\n")
    lightcurve_dir = tmp_path / "lightcurves"
    lightcurve_dir.mkdir()
    (lightcurve_dir / "ZTF1.json").write_text("[]")

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
                "availability": True,
                "fixture_used": False,
                "temporal_exactness": "exact_alert",
                "alert_id": 111,
                "n_det": 1,
                "event_time_jd": 2460001.0,
            },
            {
                "broker": "fink",
                "expert_key": "fink/rf_ia",
                "object_id": "ZTF1",
                "availability": True,
                "fixture_used": False,
                "temporal_exactness": "exact_alert",
                "alert_id": 111,
                "n_det": 1,
                "event_time_jd": 2460001.0,
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
                "label_source": "alerce_self_label",
                "label_quality": "weak",
            }
        ]
    ).to_parquet(truth_dir / "object_truth.parquet", index=False)

    pd.DataFrame(
        [
            {
                "expert": "parsnip",
                "object_id": "ZTF1",
                "n_det": 1,
                "alert_mjd": 60000.0,
                "class_probabilities": "{}",
                "model_version": "stub",
                "available": False,
            }
        ]
    ).to_parquet(local_dir / "part-latest.parquet", index=False)

    class FakeLasairAdapter:
        def probe(self) -> dict[str, object]:
            return {"broker": "lasair", "status": "ok"}

        def fetch_object(self, object_id: str):
            return type(
                "FakeBrokerOutput",
                (),
                {
                    "availability": False,
                    "fixture_used": False,
                    "fields": [],
                    "source_endpoint": "https://lasair.lsst.ac.uk/api/object/",
                    "request_params": {"objectId": object_id},
                },
            )()

    monkeypatch.setattr(readiness_module, "LasairAdapter", FakeLasairAdapter)

    result = readiness_module.evaluate_preml_readiness(
        root=run_root,
        labels_path=labels_path,
        lightcurve_dir=lightcurve_dir,
    )

    assert result["ready_for_trust_training_now"] is False
    assert result["phase1_safe_snapshot_experts"] == ["fink/snn", "fink/rf_ia"]
    assert result["lasair_dataset_probe"]["status"] == "no_live_payload"
    assert any("no strong curated truth" in blocker for blocker in result["trust_training_blockers"])
    assert any("Lasair dataset probe returned no live Sherlock context" in blocker for blocker in result["full_phase1_blockers"])
