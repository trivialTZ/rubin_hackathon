from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.audit_data_readiness import audit_data_readiness


def test_audit_data_readiness_reports_live_vs_stub_state(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    bronze_dir = run_root / "bronze"
    silver_dir = run_root / "silver"
    gold_dir = run_root / "gold"
    truth_dir = run_root / "truth"
    local_dir = silver_dir / "local_expert_outputs/parsnip"
    trust_dir = run_root / "models/trust"
    followup_dir = run_root / "models/followup"
    for path in [bronze_dir, silver_dir, gold_dir, truth_dir, local_dir, trust_dir, followup_dir]:
        path.mkdir(parents=True, exist_ok=True)

    labels_path = tmp_path / "labels.csv"
    labels_path.write_text("object_id,label\nZTF1,snia\n")
    lc_dir = tmp_path / "lightcurves"
    lc_dir.mkdir()
    (lc_dir / "ZTF1.json").write_text("[]")

    bronze = pd.DataFrame(
        [
            {"broker": "alerce", "object_id": "ZTF1", "availability": True, "fixture_used": False},
            {"broker": "fink", "object_id": "ZTF1", "availability": True, "fixture_used": False},
            {"broker": "lasair", "object_id": "ZTF1", "availability": False, "fixture_used": True},
        ]
    )
    bronze.to_parquet(bronze_dir / "part-000.parquet", index=False)

    silver = pd.DataFrame(
        [
            {
                "broker": "fink",
                "expert_key": "fink/snn",
                "object_id": "ZTF1",
                "availability": True,
                "fixture_used": False,
                "temporal_exactness": "exact_alert",
            },
            {
                "broker": "lasair",
                "expert_key": "lasair/sherlock",
                "object_id": "ZTF1",
                "availability": False,
                "fixture_used": True,
                "temporal_exactness": "static_safe",
            },
        ]
    )
    silver.to_parquet(silver_dir / "broker_events.parquet", index=False)

    snapshots = pd.DataFrame(
        [
            {
                "object_id": "ZTF1",
                "n_det": 1,
                "alert_jd": 2460001.0,
                "avail__fink__snn": 1.0,
                "avail__fink__rf_ia": 0.0,
                "avail__parsnip": 0.0,
                "avail__lasair__sherlock": 0.0,
            }
        ]
    )
    snapshots.to_parquet(gold_dir / "object_epoch_snapshots.parquet", index=False)

    truth = pd.DataFrame(
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
    )
    truth.to_parquet(truth_dir / "object_truth.parquet", index=False)

    local_rows = pd.DataFrame(
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
    )
    local_rows.to_parquet(local_dir / "part-latest.parquet", index=False)

    (trust_dir / "metadata.json").write_text(json.dumps({"experts": ["fink/snn"], "contains_weak_labels": True}))
    (followup_dir / "metadata.json").write_text(json.dumps({"feature_cols": ["q__fink__snn"]}))

    report = audit_data_readiness(
        root=run_root,
        labels_path=labels_path,
        lightcurve_dir=lc_dir,
    )

    assert report["root"] == str(run_root.resolve())
    assert report["label_rows"] == 1
    assert report["lightcurve_files"] == 1
    assert report["bronze_live_available_rows"]["fink"] == 1
    assert report["bronze_fixture_rows"]["lasair"] == 1
    assert report["snapshot_available_true"]["avail__fink__snn"] == 1
    assert report["local_experts"]["parsnip"]["all_stub_versions"] is True
    assert "weak ALeRCE self-label truth present" in report["warnings"]
    assert "Lasair is not returning live bronze payloads; check LASAIR_TOKEN or fixture fallback" in report["warnings"]
