from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from debass_meta.models.expert_trust import train_expert_trust_suite
from debass_meta.models.splitters import GroupSplit


def _toy_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    snapshots = []
    helpfulness = []
    for idx in range(12):
        object_id = f"obj_{idx}"
        label = "snia" if idx % 2 == 0 else "other"
        for n_det in (1, 2):
            base = {
                "object_id": object_id,
                "n_det": n_det,
                "alert_jd": 2460000.0 + n_det,
                "target_class": label,
                "target_follow_proxy": int(label == "snia"),
                "label_source": "external",
                "label_quality": "strong",
                "mag_first": 19.0 + idx * 0.01,
                "mag_last": 18.5 + idx * 0.01,
                "n_det_g": 1.0,
                "n_det_r": 1.0,
                "avail__fink__snn": 1.0,
                "exact__fink__snn": 1.0,
                "event_count__fink__snn": 2.0,
                "source_event_time_jd__fink__snn": 2460000.0 + n_det,
                "proj__fink__snn__p_snia": 0.8 if label == "snia" else 0.2,
                "proj__fink__snn__p_nonIa_snlike": 0.1,
                "proj__fink__snn__p_other": 0.1 if label == "snia" else 0.7,
                "proj__fink__snn__top1_prob": 0.8,
                "proj__fink__snn__margin": 0.6,
                "prediction_type__fink__snn": "class_correctness",
                "mapped_pred_class__fink__snn": "snia" if idx % 4 != 1 else "other",
                "temporal_exactness__fink__snn": "exact_alert",
            }
            snapshots.append(base)
            helpfulness.append(
                {
                    **base,
                    "expert_key": "fink/snn",
                    "available": True,
                    "temporal_exactness": "exact_alert",
                    "mapped_pred_class": base["mapped_pred_class__fink__snn"],
                    "mapped_p_true_class": 0.8,
                    "is_topclass_correct": int(base["mapped_pred_class__fink__snn"] == label),
                    "is_helpful_for_follow_proxy": int(label == "snia"),
                }
            )
    return pd.DataFrame(snapshots), pd.DataFrame(helpfulness)


def test_train_expert_trust_suite_writes_oof_trust_for_train_rows(tmp_path: Path) -> None:
    snapshots, helpfulness = _toy_rows()
    split = GroupSplit(
        train_ids={f"obj_{i}" for i in range(8)},
        cal_ids={f"obj_{i}" for i in range(8, 10)},
        test_ids={f"obj_{i}" for i in range(10, 12)},
    )

    out_snapshots, metrics, metadata = train_expert_trust_suite(
        helpfulness_df=helpfulness,
        snapshot_df=snapshots,
        split=split,
        models_dir=tmp_path / "models",
        output_snapshot_path=tmp_path / "object_epoch_snapshots_trust.parquet",
        n_splits=4,
    )

    assert "fink/snn" in metrics
    assert "fink/snn" in metadata["experts"]
    assert out_snapshots["q__fink__snn"].notna().all()
    train_sources = out_snapshots[out_snapshots["object_id"].isin(split.train_ids)]["trust_source__fink__snn"].unique().tolist()
    assert "oof" in train_sources
