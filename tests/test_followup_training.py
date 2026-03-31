from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from debass.models.followup import train_followup_model
from debass.models.splitters import GroupSplit


def test_followup_training_consumes_trust_features(tmp_path: Path) -> None:
    rows = []
    for idx in range(20):
        rows.append(
            {
                "object_id": f"obj_{idx}",
                "n_det": 2,
                "alert_jd": 2460002.0,
                "target_follow_proxy": idx % 2,
                "n_det_g": 1.0,
                "n_det_r": 1.0,
                "mag_first": 19.5,
                "mag_last": 19.0,
                "q__fink__snn": 0.9 if idx % 2 else 0.1,
                "proj__fink__snn__p_snia": 0.9 if idx % 2 else 0.1,
                "mapped_pred_class__fink__snn": "snia" if idx % 2 else "other",
            }
        )
    df = pd.DataFrame(rows)
    split = GroupSplit(
        train_ids={f"obj_{i}" for i in range(12)},
        cal_ids={f"obj_{i}" for i in range(12, 16)},
        test_ids={f"obj_{i}" for i in range(16, 20)},
    )

    artifact, metrics = train_followup_model(
        df,
        split=split,
        model_dir=tmp_path / "followup",
        n_estimators=20,
        n_jobs=1,
    )

    assert any(column.startswith("q__") for column in artifact.feature_cols)
    assert metrics["n_rows"] == 4
