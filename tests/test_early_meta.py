from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from debass_meta.models.early_meta import EarlyMetaClassifier


class DummyCalibrator:
    def transform(self, probs: np.ndarray) -> np.ndarray:
        calibrated = np.zeros_like(probs)
        calibrated[:, 0] = 0.8
        calibrated[:, 1] = 0.15
        calibrated[:, 2] = 0.05
        return calibrated


def _toy_epoch_table() -> pd.DataFrame:
    rows = []
    labels = ["snia", "nonIa_snlike", "other"]
    for obj_idx, label in enumerate(labels * 3):
        for n_det in (1, 2):
            rows.append(
                {
                    "object_id": f"obj_{obj_idx}",
                    "n_det": n_det,
                    "target_label": label,
                    "mag_first": 18.0 + obj_idx,
                    "mean_rb": 0.9 - 0.01 * obj_idx,
                }
            )
    return pd.DataFrame(rows)


def test_early_meta_tracks_object_balanced_weights() -> None:
    df = _toy_epoch_table()
    clf = EarlyMetaClassifier(n_estimators=10)
    clf.fit(df)

    summary = clf.training_weight_summary
    assert summary is not None
    assert summary["n_objects"] == 9
    assert summary["class_object_counts"] == {"nonIa_snlike": 3, "other": 3, "snia": 3}
    assert 0.9 <= summary["mean_weight"] <= 1.1


def test_early_meta_applies_calibrator_at_inference() -> None:
    df = _toy_epoch_table()
    clf = EarlyMetaClassifier(n_estimators=10)
    clf.fit(df)

    raw = clf.predict_proba(df.iloc[:3], calibrated=False)
    clf.set_calibrator(DummyCalibrator())
    calibrated = clf.predict_proba(df.iloc[:3], calibrated=True)

    assert raw.shape == (3, 3)
    assert calibrated.shape == (3, 3)
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert np.allclose(calibrated[:, 0], 0.8)
    assert not np.allclose(raw, calibrated)
