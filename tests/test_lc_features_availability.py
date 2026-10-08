"""lc_features_bv marks an epoch without class probabilities unavailable (fusion v13k)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from debass_meta.experts.local.lc_features import LcFeaturesExpert


class _Head:
    classes_ = ["nonIa_snlike", "other", "snia"]

    def predict_proba(self, X):
        return [[0.2, 0.1, 0.7] for _ in X]


def _expert() -> LcFeaturesExpert:
    e = LcFeaturesExpert(model_path=Path("/nonexistent/model.pkl"))
    e._head, e._available = _Head(), True
    return e


def _dets(n: int) -> list[dict]:
    return [{"mjd": 60000.0 + i, "band": "r", "magpsf": 19.0 - 0.1 * i, "sigmapsf": 0.05, "isdiffpos": "t"}
            for i in range(n)]


def test_below_four_detections_is_unavailable() -> None:
    out = _expert().predict_epoch("ZTF00test", _dets(3), epoch_jd=60010.0 + 2400000.5)
    assert out.available is False
    assert out.class_probabilities == {}


def test_no_usable_band_is_unavailable() -> None:
    dets = [{"mjd": 60000.0 + i, "isdiffpos": "t"} for i in range(6)]
    out = _expert().predict_epoch("ZTF00test", dets, epoch_jd=60010.0 + 2400000.5)
    assert out.available is False
