from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.score_nightly import build_score_payload


def test_score_payload_contains_expert_confidence_block() -> None:
    row = pd.Series(
        {
            "object_id": "ZTF1",
            "n_det": 4,
            "alert_jd": 2460004.0,
            "avail__fink__snn": 1.0,
            "prediction_type__fink__snn": "class_correctness",
            "temporal_exactness__fink__snn": "exact_alert",
            "mapped_pred_class__fink__snn": "snia",
            "proj__fink__snn__p_snia": 0.7,
            "proj__fink__snn__p_nonIa_snlike": 0.2,
            "proj__fink__snn__p_other": 0.1,
            "q__fink__snn": 0.8,
            "avail__lasair__sherlock": 1.0,
            "prediction_type__lasair__sherlock": "context_only",
            "context_tag__lasair__sherlock": "SN",
            "temporal_exactness__lasair__sherlock": "static_safe",
        }
    )

    payload = build_score_payload(row, trust_models={}, followup_model=None)

    assert payload["object_id"] == "ZTF1"
    assert "expert_confidence" in payload
    assert payload["expert_confidence"]["fink/snn"]["trust"] == 0.8
    assert payload["expert_confidence"]["lasair/sherlock"]["prediction_type"] == "context_only"
    assert "ensemble" in payload


def test_score_payload_cleans_nan_scalars_for_unavailable_experts() -> None:
    row = pd.Series(
        {
            "object_id": "ZTF2",
            "n_det": 3,
            "alert_jd": 2460003.0,
            "avail__parsnip": 0.0,
            "prediction_type__parsnip": np.nan,
            "temporal_exactness__parsnip": np.nan,
        }
    )

    payload = build_score_payload(row, trust_models={}, followup_model=None)
    parsnip = payload["expert_confidence"]["parsnip"]

    assert parsnip["available"] is False
    assert parsnip["trust"] is None
    assert parsnip["prediction_type"] is None
    assert parsnip["exactness"] is None
