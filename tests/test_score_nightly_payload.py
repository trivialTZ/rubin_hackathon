from __future__ import annotations

import sys
from pathlib import Path

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
