from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.validate_truth_table import validate_truth_frame


def test_validate_truth_frame_accepts_canonical_strong_truth() -> None:
    df = pd.DataFrame(
        [
            {
                "object_id": "ZTF1",
                "final_class_raw": "Ia",
                "final_class_ternary": "snia",
                "follow_proxy": 1,
                "label_source": "external_curated",
                "label_quality": "strong",
            }
        ]
    )

    result = validate_truth_frame(df, require_strong=True)

    assert result["passed"] is True
    assert result["errors"] == []


def test_validate_truth_frame_rejects_inconsistent_follow_proxy_and_duplicates() -> None:
    df = pd.DataFrame(
        [
            {
                "object_id": "ZTF1",
                "final_class_raw": "Ia",
                "final_class_ternary": "snia",
                "follow_proxy": 0,
                "label_source": "external_curated",
                "label_quality": "strong",
            },
            {
                "object_id": "ZTF1",
                "final_class_raw": "II",
                "final_class_ternary": "nonIa_snlike",
                "follow_proxy": 0,
                "label_source": "external_curated",
                "label_quality": "strong",
            },
        ]
    )

    result = validate_truth_frame(df, require_strong=True)

    assert result["passed"] is False
    assert any("duplicate object_id" in error for error in result["errors"])
    assert any("follow_proxy=0" in error for error in result["errors"])
