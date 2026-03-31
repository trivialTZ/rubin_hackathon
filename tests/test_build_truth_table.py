from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.build_truth_table import build_truth_table


def test_build_truth_table_marks_labels_csv_as_weak(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.csv"
    labels_path.write_text("object_id,label\nZTF1,snia\nZTF2,other\n")

    out_path = build_truth_table(
        input_path=None,
        labels_path=labels_path,
        output_path=tmp_path / "object_truth.parquet",
    )
    truth_df = pd.read_parquet(out_path).sort_values("object_id").reset_index(drop=True)

    assert truth_df["label_source"].tolist() == ["alerce_self_label", "alerce_self_label"]
    assert truth_df["label_quality"].tolist() == ["weak", "weak"]
    assert truth_df["final_class_ternary"].tolist() == ["snia", "other"]
