"""Build expert-helpfulness rows for trust-head training."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from debass.features.lightcurve import FEATURE_NAMES
from debass.projectors import PHASE1_EXPERT_KEYS, sanitize_expert_key


def build_expert_helpfulness(snapshot_path: Path, output_path: Path) -> Path:
    import pandas as pd

    snapshot_df = pd.read_parquet(snapshot_path)
    rows: list[dict] = []
    generic_feature_cols = [column for column in ["alert_jd", *FEATURE_NAMES] if column in snapshot_df.columns]

    for _, row in snapshot_df.iterrows():
        for expert_key in PHASE1_EXPERT_KEYS:
            san = sanitize_expert_key(expert_key)
            if not bool(row.get(f"avail__{san}", 0)):
                continue
            record = {
                "object_id": row["object_id"],
                "n_det": int(row["n_det"]),
                "alert_jd": float(row["alert_jd"]),
                "expert_key": expert_key,
                "available": True,
                "temporal_exactness": row.get(f"temporal_exactness__{san}"),
                "mapped_pred_class": row.get(f"mapped_pred_class__{san}"),
                "target_class": row.get("target_class"),
                "target_follow_proxy": row.get("target_follow_proxy"),
                "prediction_type": row.get(f"prediction_type__{san}"),
                "reason": row.get(f"reason__{san}"),
            }
            for column in generic_feature_cols:
                record[column] = row.get(column)
            for column in snapshot_df.columns:
                if column.startswith(f"proj__{san}__") or column in {f"avail__{san}", f"exact__{san}", f"event_count__{san}", f"source_event_time_jd__{san}"}:
                    record[column] = row.get(column)

            target_class = row.get("target_class")
            mapped_pred_class = row.get(f"mapped_pred_class__{san}")
            record["is_topclass_correct"] = (
                int(mapped_pred_class == target_class)
                if mapped_pred_class is not None and target_class is not None and row.get(f"prediction_type__{san}") != "context_only"
                else None
            )

            p_true_class = None
            if target_class is not None:
                ternary_col = f"proj__{san}__p_{target_class}"
                if ternary_col in record and record.get(ternary_col) is not None:
                    p_true_class = record.get(ternary_col)
                elif target_class == "snia" and record.get(f"proj__{san}__p_snia_scalar") is not None:
                    p_true_class = record.get(f"proj__{san}__p_snia_scalar")
                elif target_class != "snia" and record.get(f"proj__{san}__p_non_snia_scalar") is not None:
                    p_true_class = record.get(f"proj__{san}__p_non_snia_scalar")
            record["mapped_p_true_class"] = p_true_class

            follow_pred = None
            if record.get(f"proj__{san}__p_snia") is not None:
                follow_pred = int(record[f"proj__{san}__p_snia"] >= 0.5)
            elif record.get(f"proj__{san}__p_snia_scalar") is not None:
                follow_pred = int(record[f"proj__{san}__p_snia_scalar"] >= 0.5)
            elif mapped_pred_class is not None:
                follow_pred = int(mapped_pred_class == "snia")

            target_follow_proxy = row.get("target_follow_proxy")
            record["is_helpful_for_follow_proxy"] = (
                int(follow_pred == int(target_follow_proxy))
                if follow_pred is not None and target_follow_proxy is not None
                else None
            )
            rows.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build expert_helpfulness.parquet")
    parser.add_argument("--snapshots", default="data/gold/object_epoch_snapshots.parquet")
    parser.add_argument("--output", default="data/gold/expert_helpfulness.parquet")
    args = parser.parse_args()

    path = build_expert_helpfulness(Path(args.snapshots), Path(args.output))
    print(f"Wrote expert helpfulness → {path}")


if __name__ == "__main__":
    main()
