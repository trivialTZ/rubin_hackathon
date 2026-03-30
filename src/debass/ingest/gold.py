"""Gold layer: build object-by-epoch snapshot rows from silver data.

Each row = (object_id, epoch_index) with availability mask, features,
broker scores, and optional ground-truth label.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

_SILVER_DIR = Path("data/silver")
_GOLD_DIR = Path("data/gold")

# Brokers/experts whose scores are tracked in the availability mask
_SOURCES = ["alerce", "fink", "lasair", "supernnova", "parsnip", "ampel"]


def silver_to_gold(
    silver_dir: Path = _SILVER_DIR,
    gold_dir: Path = _GOLD_DIR,
    label_map: dict[str, str] | None = None,  # object_id -> ternary label
    label_source: str = "manual",
) -> Path:
    """Build epoch snapshot table from silver broker_outputs.

    Returns path of written gold parquet file.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError("pandas required") from e

    silver_path = silver_dir / "broker_outputs.parquet"
    if not silver_path.exists():
        raise FileNotFoundError(f"Silver file not found: {silver_path}")

    df = pd.read_parquet(silver_path)
    label_map = label_map or {}
    gold_rows: list[dict[str, Any]] = []

    for object_id, grp in df.groupby("object_id"):
        # Sort by query_time to assign epoch indices
        grp = grp.sort_values("query_time")
        # Group by (broker, approximate JD) — use query_time buckets as epochs
        # Simple approach: each unique query_time round to nearest minute = epoch
        grp["epoch_bucket"] = (grp["query_time"] // 60).astype(int)
        for epoch_idx, (bucket, epoch_grp) in enumerate(
            grp.groupby("epoch_bucket", sort=True)
        ):
            availability_mask = {
                src: bool((epoch_grp["broker"] == src).any()) for src in _SOURCES
            }
            # Collect broker scores: broker__field -> canonical_projection
            broker_scores: dict[str, Any] = {}
            for _, r in epoch_grp.iterrows():
                key = f"{r['broker']}__{r['field']}"
                broker_scores[key] = {
                    "raw_label_or_score": r["raw_label_or_score"],
                    "semantic_type": r["semantic_type"],
                    "canonical_projection": r["canonical_projection"],
                }

            gold_rows.append({
                "object_id": object_id,
                "epoch_index": epoch_idx,
                "epoch_jd": float(epoch_grp["query_time"].mean()),
                "survey": epoch_grp["survey"].iloc[0] if "survey" in epoch_grp else "ZTF",
                "availability_mask": availability_mask,
                "broker_scores": broker_scores,
                "target_label": label_map.get(str(object_id)),
                "label_source": label_source if str(object_id) in label_map else None,
            })

    gold_dir.mkdir(parents=True, exist_ok=True)
    out_path = gold_dir / "snapshots.parquet"
    pd.DataFrame(gold_rows).to_parquet(out_path, index=False)
    return out_path
