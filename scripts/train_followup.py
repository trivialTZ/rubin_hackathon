"""Train the trust-aware follow-up proxy head."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from debass.models.followup import train_followup_model
from debass.models.splitters import GroupSplit


def _load_split(models_dir: Path) -> GroupSplit:
    with open(models_dir / "metadata.json") as fh:
        metadata = json.load(fh)
    return GroupSplit(
        train_ids=set(metadata["train_ids"]),
        cal_ids=set(metadata["cal_ids"]),
        test_ids=set(metadata["test_ids"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train follow-up proxy model")
    parser.add_argument("--snapshots", default="data/gold/object_epoch_snapshots_trust.parquet")
    parser.add_argument("--trust-models-dir", default="models/trust")
    parser.add_argument("--model-dir", default="models/followup")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    snapshot_df = pd.read_parquet(args.snapshots)
    split = _load_split(Path(args.trust_models_dir))
    _, metrics = train_followup_model(
        snapshot_df,
        split=split,
        model_dir=Path(args.model_dir),
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
    )

    reports_dir = Path("reports/metrics")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "followup_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Wrote follow-up metrics → {reports_dir / 'followup_metrics.json'}")


if __name__ == "__main__":
    main()
