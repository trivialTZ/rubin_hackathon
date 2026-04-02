"""Train per-expert trust heads with grouped OOF stacking."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from debass_meta.models.expert_trust import train_expert_trust_suite
from debass_meta.models.splitters import group_train_cal_test_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Train expert trust heads")
    parser.add_argument("--snapshots", default="data/gold/object_epoch_snapshots.parquet")
    parser.add_argument("--helpfulness", default="data/gold/expert_helpfulness.parquet")
    parser.add_argument("--models-dir", default="models/trust")
    parser.add_argument("--output-snapshots", default="data/gold/object_epoch_snapshots_trust.parquet")
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--strict-labels", action="store_true")
    parser.add_argument("--allow-unsafe-alerce", action="store_true",
                        help="Include ALeRCE latest_object_unsafe snapshots in trust training")
    parser.add_argument("--exclude-unsafe-alerce", action="store_true",
                        help="Exclude ALeRCE latest_object_unsafe snapshots from trust training")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Number of threads for LightGBM (must match -pe omp slots)")
    args = parser.parse_args()

    snapshot_df = pd.read_parquet(args.snapshots)
    helpfulness_df = pd.read_parquet(args.helpfulness)
    object_to_label = (
        snapshot_df.groupby("object_id")["target_class"].first().to_dict()
    )

    if args.strict_labels and snapshot_df["label_quality"].fillna("").eq("weak").any():
        raise SystemExit("Weak-label truth present; strict mode blocks expert-trust training")

    if snapshot_df["label_quality"].fillna("").eq("weak").any():
        print("WARNING: weak-label truth is present in snapshots; trust metrics should be treated as weak-label results.")
    include_unsafe = args.allow_unsafe_alerce and not args.exclude_unsafe_alerce
    if not include_unsafe:
        print("Excluding ALeRCE latest-object snapshots from trust training by default.")
    else:
        print("Including ALeRCE latest_object_unsafe snapshots in trust training. "
              "The exact__ flag lets the trust model learn temporal reliability.")

    split = group_train_cal_test_split(object_to_label)
    _, metrics_report, metadata = train_expert_trust_suite(
        helpfulness_df=helpfulness_df,
        snapshot_df=snapshot_df,
        split=split,
        models_dir=Path(args.models_dir),
        output_snapshot_path=Path(args.output_snapshots),
        n_splits=args.n_splits,
        allow_unsafe_latest_snapshot=include_unsafe,
        n_jobs=args.n_jobs,
    )

    metrics_out = Path(args.metrics_out) if args.metrics_out else Path(args.models_dir).parent.parent / "reports/metrics/expert_trust_metrics.json"
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w") as fh:
        json.dump(metrics_report, fh, indent=2)
    metadata["allow_unsafe_alerce"] = include_unsafe
    metadata["contains_weak_labels"] = bool(snapshot_df["label_quality"].fillna("").eq("weak").any())
    with open(Path(args.models_dir) / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"Wrote expert trust metrics → {metrics_out}")
    print(f"Wrote trust metadata → {Path(args.models_dir) / 'metadata.json'}")


if __name__ == "__main__":
    main()
