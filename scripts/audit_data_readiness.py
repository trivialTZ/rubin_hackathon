"""Summarize whether the current data are ready for trust training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd


def audit_data_readiness(*, root: Path) -> dict:
    bronze_dir = root / "data/bronze"
    silver_events = root / "data/silver/broker_events.parquet"
    snapshots = root / "data/gold/object_epoch_snapshots.parquet"
    truth = root / "data/truth/object_truth.parquet"

    report: dict = {
        "bronze_files": len(list(bronze_dir.glob("*.parquet"))) if bronze_dir.exists() else 0,
        "silver_present": silver_events.exists(),
        "snapshots_present": snapshots.exists(),
        "truth_present": truth.exists(),
    }

    if silver_events.exists():
        silver_df = pd.read_parquet(silver_events)
        report["silver_broker_counts"] = silver_df["broker"].value_counts().to_dict()
        report["silver_expert_counts"] = silver_df["expert_key"].value_counts().to_dict()
        report["temporal_exactness_counts"] = silver_df["temporal_exactness"].value_counts(dropna=False).to_dict()
    if truth.exists():
        truth_df = pd.read_parquet(truth)
        report["truth_label_sources"] = truth_df["label_source"].value_counts(dropna=False).to_dict()
        report["truth_label_quality"] = truth_df["label_quality"].value_counts(dropna=False).to_dict()
    if snapshots.exists():
        snap_df = pd.read_parquet(snapshots)
        report["snapshot_rows"] = int(len(snap_df))
        report["snapshot_objects"] = int(snap_df["object_id"].nunique())
        report["snapshot_non_null"] = {
            column: int(snap_df[column].notna().sum())
            for column in snap_df.columns
            if column.startswith("proj__") or column.startswith("avail__")
        }
    report["warnings"] = []
    if report.get("truth_label_sources", {}).get("alerce_self_label"):
        report["warnings"].append("weak ALeRCE self-label truth present")
    if report.get("silver_expert_counts", {}).keys() == {"alerce/lc_classifier_transient"}:
        report["warnings"].append("silver is effectively ALeRCE-only")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit DEBASS data readiness")
    parser.add_argument("--root", default=".")
    parser.add_argument("--json-out", default="reports/summary/data_readiness.json")
    parser.add_argument("--md-out", default="reports/summary/data_readiness.md")
    args = parser.parse_args()

    report = audit_data_readiness(root=Path(args.root))
    json_out = Path(args.json_out)
    md_out = Path(args.md_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, "w") as fh:
        json.dump(report, fh, indent=2)
    with open(md_out, "w") as fh:
        fh.write("# Data Readiness\n\n")
        for key, value in report.items():
            fh.write(f"- **{key}**: `{value}`\n")
    print(f"Wrote readiness report → {json_out}")
    print(f"Wrote readiness markdown → {md_out}")


if __name__ == "__main__":
    main()
