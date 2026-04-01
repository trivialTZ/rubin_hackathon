"""Summarize whether the current data are ready for trust training."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd


def _count_true_by(frame: pd.DataFrame, group_col: str, flag_col: str) -> dict[str, int]:
    if group_col not in frame.columns or flag_col not in frame.columns:
        return {}
    grouped = (
        frame[frame[flag_col].fillna(False).astype(bool)]
        .groupby(group_col)
        .size()
        .sort_values(ascending=False)
    )
    return {str(key): int(value) for key, value in grouped.items()}


def _count_false_by(frame: pd.DataFrame, group_col: str, flag_col: str) -> dict[str, int]:
    if group_col not in frame.columns or flag_col not in frame.columns:
        return {}
    grouped = (
        frame[~frame[flag_col].fillna(False).astype(bool)]
        .groupby(group_col)
        .size()
        .sort_values(ascending=False)
    )
    return {str(key): int(value) for key, value in grouped.items()}


def _read_label_ids(labels_path: Path) -> list[str]:
    with open(labels_path) as fh:
        return [row["object_id"] for row in csv.DictReader(fh)]


def _load_bronze_frame(bronze_dir: Path) -> pd.DataFrame:
    parts = []
    for path in sorted(bronze_dir.glob("*.parquet")):
        parts.append(pd.read_parquet(path))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True, sort=False)


def _summarize_local_experts(local_dir: Path) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    if not local_dir.exists():
        return summary
    for expert_dir in sorted(path for path in local_dir.iterdir() if path.is_dir()):
        parts = [pd.read_parquet(path) for path in sorted(expert_dir.glob("*.parquet"))]
        if not parts:
            continue
        frame = pd.concat(parts, ignore_index=True, sort=False)
        model_versions = []
        if "model_version" in frame.columns:
            model_versions = sorted(str(value) for value in frame["model_version"].dropna().astype(str).unique())
        summary[expert_dir.name] = {
            "rows": int(len(frame)),
            "objects": int(frame["object_id"].nunique()) if "object_id" in frame.columns else 0,
            "available_true": int(frame["available"].fillna(False).astype(bool).sum()) if "available" in frame.columns else 0,
            "model_versions": model_versions,
            "all_stub_versions": bool(model_versions) and all(version == "stub" for version in model_versions),
        }
    return summary


def audit_data_readiness(
    *,
    root: Path,
    labels_path: Path | None = None,
    lightcurve_dir: Path | None = None,
) -> dict:
    data_root = root / "data" if (root / "data").exists() else root
    bronze_dir = data_root / "bronze"
    silver_events = data_root / "silver/broker_events.parquet"
    snapshots = data_root / "gold/object_epoch_snapshots.parquet"
    truth = data_root / "truth/object_truth.parquet"
    labels_csv = labels_path if labels_path is not None else data_root / "labels.csv"
    lightcurve_dir = lightcurve_dir if lightcurve_dir is not None else data_root / "lightcurves"
    local_dir = data_root / "silver/local_expert_outputs"
    trust_models_dir = root / "models/trust"
    followup_model_dir = root / "models/followup"

    report: dict = {
        "root": str(root.resolve()),
        "data_root": str(data_root.resolve()),
        "bronze_files": len(list(bronze_dir.glob("*.parquet"))) if bronze_dir.exists() else 0,
        "silver_present": silver_events.exists(),
        "snapshots_present": snapshots.exists(),
        "truth_present": truth.exists(),
    }

    if labels_csv.exists():
        label_ids = _read_label_ids(labels_csv)
        report["label_rows"] = int(len(label_ids))
        report["label_objects"] = int(len(set(label_ids)))
        if lightcurve_dir.exists():
            lightcurve_ids = {path.stem for path in lightcurve_dir.glob("*.json")}
            missing_lightcurves = sorted(set(label_ids) - lightcurve_ids)
            report["lightcurve_files"] = int(len(lightcurve_ids))
            report["label_objects_missing_lightcurves"] = missing_lightcurves
            report["label_objects_with_lightcurves"] = int(len(set(label_ids) & lightcurve_ids))

    if bronze_dir.exists():
        bronze_df = _load_bronze_frame(bronze_dir)
        if len(bronze_df) > 0:
            report["bronze_broker_counts"] = bronze_df["broker"].value_counts().to_dict()
            if "object_id" in bronze_df.columns:
                report["bronze_object_coverage"] = bronze_df.groupby("broker")["object_id"].nunique().to_dict()
            report["bronze_available_true"] = _count_true_by(bronze_df, "broker", "availability")
            report["bronze_fixture_rows"] = _count_true_by(bronze_df, "broker", "fixture_used")
            if {"availability", "fixture_used", "broker"}.issubset(bronze_df.columns):
                live_rows = bronze_df[
                    bronze_df["availability"].fillna(False).astype(bool)
                    & ~bronze_df["fixture_used"].fillna(False).astype(bool)
                ]
                report["bronze_live_available_rows"] = (
                    live_rows.groupby("broker").size().sort_values(ascending=False).astype(int).to_dict()
                    if len(live_rows) > 0
                    else {}
                )

    if silver_events.exists():
        silver_df = pd.read_parquet(silver_events)
        report["silver_broker_counts"] = silver_df["broker"].value_counts().to_dict()
        report["silver_expert_counts"] = silver_df["expert_key"].value_counts().to_dict()
        report["temporal_exactness_counts"] = silver_df["temporal_exactness"].value_counts(dropna=False).to_dict()
        report["silver_available_true_by_expert"] = _count_true_by(silver_df, "expert_key", "availability")
        report["silver_fixture_rows_by_expert"] = _count_true_by(silver_df, "expert_key", "fixture_used")
        if {"availability", "fixture_used", "expert_key"}.issubset(silver_df.columns):
            live_rows = silver_df[
                silver_df["availability"].fillna(False).astype(bool)
                & ~silver_df["fixture_used"].fillna(False).astype(bool)
            ]
            report["silver_live_available_by_expert"] = (
                live_rows.groupby("expert_key").size().sort_values(ascending=False).astype(int).to_dict()
                if len(live_rows) > 0
                else {}
            )
        exact_alert = silver_df[silver_df["temporal_exactness"] == "exact_alert"].copy()
        report["exact_alert_rows"] = int(len(exact_alert))
        report["exact_alert_rows_with_alert_id"] = int(exact_alert["alert_id"].notna().sum()) if "alert_id" in exact_alert.columns else 0
        report["exact_alert_rows_missing_n_det"] = int(exact_alert["n_det"].isna().sum()) if "n_det" in exact_alert.columns else int(len(exact_alert))
        report["exact_alert_rows_missing_event_time_jd"] = int(exact_alert["event_time_jd"].isna().sum()) if "event_time_jd" in exact_alert.columns else int(len(exact_alert))
    if truth.exists():
        truth_df = pd.read_parquet(truth)
        report["truth_label_sources"] = truth_df["label_source"].value_counts(dropna=False).to_dict()
        report["truth_label_quality"] = truth_df["label_quality"].value_counts(dropna=False).to_dict()
    if snapshots.exists():
        snap_df = pd.read_parquet(snapshots)
        report["snapshot_rows"] = int(len(snap_df))
        report["snapshot_objects"] = int(snap_df["object_id"].nunique())
        report["snapshot_non_null"] = {}
        report["snapshot_available_true"] = {}
        for column in snap_df.columns:
            if column.startswith("proj__"):
                report["snapshot_non_null"][column] = int(snap_df[column].notna().sum())
            elif column.startswith("avail__"):
                report["snapshot_available_true"][column] = int(
                    snap_df[column].fillna(0).astype(bool).sum()
                )
    report["local_experts"] = _summarize_local_experts(local_dir)
    if (trust_models_dir / "metadata.json").exists():
        with open(trust_models_dir / "metadata.json") as fh:
            trust_metadata = json.load(fh)
        report["trust_models"] = {
            "path": str(trust_models_dir),
            "experts": trust_metadata.get("experts", []),
            "allow_unsafe_alerce": trust_metadata.get("allow_unsafe_alerce"),
            "contains_weak_labels": trust_metadata.get("contains_weak_labels"),
        }
    if (followup_model_dir / "metadata.json").exists():
        with open(followup_model_dir / "metadata.json") as fh:
            followup_metadata = json.load(fh)
        report["followup_model"] = {
            "path": str(followup_model_dir),
            "feature_count": len(followup_metadata.get("feature_cols", [])),
        }

    report["warnings"] = []
    if not report["bronze_files"]:
        report["warnings"].append("no bronze broker payloads found")
    if not report["silver_present"]:
        report["warnings"].append("silver broker_events.parquet is missing")
    if not report["truth_present"]:
        report["warnings"].append("truth table is missing")
    if not report["snapshots_present"]:
        report["warnings"].append("object_epoch_snapshots.parquet is missing")
    if report.get("truth_label_sources", {}).get("alerce_self_label"):
        report["warnings"].append("weak ALeRCE self-label truth present")
    silver_experts = set(report.get("silver_expert_counts", {}).keys())
    if silver_experts:
        if not any(expert.startswith("fink/") for expert in silver_experts):
            report["warnings"].append("no Fink expert evidence present in silver")
        if "lasair/sherlock" not in silver_experts:
            report["warnings"].append("no Lasair Sherlock context present in silver")
        if silver_experts.issubset({"alerce/lc_classifier_transient", "alerce/stamp_classifier", "alerce/lc_classifier", "alerce/lc_classifier_top", "alerce/lc_classifier_periodic", "alerce/lc_classifier_stochastic"}):
            report["warnings"].append("silver is effectively ALeRCE-only")
    bronze_live_rows = report.get("bronze_live_available_rows", {})
    if report.get("bronze_broker_counts") and not bronze_live_rows.get("lasair"):
        report["warnings"].append("Lasair is not returning live bronze payloads; check LASAIR_TOKEN or fixture fallback")
    if report.get("bronze_broker_counts") and not bronze_live_rows.get("fink"):
        report["warnings"].append("Fink is not returning live bronze payloads")
    if truth.exists():
        truth_df = pd.read_parquet(truth)
        if truth_df["label_quality"].fillna("").eq("strong").sum() == 0:
            report["warnings"].append("no strong truth rows present")
    if labels_csv.exists() and report.get("label_objects_missing_lightcurves"):
        report["warnings"].append("some labelled objects are missing downloaded lightcurves")
    if int(report.get("exact_alert_rows", 0)) > 0 and int(report.get("exact_alert_rows_with_alert_id", 0)) < int(report.get("exact_alert_rows", 0)):
        report["warnings"].append("some exact-alert rows are missing alert_id")
    if int(report.get("exact_alert_rows_missing_n_det", 0)) > 0:
        report["warnings"].append("some exact-alert rows are missing n_det")
    if int(report.get("exact_alert_rows_missing_event_time_jd", 0)) > 0:
        report["warnings"].append("some exact-alert rows are missing event_time_jd")
    if not local_dir.exists() or not any(local_dir.glob("*/*.parquet")):
        report["warnings"].append("no local expert rerun outputs present")
    elif report["local_experts"] and all(info["available_true"] == 0 for info in report["local_experts"].values()):
        report["warnings"].append("local expert outputs are present but none are marked available")
    if report.get("local_experts") and any(info["all_stub_versions"] for info in report["local_experts"].values()):
        report["warnings"].append("one or more local experts are still stub-only")
    if report.get("snapshot_available_true") and not any(report["snapshot_available_true"].values()):
        report["warnings"].append("no experts are actually available in object-epoch snapshots")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit DEBASS data readiness")
    parser.add_argument("--root", default=".")
    parser.add_argument("--labels", default=None)
    parser.add_argument("--lightcurves-dir", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--md-out", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    report = audit_data_readiness(
        root=root,
        labels_path=Path(args.labels) if args.labels else None,
        lightcurve_dir=Path(args.lightcurves_dir) if args.lightcurves_dir else None,
    )
    json_out = Path(args.json_out) if args.json_out else root / "reports/summary/data_readiness.json"
    md_out = Path(args.md_out) if args.md_out else root / "reports/summary/data_readiness.md"
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
