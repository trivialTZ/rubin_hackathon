"""Summarize whether the current data are ready for trust training."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pyarrow.parquet as pq


BATCH_SIZE = 65_536
UNIQUE_VALUE_LIMIT = 1_000_000


def _parquet_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.parquet"))
    return []


def _schema_names(paths: Iterable[Path]) -> set[str]:
    names: set[str] = set()
    for path in paths:
        names.update(pq.ParquetFile(path).schema_arrow.names)
    return names


def _row_count(paths: Iterable[Path]) -> int:
    return int(sum(pq.ParquetFile(path).metadata.num_rows for path in paths))


def _counter_to_dict(counter: Counter[str]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))
    }


def _key(value: object) -> str:
    if value is None:
        return "None"
    return str(value)


def _truthy(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def _iter_column_batches(paths: Iterable[Path], columns: Iterable[str]):
    requested = list(dict.fromkeys(columns))
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        names = set(parquet_file.schema_arrow.names)
        read_columns = [column for column in requested if column in names]
        if not read_columns:
            continue
        for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=read_columns):
            arrays = {
                column: batch.column(batch.schema.get_field_index(column)).to_pylist()
                for column in read_columns
            }
            yield arrays, int(batch.num_rows)


def _value_counts(paths: Iterable[Path], column: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for arrays, _ in _iter_column_batches(paths, [column]):
        for value in arrays.get(column, []):
            counter[_key(value)] += 1
    return _counter_to_dict(counter)


def _count_true_by(paths: Iterable[Path], group_col: str, flag_col: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for arrays, rows in _iter_column_batches(paths, [group_col, flag_col]):
        groups = arrays.get(group_col)
        flags = arrays.get(flag_col)
        if groups is None or flags is None:
            continue
        for idx in range(rows):
            if _truthy(flags[idx]):
                counter[_key(groups[idx])] += 1
    return _counter_to_dict(counter)


def _count_false_by(paths: Iterable[Path], group_col: str, flag_col: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for arrays, rows in _iter_column_batches(paths, [group_col, flag_col]):
        groups = arrays.get(group_col)
        flags = arrays.get(flag_col)
        if groups is None or flags is None:
            continue
        for idx in range(rows):
            if not _truthy(flags[idx]):
                counter[_key(groups[idx])] += 1
    return _counter_to_dict(counter)


def _count_live_available_by(paths: Iterable[Path], group_col: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for arrays, rows in _iter_column_batches(paths, [group_col, "availability", "fixture_used"]):
        groups = arrays.get(group_col)
        availability = arrays.get("availability")
        fixture_used = arrays.get("fixture_used")
        if groups is None or availability is None or fixture_used is None:
            continue
        for idx in range(rows):
            if _truthy(availability[idx]) and not _truthy(fixture_used[idx]):
                counter[_key(groups[idx])] += 1
    return _counter_to_dict(counter)


def _unique_count(paths: Iterable[Path], column: str) -> tuple[int, bool]:
    values: set[str] = set()
    exact = True
    for arrays, _ in _iter_column_batches(paths, [column]):
        for value in arrays.get(column, []):
            if value is None:
                continue
            values.add(_key(value))
            if len(values) > UNIQUE_VALUE_LIMIT:
                exact = False
                return len(values), exact
    return len(values), exact


def _unique_values(paths: Iterable[Path], column: str) -> tuple[list[str], bool]:
    values: set[str] = set()
    exact = True
    for arrays, _ in _iter_column_batches(paths, [column]):
        for value in arrays.get(column, []):
            if value is None:
                continue
            values.add(_key(value))
            if len(values) > UNIQUE_VALUE_LIMIT:
                exact = False
                return sorted(values), exact
    return sorted(values), exact


def _unique_count_by(paths: Iterable[Path], group_col: str, value_col: str) -> dict[str, int]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for arrays, rows in _iter_column_batches(paths, [group_col, value_col]):
        groups = arrays.get(group_col)
        values = arrays.get(value_col)
        if groups is None or values is None:
            continue
        for idx in range(rows):
            if values[idx] is not None:
                grouped[_key(groups[idx])].add(_key(values[idx]))
    return {
        key: int(len(values))
        for key, values in sorted(grouped.items(), key=lambda item: (-len(item[1]), str(item[0])))
    }


def _non_null_counts(paths: Iterable[Path], columns: Iterable[str]) -> dict[str, int]:
    counters = {column: 0 for column in columns}
    for arrays, rows in _iter_column_batches(paths, columns):
        for column, values in arrays.items():
            counters[column] += sum(1 for idx in range(rows) if values[idx] is not None)
    return {column: int(value) for column, value in counters.items()}


def _true_counts_for_columns(paths: Iterable[Path], columns: Iterable[str]) -> dict[str, int]:
    counters = {column: 0 for column in columns}
    for arrays, rows in _iter_column_batches(paths, columns):
        for column, values in arrays.items():
            counters[column] += sum(1 for idx in range(rows) if _truthy(values[idx]))
    return {column: int(value) for column, value in counters.items()}


def _read_label_manifest(labels_path: Path) -> dict[str, list[str]]:
    object_ids: list[str] = []
    labelled_object_ids: list[str] = []
    unlabelled_object_ids: list[str] = []
    with open(labels_path) as fh:
        reader = csv.DictReader(fh)
        has_label_column = bool(reader.fieldnames and "label" in reader.fieldnames)
        for row in reader:
            object_id = str(row["object_id"]).strip()
            if not object_id:
                continue
            object_ids.append(object_id)
            label = str(row.get("label") or "").strip()
            if has_label_column and not label:
                unlabelled_object_ids.append(object_id)
            else:
                labelled_object_ids.append(object_id)
    return {
        "object_ids": object_ids,
        "labelled_object_ids": labelled_object_ids,
        "unlabelled_object_ids": unlabelled_object_ids,
    }


def _summarize_silver_events(path: Path) -> dict:
    paths = _parquet_paths(path)
    report: dict = {
        "silver_broker_counts": _value_counts(paths, "broker"),
        "silver_expert_counts": _value_counts(paths, "expert_key"),
        "temporal_exactness_counts": _value_counts(paths, "temporal_exactness"),
        "silver_available_true_by_expert": _count_true_by(paths, "expert_key", "availability"),
        "silver_fixture_rows_by_expert": _count_true_by(paths, "expert_key", "fixture_used"),
        "silver_live_available_by_expert": _count_live_available_by(paths, "expert_key"),
    }

    exact_rows = 0
    exact_with_alert_id = 0
    exact_missing_n_det = 0
    exact_missing_event_time_jd = 0
    for arrays, rows in _iter_column_batches(
        paths,
        ["temporal_exactness", "alert_id", "n_det", "event_time_jd"],
    ):
        temporal_exactness = arrays.get("temporal_exactness", [None] * rows)
        alert_ids = arrays.get("alert_id", [None] * rows)
        n_dets = arrays.get("n_det", [None] * rows)
        event_times = arrays.get("event_time_jd", [None] * rows)
        for idx in range(rows):
            if temporal_exactness[idx] != "exact_alert":
                continue
            exact_rows += 1
            if alert_ids[idx] is not None:
                exact_with_alert_id += 1
            if n_dets[idx] is None:
                exact_missing_n_det += 1
            if event_times[idx] is None:
                exact_missing_event_time_jd += 1

    schema = _schema_names(paths)
    temporal_columns = ["event_time_jd", "n_det", "alert_id", "event_scope", "temporal_exactness"]
    report.update(
        {
            "silver_columns": sorted(schema),
            "silver_required_temporal_columns_present": {
                column: column in schema for column in temporal_columns
            },
            "exact_alert_rows": int(exact_rows),
            "exact_alert_rows_with_alert_id": int(exact_with_alert_id),
            "exact_alert_rows_missing_n_det": int(exact_missing_n_det),
            "exact_alert_rows_missing_event_time_jd": int(exact_missing_event_time_jd),
            "silver_unsafe_alerce_rows_by_expert": _count_temporal_exactness_by_expert(
                paths,
                "latest_object_unsafe",
            ),
        }
    )
    return report


def _count_temporal_exactness_by_expert(paths: Iterable[Path], value: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for arrays, rows in _iter_column_batches(paths, ["expert_key", "temporal_exactness"]):
        expert_keys = arrays.get("expert_key")
        exactness = arrays.get("temporal_exactness")
        if expert_keys is None or exactness is None:
            continue
        for idx in range(rows):
            if exactness[idx] == value:
                counter[_key(expert_keys[idx])] += 1
    return _counter_to_dict(counter)


def _summarize_snapshots(path: Path) -> dict:
    paths = _parquet_paths(path)
    schema = _schema_names(paths)
    proj_cols = sorted(column for column in schema if column.startswith("proj__"))
    avail_cols = sorted(column for column in schema if column.startswith("avail__"))
    snapshot_objects, exact = _unique_count(paths, "object_id")
    temporal_columns = ["event_time_jd", "n_det", "alert_id", "event_scope", "temporal_exactness"]
    return {
        "snapshot_rows": _row_count(paths),
        "snapshot_objects": int(snapshot_objects),
        "snapshot_objects_exact": bool(exact),
        "snapshot_columns": sorted(schema),
        "snapshot_required_temporal_columns_present": {
            column: column in schema for column in temporal_columns
        },
        "snapshot_non_null": _non_null_counts(paths, proj_cols),
        "snapshot_available_true": _true_counts_for_columns(paths, avail_cols),
    }


def _summarize_local_experts(local_dir: Path) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    if not local_dir.exists():
        return summary
    for expert_dir in sorted(path for path in local_dir.iterdir() if path.is_dir()):
        paths = _parquet_paths(expert_dir)
        if not paths:
            continue
        model_versions, _versions_exact = _unique_values(paths, "model_version")
        objects, objects_exact = _unique_count(paths, "object_id")
        summary[expert_dir.name] = {
            "rows": _row_count(paths),
            "objects": int(objects),
            "objects_exact": bool(objects_exact),
            "available_true": int(sum(_true_counts_for_columns(paths, ["available"]).values())),
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

    bronze_paths = _parquet_paths(bronze_dir)
    silver_paths = _parquet_paths(silver_events)
    snapshot_paths = _parquet_paths(snapshots)
    truth_paths = _parquet_paths(truth)

    report: dict = {
        "root": str(root.resolve()),
        "data_root": str(data_root.resolve()),
        "audit_mode": "streaming_parquet_batches",
        "bronze_files": len(bronze_paths),
        "silver_present": bool(silver_paths),
        "snapshots_present": bool(snapshot_paths),
        "truth_present": bool(truth_paths),
    }

    if labels_csv.exists():
        label_manifest = _read_label_manifest(labels_csv)
        label_ids = label_manifest["object_ids"]
        labelled_ids = label_manifest["labelled_object_ids"]
        unlabelled_ids = label_manifest["unlabelled_object_ids"]
        report["label_rows"] = int(len(label_ids))
        report["label_objects"] = int(len(set(label_ids)))
        report["labelled_rows"] = int(len(labelled_ids))
        report["labelled_objects"] = int(len(set(labelled_ids)))
        report["unlabelled_rows"] = int(len(unlabelled_ids))
        report["unlabelled_objects"] = int(len(set(unlabelled_ids)))
        if lightcurve_dir.exists():
            lightcurve_ids = {path.stem for path in lightcurve_dir.glob("*.json")}
            missing_lightcurves = sorted(set(labelled_ids) - lightcurve_ids)
            missing_input_lightcurves = sorted(set(label_ids) - lightcurve_ids)
            report["lightcurve_files"] = int(len(lightcurve_ids))
            report["label_objects_missing_lightcurves"] = missing_lightcurves
            report["label_objects_with_lightcurves"] = int(len(set(labelled_ids) & lightcurve_ids))
            report["input_objects_missing_lightcurves_count"] = int(len(missing_input_lightcurves))
            report["input_objects_missing_lightcurves_sample"] = missing_input_lightcurves[:20]

    if bronze_paths:
        report["bronze_broker_counts"] = _value_counts(bronze_paths, "broker")
        report["bronze_object_coverage"] = _unique_count_by(bronze_paths, "broker", "object_id")
        report["bronze_available_true"] = _count_true_by(bronze_paths, "broker", "availability")
        report["bronze_fixture_rows"] = _count_true_by(bronze_paths, "broker", "fixture_used")
        report["bronze_unavailable_rows"] = _count_false_by(bronze_paths, "broker", "availability")
        report["bronze_live_available_rows"] = _count_live_available_by(bronze_paths, "broker")

    if silver_paths:
        report.update(_summarize_silver_events(silver_events))

    if truth_paths:
        report["truth_label_sources"] = _value_counts(truth_paths, "label_source")
        report["truth_label_quality"] = _value_counts(truth_paths, "label_quality")

    if snapshot_paths:
        report.update(_summarize_snapshots(snapshots))

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
        if silver_experts.issubset({
            "alerce/lc_classifier_transient",
            "alerce/stamp_classifier",
            "alerce/lc_classifier",
            "alerce/lc_classifier_top",
            "alerce/lc_classifier_periodic",
            "alerce/lc_classifier_stochastic",
        }):
            report["warnings"].append("silver is effectively ALeRCE-only")
    bronze_live_rows = report.get("bronze_live_available_rows", {})
    if report.get("bronze_broker_counts") and not bronze_live_rows.get("lasair"):
        report["warnings"].append("Lasair is not returning live bronze payloads; check LASAIR_TOKEN or fixture fallback")
    if report.get("bronze_broker_counts") and not bronze_live_rows.get("fink"):
        report["warnings"].append("Fink is not returning live bronze payloads")
    truth_quality = report.get("truth_label_quality", {})
    strong_like_truth = (
        int(truth_quality.get("strong", 0))
        + int(truth_quality.get("spectroscopic", 0))
        + int(truth_quality.get("consensus", 0))
    )
    if truth_paths and strong_like_truth == 0:
        report["warnings"].append("no strong, spectroscopic, or consensus truth rows present")
    if labels_csv.exists() and report.get("label_objects_missing_lightcurves"):
        report["warnings"].append("some labelled objects are missing downloaded lightcurves")
    if int(report.get("exact_alert_rows", 0)) > 0 and int(report.get("exact_alert_rows_with_alert_id", 0)) < int(report.get("exact_alert_rows", 0)):
        report["warnings"].append("some exact-alert rows are missing alert_id")
    if int(report.get("exact_alert_rows_missing_n_det", 0)) > 0:
        report["warnings"].append("some exact-alert rows are missing n_det")
    if int(report.get("exact_alert_rows_missing_event_time_jd", 0)) > 0:
        report["warnings"].append("some exact-alert rows are missing event_time_jd")
    missing_silver_temporal = [
        column
        for column, present in (report.get("silver_required_temporal_columns_present") or {}).items()
        if not present
    ]
    if missing_silver_temporal:
        report["warnings"].append(f"silver is missing temporal contract columns: {missing_silver_temporal}")
    unsafe_alerce_rows = report.get("silver_unsafe_alerce_rows_by_expert", {})
    if unsafe_alerce_rows:
        report["warnings"].append("unsafe historical ALeRCE object snapshots are present in silver")
    trust_models = report.get("trust_models") or {}
    if trust_models.get("allow_unsafe_alerce"):
        report["warnings"].append("trust metadata allows unsafe ALeRCE snapshots")
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
    parser = argparse.ArgumentParser(description="Audit metaDEBASS data readiness")
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
    print(f"Wrote readiness report -> {json_out}")
    print(f"Wrote readiness markdown -> {md_out}")


if __name__ == "__main__":
    main()
