"""Summarize DEBASS CPU-prep outputs and on-disk data structures.

Usage:
    python scripts/summarize_cpu_prep.py
    python scripts/summarize_cpu_prep.py --json-out reports/summary/cpu_prep_summary.json
    python scripts/summarize_cpu_prep.py --strict
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from debass_meta.features.lightcurve import FEATURE_NAMES


def _iso_mtime(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _path_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    return {
        "exists": True,
        "path": str(path),
        "bytes": path.stat().st_size,
        "mtime_utc": _iso_mtime(path),
    }


def _numeric_summary(values: list[int]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": round(sum(values) / len(values), 3),
        "max": max(values),
    }


def _series_counts(series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).items()}


def _labels_summary(labels_path: Path) -> tuple[dict[str, Any], set[str]]:
    import pandas as pd

    summary: dict[str, Any] = {"file": _path_info(labels_path)}
    if not labels_path.exists():
        summary["status"] = "missing"
        return summary, set()

    df = pd.read_csv(labels_path)
    object_ids = set(df["object_id"].astype(str)) if "object_id" in df.columns else set()
    duplicates = (
        df[df["object_id"].astype(str).duplicated(keep=False)]["object_id"].astype(str).unique().tolist()
        if "object_id" in df.columns
        else []
    )
    summary.update(
        {
            "status": "ok",
            "columns": df.columns.tolist(),
            "rows": int(len(df)),
            "unique_objects": int(len(object_ids)),
            "label_counts": _series_counts(df["label"]) if "label" in df.columns else {},
            "duplicate_object_ids": sorted(duplicates),
            "sample_object_ids": sorted(object_ids)[:5],
        }
    )
    return summary, object_ids


def _load_lightcurve_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return [rec for rec in payload if isinstance(rec, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("detections"), list):
        return [rec for rec in payload["detections"] if isinstance(rec, dict)]
    return []


def _lightcurves_summary(lc_dir: Path, label_object_ids: set[str]) -> tuple[dict[str, Any], set[str]]:
    summary: dict[str, Any] = {"dir": _path_info(lc_dir)}
    if not lc_dir.exists():
        summary["status"] = "missing"
        return summary, set()

    files = sorted(lc_dir.glob("*.json"))
    if not files:
        summary["status"] = "empty"
        summary["files"] = 0
        return summary, set()

    object_ids: set[str] = set()
    det_counts: list[int] = []
    band_counts: Counter[str] = Counter()
    sample_keys: list[str] = []
    read_errors: list[str] = []

    for path in files:
        try:
            records = _load_lightcurve_records(path)
        except Exception:
            read_errors.append(path.name)
            continue
        object_ids.add(path.stem)
        det_counts.append(len(records))
        if not sample_keys and records:
            sample_keys = sorted(records[0].keys())
        for rec in records:
            fid = rec.get("fid")
            if fid is not None:
                band_counts[str(fid)] += 1

    missing_for_labels = sorted(label_object_ids - object_ids) if label_object_ids else []
    extra_without_label = sorted(object_ids - label_object_ids) if label_object_ids else []

    summary.update(
        {
            "status": "ok",
            "files": int(len(files)),
            "objects": int(len(object_ids)),
            "detection_count_summary": _numeric_summary(det_counts),
            "band_counts": dict(sorted(band_counts.items())),
            "sample_object_ids": sorted(object_ids)[:5],
            "sample_detection_keys": sample_keys,
            "read_errors": read_errors[:10],
            "missing_lightcurves_for_labels": missing_for_labels[:20],
            "extra_lightcurves_without_label": extra_without_label[:20],
        }
    )
    return summary, object_ids


def _bronze_summary(bronze_dir: Path) -> tuple[dict[str, Any], set[str]]:
    import pandas as pd

    summary: dict[str, Any] = {
        "dir": _path_info(bronze_dir),
        "schema": str((Path("schemas") / "bronze.json").resolve()),
    }
    if not bronze_dir.exists():
        summary["status"] = "missing"
        return summary, set()

    files = sorted(bronze_dir.glob("*.parquet"))
    if not files:
        summary["status"] = "empty"
        summary["files"] = 0
        return summary, set()

    broker_counts: Counter[str] = Counter()
    semantic_counts: Counter[str] = Counter()
    objects: set[str] = set()
    columns: set[str] = set()
    fixture_rows = 0
    per_file: list[dict[str, Any]] = []
    total_rows = 0

    for path in files:
        df = pd.read_parquet(path)
        total_rows += len(df)
        if "broker" in df.columns:
            broker_counts.update(df["broker"].astype(str).tolist())
        if "semantic_type" in df.columns:
            semantic_counts.update(df["semantic_type"].astype(str).tolist())
        if "object_id" in df.columns:
            objects.update(df["object_id"].astype(str).tolist())
        if "fixture_used" in df.columns:
            fixture_rows += int(df["fixture_used"].fillna(False).astype(bool).sum())
        columns.update(df.columns.tolist())
        per_file.append(
            {
                "name": path.name,
                "rows": int(len(df)),
                "brokers": _series_counts(df["broker"]) if "broker" in df.columns else {},
            }
        )

    summary.update(
        {
            "status": "ok",
            "files": int(len(files)),
            "total_rows": int(total_rows),
            "objects": int(len(objects)),
            "brokers": dict(sorted(broker_counts.items())),
            "semantic_types": dict(sorted(semantic_counts.items())),
            "fixture_rows": int(fixture_rows),
            "columns": sorted(columns),
            "per_file": per_file,
        }
    )
    return summary, objects


def _silver_summary(silver_path: Path) -> tuple[dict[str, Any], set[str]]:
    import pandas as pd

    summary: dict[str, Any] = {
        "file": _path_info(silver_path),
        "schema": str((Path("schemas") / "silver.json").resolve()),
    }
    if not silver_path.exists():
        summary["status"] = "missing"
        return summary, set()

    df = pd.read_parquet(silver_path)
    objects = set(df["object_id"].astype(str)) if "object_id" in df.columns else set()

    fields_by_broker: dict[str, list[str]] = {}
    canonical_non_null: dict[str, int] = {}
    if {"broker", "field"}.issubset(df.columns):
        for broker, broker_df in df.groupby("broker"):
            fields_by_broker[str(broker)] = sorted(broker_df["field"].astype(str).unique().tolist())[:20]
            if "canonical_projection" in broker_df.columns:
                canonical_non_null[str(broker)] = int(broker_df["canonical_projection"].notna().sum())

    summary.update(
        {
            "status": "ok",
            "rows": int(len(df)),
            "objects": int(len(objects)),
            "brokers": _series_counts(df["broker"]) if "broker" in df.columns else {},
            "fields_by_broker": fields_by_broker,
            "canonical_projection_non_null_by_broker": canonical_non_null,
            "fixture_rows": int(df["fixture_used"].fillna(False).astype(bool).sum()) if "fixture_used" in df.columns else 0,
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample_rows": df.head(3).to_dict(orient="records"),
        }
    )
    return summary, objects


def _epochs_summary(epochs_path: Path) -> tuple[dict[str, Any], set[str]]:
    import pandas as pd

    summary: dict[str, Any] = {"file": _path_info(epochs_path)}
    if not epochs_path.exists():
        summary["status"] = "missing"
        return summary, set()

    df = pd.read_parquet(epochs_path)
    objects = set(df["object_id"].astype(str)) if "object_id" in df.columns else set()

    availability_cols = sorted([col for col in df.columns if col.endswith("_available")])
    local_expert_cols = [col for col in ("snn_prob_ia", "parsnip_prob_ia") if col in df.columns]
    broker_score_cols = sorted(
        [
            col for col in df.columns
            if (
                col.startswith(("alerce_", "fink_"))
                and col not in availability_cols
            )
        ]
    )

    non_null_fraction = {
        col: round(float(df[col].notna().mean()), 4)
        for col in broker_score_cols + local_expert_cols
        if col in df.columns
    }
    row_counts_by_n_det = (
        {int(k): int(v) for k, v in df["n_det"].value_counts().sort_index().items()}
        if "n_det" in df.columns
        else {}
    )

    summary.update(
        {
            "status": "ok",
            "rows": int(len(df)),
            "objects": int(len(objects)),
            "columns": df.columns.tolist(),
            "metadata_columns": [col for col in ("object_id", "n_det", "alert_mjd", "target_label") if col in df.columns],
            "lightcurve_feature_columns": [col for col in FEATURE_NAMES if col in df.columns],
            "broker_score_columns": broker_score_cols,
            "availability_columns": availability_cols,
            "local_expert_columns": local_expert_cols,
            "label_counts": _series_counts(df["target_label"].dropna()) if "target_label" in df.columns else {},
            "n_det_summary": _numeric_summary(df["n_det"].astype(int).tolist()) if "n_det" in df.columns else None,
            "row_counts_by_n_det": row_counts_by_n_det,
            "rows_n_det_le_5": int((df["n_det"] <= 5).sum()) if "n_det" in df.columns else 0,
            "non_null_fraction": non_null_fraction,
            "sample_rows": (
                df[[col for col in ("object_id", "n_det", "alert_mjd", "target_label") if col in df.columns]]
                .head(5)
                .to_dict(orient="records")
            ),
        }
    )
    return summary, objects


def _local_expert_summary(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"file": _path_info(path)}
    if not path.exists():
        summary["status"] = "missing_optional"
        return summary

    try:
        records = json.loads(path.read_text())
    except Exception as exc:
        summary["status"] = "read_error"
        summary["error"] = str(exc)
        return summary

    expert_counts: Counter[str] = Counter()
    objects: set[str] = set()
    ndets: list[int] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        expert_counts.update([str(rec.get("expert", "<missing>"))])
        if rec.get("object_id"):
            objects.add(str(rec["object_id"]))
        if rec.get("n_det") is not None:
            try:
                ndets.append(int(rec["n_det"]))
            except Exception:
                pass

    summary.update(
        {
            "status": "ok",
            "records": int(len(records)),
            "objects": int(len(objects)),
            "experts": dict(sorted(expert_counts.items())),
            "n_det_summary": _numeric_summary(ndets),
            "sample_rows": records[:3],
        }
    )
    return summary


def _build_checks(
    label_object_ids: set[str],
    lc_object_ids: set[str],
    bronze_object_ids: set[str],
    silver_object_ids: set[str],
    epoch_object_ids: set[str],
) -> dict[str, Any]:
    missing_lc = sorted(label_object_ids - lc_object_ids)
    missing_silver = sorted(label_object_ids - silver_object_ids)
    missing_epochs = sorted(label_object_ids - epoch_object_ids)
    checks = {
        "labels_have_lightcurves": {
            "ok": len(missing_lc) == 0,
            "missing_count": len(missing_lc),
            "missing_sample": missing_lc[:20],
        },
        "labels_represented_in_silver": {
            "ok": len(missing_silver) == 0,
            "missing_count": len(missing_silver),
            "missing_sample": missing_silver[:20],
        },
        "labels_represented_in_epochs": {
            "ok": len(missing_epochs) == 0,
            "missing_count": len(missing_epochs),
            "missing_sample": missing_epochs[:20],
        },
        "bronze_objects_overlap_labels": {
            "ok": bool(bronze_object_ids & label_object_ids) if bronze_object_ids and label_object_ids else False,
            "overlap_count": len(bronze_object_ids & label_object_ids),
        },
    }
    checks["overall_ok"] = all(item["ok"] for item in checks.values())
    return checks


def _print_summary(summary: dict[str, Any]) -> None:
    def p(line: str = "") -> None:
        print(line)

    labels = summary["labels"]
    lc = summary["lightcurves"]
    bronze = summary["bronze"]
    silver = summary["silver"]
    epochs = summary["epochs"]
    local_experts = summary["local_expert_outputs"]
    checks = summary["checks"]

    p("DEBASS CPU Prep Summary")
    p(f"generated_at_utc: {summary['generated_at_utc']}")
    p(f"data_root:         {summary['data_root']}")
    p("")

    p("Labels")
    p(f"  rows:            {labels.get('rows', 0)}")
    p(f"  unique_objects:  {labels.get('unique_objects', 0)}")
    p(f"  label_counts:    {labels.get('label_counts', {})}")
    p(f"  file:            {labels['file']['path']}")
    p("")

    p("Lightcurves")
    p(f"  files:           {lc.get('files', 0)}")
    p(f"  detection_stats: {lc.get('detection_count_summary')}")
    p(f"  sample_keys:     {lc.get('sample_detection_keys', [])}")
    if lc.get("missing_lightcurves_for_labels"):
        p(f"  missing_labels:  {lc['missing_lightcurves_for_labels']}")
    p("")

    p("Bronze")
    p(f"  files:           {bronze.get('files', 0)}")
    p(f"  rows:            {bronze.get('total_rows', 0)}")
    p(f"  brokers:         {bronze.get('brokers', {})}")
    p("")

    p("Silver")
    p(f"  rows:            {silver.get('rows', 0)}")
    p(f"  objects:         {silver.get('objects', 0)}")
    p(f"  brokers:         {silver.get('brokers', {})}")
    p(f"  columns:         {len(silver.get('columns', []))}")
    p("")

    p("Epochs")
    p(f"  rows:            {epochs.get('rows', 0)}")
    p(f"  objects:         {epochs.get('objects', 0)}")
    p(f"  n_det_summary:   {epochs.get('n_det_summary')}")
    p(f"  rows_n_det<=5:   {epochs.get('rows_n_det_le_5', 0)}")
    p(f"  label_counts:    {epochs.get('label_counts', {})}")
    p(f"  broker_cols:     {epochs.get('broker_score_columns', [])}")
    p(f"  expert_cols:     {epochs.get('local_expert_columns', [])}")
    p("")

    p("Local Expert Outputs")
    p(f"  status:          {local_experts.get('status')}")
    if local_experts.get("status") == "ok":
        p(f"  records:         {local_experts.get('records', 0)}")
        p(f"  experts:         {local_experts.get('experts', {})}")
    p("")

    p("Checks")
    for name, info in checks.items():
        if name == "overall_ok":
            continue
        p(f"  {name}: {info}")
    p(f"  overall_ok:      {checks['overall_ok']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize DEBASS CPU-prep outputs")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--json-out", default=None, help="Optional JSON report path")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if core CPU checks fail")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    labels_path = data_root / "labels.csv"
    lc_dir = data_root / "lightcurves"
    bronze_dir = data_root / "bronze"
    silver_path = data_root / "silver" / "broker_events.parquet"
    local_expert_path = data_root / "silver" / "local_expert_outputs.json"
    epochs_path = data_root / "epochs" / "epoch_features.parquet"

    labels, label_object_ids = _labels_summary(labels_path)
    lightcurves, lc_object_ids = _lightcurves_summary(lc_dir, label_object_ids)
    bronze, bronze_object_ids = _bronze_summary(bronze_dir)
    silver, silver_object_ids = _silver_summary(silver_path)
    epochs, epoch_object_ids = _epochs_summary(epochs_path)
    local_experts = _local_expert_summary(local_expert_path)
    checks = _build_checks(
        label_object_ids=label_object_ids,
        lc_object_ids=lc_object_ids,
        bronze_object_ids=bronze_object_ids,
        silver_object_ids=silver_object_ids,
        epoch_object_ids=epoch_object_ids,
    )

    summary = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "data_root": str(data_root.resolve()),
        "labels": labels,
        "lightcurves": lightcurves,
        "bronze": bronze,
        "silver": silver,
        "local_expert_outputs": local_experts,
        "epochs": epochs,
        "checks": checks,
    }

    _print_summary(summary)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print("")
        print(f"Wrote JSON summary → {out_path}")

    if args.strict and not checks["overall_ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
