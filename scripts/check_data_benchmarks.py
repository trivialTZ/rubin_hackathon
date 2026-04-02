"""Validate run artifacts against editable data-engineering benchmarks."""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from audit_data_readiness import audit_data_readiness
from debass_meta.projectors import ALL_EXPERT_KEYS, PHASE1_EXPERT_KEYS, sanitize_expert_key


DEFAULT_BENCHMARKS: dict[str, Any] = {
    "readiness": {
        "require_silver": True,
        "require_truth": True,
        "require_snapshots": True,
        "require_followup_model": True,
        "require_complete_lightcurves": True,
        "allow_weak_truth": True,
        "require_strong_truth": False,
        "require_exact_alert_alert_id": True,
        "require_exact_alert_n_det": True,
        "require_exact_alert_event_time_jd": True,
        "min_label_rows": 1,
        "min_snapshot_rows": 1,
        "min_trained_experts": 1,
        "require_live_brokers": ["alerce", "fink"],
        "require_snapshot_experts": ["fink/snn", "fink/rf_ia"],
        "require_trust_models": ["fink/snn", "fink/rf_ia"],
    },
    "payload": {
        "enabled": True,
        "scores_path": "reports/scores_ndet4.jsonl",
        "min_rows": 1,
        "require_all_phase1_experts": True,
        "require_contract_fields": True,
        "required_available_experts": ["fink/snn", "fink/rf_ia"],
    },
    "metrics": {
        "expert_trust": {
            "enabled": True,
            "path": "reports/metrics/expert_trust_metrics.json",
            "require_metrics_for": ["fink/snn", "fink/rf_ia"],
            "min_test_rows": 1,
        },
        "followup": {
            "enabled": True,
            "path": "reports/metrics/followup_metrics.json",
            "min_test_rows": 1,
        },
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_benchmark_spec(path: Path | None = None) -> dict[str, Any]:
    spec = deepcopy(DEFAULT_BENCHMARKS)
    if path is None:
        return spec
    with open(path, "rb") as fh:
        loaded = tomllib.load(fh)
    return _deep_merge(spec, loaded)


def _resolve_path(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def _load_jsonl(path: Path | None) -> list[dict[str, Any]] | None:
    if path is None or not path.exists():
        return None
    rows: list[dict[str, Any]] = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno} of {path}: {exc}") from exc
    return rows


def _check_metric_thresholds(
    *,
    prefix: str,
    metrics: dict[str, Any],
    spec: dict[str, Any],
    failures: list[str],
) -> None:
    if spec.get("min_test_rows") is not None and metrics.get("n_rows", 0) < int(spec["min_test_rows"]):
        failures.append(
            f"{prefix} has {metrics.get('n_rows', 0)} test rows; need at least {int(spec['min_test_rows'])}"
        )
    if spec.get("min_roc_auc") is not None:
        roc_auc = metrics.get("roc_auc")
        if roc_auc is None or float(roc_auc) < float(spec["min_roc_auc"]):
            failures.append(
                f"{prefix} roc_auc={roc_auc!r}; need at least {float(spec['min_roc_auc'])}"
            )
    if spec.get("max_brier") is not None:
        brier = metrics.get("brier")
        if brier is None or float(brier) > float(spec["max_brier"]):
            failures.append(
                f"{prefix} brier={brier!r}; need at most {float(spec['max_brier'])}"
            )


def _evaluate_readiness(report: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    readiness = spec.get("readiness", {})

    if readiness.get("require_silver", False) and not report.get("silver_present"):
        failures.append("silver broker_events.parquet is missing")
    if readiness.get("require_truth", False) and not report.get("truth_present"):
        failures.append("truth table is missing")
    if readiness.get("require_snapshots", False) and not report.get("snapshots_present"):
        failures.append("object_epoch_snapshots.parquet is missing")
    if readiness.get("require_followup_model", False) and not report.get("followup_model"):
        failures.append("follow-up model metadata is missing")
    if readiness.get("require_complete_lightcurves", False):
        missing = report.get("label_objects_missing_lightcurves") or []
        if missing:
            failures.append(f"{len(missing)} labelled objects are missing cached lightcurves")
    if readiness.get("min_label_rows") is not None and int(report.get("label_rows") or 0) < int(readiness["min_label_rows"]):
        failures.append(
            f"label_rows={report.get('label_rows')!r}; need at least {int(readiness['min_label_rows'])}"
        )
    if readiness.get("min_snapshot_rows") is not None and int(report.get("snapshot_rows") or 0) < int(readiness["min_snapshot_rows"]):
        failures.append(
            f"snapshot_rows={report.get('snapshot_rows')!r}; need at least {int(readiness['min_snapshot_rows'])}"
        )
    if not readiness.get("allow_weak_truth", True) and int((report.get("truth_label_quality") or {}).get("weak", 0)) > 0:
        failures.append("weak truth rows are present but benchmarks forbid them")
    if readiness.get("require_strong_truth", False) and int((report.get("truth_label_quality") or {}).get("strong", 0)) == 0:
        failures.append("no strong truth rows are present")
    if readiness.get("require_exact_alert_alert_id", False):
        exact_rows = int(report.get("exact_alert_rows", 0))
        exact_with_alert_id = int(report.get("exact_alert_rows_with_alert_id", 0))
        if exact_rows > 0 and exact_with_alert_id < exact_rows:
            failures.append(
                f"exact-alert alert_id coverage is {exact_with_alert_id}/{exact_rows}; need full coverage"
            )
    if readiness.get("require_exact_alert_n_det", False) and int(report.get("exact_alert_rows_missing_n_det", 0)) > 0:
        failures.append(
            f"exact-alert rows missing n_det: {int(report.get('exact_alert_rows_missing_n_det', 0))}"
        )
    if readiness.get("require_exact_alert_event_time_jd", False) and int(report.get("exact_alert_rows_missing_event_time_jd", 0)) > 0:
        failures.append(
            f"exact-alert rows missing event_time_jd: {int(report.get('exact_alert_rows_missing_event_time_jd', 0))}"
        )

    bronze_live_rows = report.get("bronze_live_available_rows", {})
    for broker in readiness.get("require_live_brokers", []):
        if int(bronze_live_rows.get(broker, 0)) <= 0:
            failures.append(f"broker {broker} has no live, non-fixture bronze rows")

    snapshot_available = report.get("snapshot_available_true", {})
    for expert_key in readiness.get("require_snapshot_experts", []):
        avail_col = f"avail__{sanitize_expert_key(expert_key)}"
        if int(snapshot_available.get(avail_col, 0)) <= 0:
            failures.append(f"expert {expert_key} is never available in object-epoch snapshots")

    trust_models = report.get("trust_models") or {}
    trained = set(str(value) for value in trust_models.get("experts", []))
    if readiness.get("min_trained_experts") is not None and len(trained) < int(readiness["min_trained_experts"]):
        failures.append(
            f"trained_experts={len(trained)}; need at least {int(readiness['min_trained_experts'])}"
        )
    for expert_key in readiness.get("require_trust_models", []):
        if expert_key not in trained:
            failures.append(f"trust model missing for expert {expert_key}")

    return failures


def _evaluate_metrics(
    *,
    root: Path,
    spec: dict[str, Any],
    expert_metrics_path: Path | None = None,
    followup_metrics_path: Path | None = None,
) -> list[str]:
    failures: list[str] = []
    metrics_spec = spec.get("metrics", {})

    expert_spec = metrics_spec.get("expert_trust", {})
    if expert_spec.get("enabled", True):
        expert_path = expert_metrics_path or _resolve_path(root, expert_spec.get("path"))
        expert_metrics = _load_json(expert_path)
        if expert_spec.get("require_metrics_for"):
            if expert_metrics is None:
                failures.append(f"expert trust metrics file is missing: {expert_path}")
            else:
                per_expert = expert_spec.get("per_expert", {})
                for expert_key in expert_spec.get("require_metrics_for", []):
                    metrics = expert_metrics.get(expert_key)
                    if metrics is None:
                        failures.append(f"expert trust metrics missing block for {expert_key}")
                        continue
                    effective_spec = {
                        key: expert_spec[key]
                        for key in ("min_test_rows", "min_roc_auc", "max_brier")
                        if key in expert_spec
                    }
                    effective_spec.update(per_expert.get(expert_key, {}))
                    _check_metric_thresholds(
                        prefix=f"expert_trust[{expert_key}]",
                        metrics=metrics,
                        spec=effective_spec,
                        failures=failures,
                    )

    followup_spec = metrics_spec.get("followup", {})
    if followup_spec.get("enabled", True):
        follow_path = followup_metrics_path or _resolve_path(root, followup_spec.get("path"))
        followup_metrics = _load_json(follow_path)
        if any(key in followup_spec for key in ("min_test_rows", "min_roc_auc", "max_brier")):
            if followup_metrics is None:
                failures.append(f"follow-up metrics file is missing: {follow_path}")
            else:
                _check_metric_thresholds(
                    prefix="followup",
                    metrics=followup_metrics,
                    spec=followup_spec,
                    failures=failures,
                )

    return failures


def _evaluate_payload(
    *,
    root: Path,
    spec: dict[str, Any],
    scores_path: Path | None = None,
) -> tuple[list[str], int]:
    failures: list[str] = []
    payload_spec = spec.get("payload", {})
    if not payload_spec.get("enabled", True):
        return failures, 0
    path = scores_path or _resolve_path(root, payload_spec.get("scores_path"))
    rows = _load_jsonl(path)
    if rows is None:
        if payload_spec.get("min_rows") is not None or payload_spec.get("require_all_phase1_experts", False):
            failures.append(f"score payload file is missing: {path}")
        return failures, 0

    min_rows = payload_spec.get("min_rows")
    if min_rows is not None and len(rows) < int(min_rows):
        failures.append(f"score payload has {len(rows)} rows; need at least {int(min_rows)}")

    required_top_level = {"object_id", "n_det", "alert_jd", "expert_confidence", "ensemble"}
    required_block_fields = {"available", "trust", "prediction_type", "exactness"}
    availability_counts = {expert_key: 0 for expert_key in payload_spec.get("required_available_experts", [])}

    for idx, row in enumerate(rows):
        if payload_spec.get("require_contract_fields", False):
            missing = required_top_level - set(row.keys())
            if missing:
                failures.append(f"score row {idx} is missing top-level keys: {sorted(missing)}")
                continue

        expert_confidence = row.get("expert_confidence", {})
        if payload_spec.get("require_all_phase1_experts", False):
            missing_experts = [expert for expert in PHASE1_EXPERT_KEYS if expert not in expert_confidence]
            if missing_experts:
                failures.append(f"score row {idx} is missing expert blocks: {missing_experts}")
                continue

        if payload_spec.get("require_contract_fields", False):
            for expert_key, block in expert_confidence.items():
                if not isinstance(block, dict):
                    failures.append(f"score row {idx} expert {expert_key} block is not an object")
                    continue
                missing_fields = required_block_fields - set(block.keys())
                if missing_fields:
                    failures.append(
                        f"score row {idx} expert {expert_key} is missing fields: {sorted(missing_fields)}"
                    )

        for expert_key in availability_counts:
            block = expert_confidence.get(expert_key)
            if isinstance(block, dict) and bool(block.get("available")):
                availability_counts[expert_key] += 1

    for expert_key, count in availability_counts.items():
        if count <= 0:
            failures.append(f"score payload never marks {expert_key} as available")

    return failures, len(rows)


def evaluate_run(
    *,
    root: Path,
    benchmarks: dict[str, Any],
    labels_path: Path | None = None,
    lightcurve_dir: Path | None = None,
    scores_path: Path | None = None,
    expert_metrics_path: Path | None = None,
    followup_metrics_path: Path | None = None,
) -> dict[str, Any]:
    readiness = audit_data_readiness(
        root=root,
        labels_path=labels_path,
        lightcurve_dir=lightcurve_dir,
    )
    failures = _evaluate_readiness(readiness, benchmarks)
    failures.extend(
        _evaluate_metrics(
            root=root,
            spec=benchmarks,
            expert_metrics_path=expert_metrics_path,
            followup_metrics_path=followup_metrics_path,
        )
    )
    payload_failures, score_rows = _evaluate_payload(
        root=root,
        spec=benchmarks,
        scores_path=scores_path,
    )
    failures.extend(payload_failures)
    return {
        "passed": not failures,
        "failures": failures,
        "score_rows": score_rows,
        "readiness": readiness,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check DEBASS data-engineering benchmarks")
    parser.add_argument("--root", default=".")
    parser.add_argument("--benchmarks", default="benchmarks/data_engineering.toml")
    parser.add_argument("--labels", default=None)
    parser.add_argument("--lightcurves-dir", default=None)
    parser.add_argument("--scores", default=None)
    parser.add_argument("--expert-trust-metrics", default=None)
    parser.add_argument("--followup-metrics", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    benchmark_path = Path(args.benchmarks) if args.benchmarks else None
    benchmarks = load_benchmark_spec(benchmark_path if benchmark_path and benchmark_path.exists() else None)

    root = Path(args.root)
    result = evaluate_run(
        root=root,
        benchmarks=benchmarks,
        labels_path=Path(args.labels) if args.labels else None,
        lightcurve_dir=Path(args.lightcurves_dir) if args.lightcurves_dir else None,
        scores_path=Path(args.scores) if args.scores else None,
        expert_metrics_path=Path(args.expert_trust_metrics) if args.expert_trust_metrics else None,
        followup_metrics_path=Path(args.followup_metrics) if args.followup_metrics else None,
    )

    json_out = Path(args.json_out) if args.json_out else root / "reports/summary/benchmark_check.json"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, "w") as fh:
        json.dump(result, fh, indent=2)

    if result["passed"]:
        print(f"Benchmark check passed → {json_out}")
        return

    print(f"Benchmark check failed → {json_out}")
    for failure in result["failures"]:
        print(f" - {failure}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
