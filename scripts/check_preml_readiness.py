"""Report whether the repository is ready for pre-ML science work."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from audit_data_readiness import audit_data_readiness
from debass_meta.access.lasair import LasairAdapter
from debass_meta.experts.local.parsnip import ParSNIPExpert
from debass_meta.experts.local.supernnova import SuperNNovaExpert
from debass_meta.projectors import PHASE1_EXPERT_KEYS, sanitize_expert_key


def _phase1_snapshot_experts(report: dict[str, Any]) -> list[str]:
    available = report.get("snapshot_available_true", {}) or {}
    experts: list[str] = []
    for expert_key in PHASE1_EXPERT_KEYS:
        column = f"avail__{sanitize_expert_key(expert_key)}"
        if int(available.get(column, 0)) > 0:
            experts.append(expert_key)
    return experts


def _sample_object_ids(labels_path: Path | None, limit: int = 5) -> list[str]:
    if labels_path is None or not labels_path.exists():
        return []
    object_ids: list[str] = []
    with open(labels_path) as fh:
        for row in csv.DictReader(fh):
            object_id = str(row.get("object_id") or "").strip()
            if not object_id:
                continue
            object_ids.append(object_id)
            if len(object_ids) >= limit:
                break
    return object_ids


def _probe_lasair_dataset(lasair: LasairAdapter, labels_path: Path | None) -> dict[str, Any]:
    sample_ids = _sample_object_ids(labels_path)
    if not sample_ids:
        return {"status": "no_sample_objects", "sample_object_ids": []}

    sample_results: list[dict[str, Any]] = []
    for object_id in sample_ids:
        out = lasair.fetch_object(object_id)
        sample_results.append({
            "object_id": object_id,
            "availability": bool(out.availability),
            "fixture_used": bool(out.fixture_used),
            "field_count": len(out.fields),
            "source_endpoint": out.source_endpoint,
            "request_params": out.request_params,
        })

    any_live_payload = any(item["availability"] for item in sample_results)
    any_live_fields = any(item["availability"] and item["field_count"] > 0 for item in sample_results)
    return {
        "status": "ok" if any_live_fields else "no_live_fields" if any_live_payload else "no_live_payload",
        "sample_object_ids": sample_ids,
        "sample_results": sample_results,
    }


def evaluate_preml_readiness(
    *,
    root: Path,
    labels_path: Path | None = None,
    lightcurve_dir: Path | None = None,
) -> dict[str, Any]:
    report = audit_data_readiness(root=root, labels_path=labels_path, lightcurve_dir=lightcurve_dir)

    phase1_safe_experts = _phase1_snapshot_experts(report)
    truth_quality = report.get("truth_label_quality", {}) or {}
    spectroscopic_rows = int(truth_quality.get("spectroscopic", 0))
    consensus_rows = int(truth_quality.get("consensus", 0))
    strong_truth_rows = int(truth_quality.get("strong", 0)) + spectroscopic_rows
    weak_truth_rows = int(truth_quality.get("weak", 0))

    lasair = LasairAdapter()
    lasair_probe = lasair.probe()
    lasair_dataset_probe = _probe_lasair_dataset(lasair, labels_path)
    parsnip_meta = ParSNIPExpert().metadata()
    supernnova_meta = SuperNNovaExpert().metadata()

    trust_training_blockers: list[str] = []
    full_phase1_blockers: list[str] = []

    if strong_truth_rows == 0 and consensus_rows == 0:
        trust_training_blockers.append("no strong curated truth is available")
        full_phase1_blockers.append("no strong curated truth is available")
    elif strong_truth_rows == 0:
        # consensus rows can unblock trust training but full phase1 wants spectroscopic
        full_phase1_blockers.append("no spectroscopic truth available (consensus-only)")
    if not phase1_safe_experts:
        trust_training_blockers.append("no phase-1 experts have epoch-safe snapshot coverage")
    if "fink/snn" not in phase1_safe_experts or "fink/rf_ia" not in phase1_safe_experts:
        trust_training_blockers.append("safe Fink expert coverage is incomplete")

    if lasair_probe.get("status") != "ok":
        full_phase1_blockers.append(f"Lasair live access unavailable: {lasair_probe.get('reason') or lasair_probe.get('status')}")
    elif lasair_dataset_probe.get("status") not in {"ok", "no_sample_objects"}:
        full_phase1_blockers.append(
            "Lasair dataset probe returned no live Sherlock context for sampled object_ids"
        )
    if not parsnip_meta.get("model_loaded"):
        full_phase1_blockers.append("ParSNIP live model artifacts are not installed")
    if not supernnova_meta.get("available"):
        full_phase1_blockers.append("SuperNNova package is not installed")
    elif not supernnova_meta.get("inference_implemented"):
        full_phase1_blockers.append("SuperNNova inference wrapper is not implemented")

    inference_only_experts: list[str] = []
    silver_live = report.get("silver_live_available_by_expert", {}) or {}
    snapshot_available = set(phase1_safe_experts)
    for expert_key in PHASE1_EXPERT_KEYS:
        if int(silver_live.get(expert_key, 0)) > 0 and expert_key not in snapshot_available:
            inference_only_experts.append(expert_key)

    return {
        "ready_for_trust_training_now": not trust_training_blockers,
        "ready_for_full_phase1_goal": not full_phase1_blockers,
        "truth_status": {
            "spectroscopic_rows": spectroscopic_rows,
            "consensus_rows": consensus_rows,
            "strong_rows": strong_truth_rows,
            "weak_rows": weak_truth_rows,
        },
        "phase1_safe_snapshot_experts": phase1_safe_experts,
        "phase1_inference_only_experts": inference_only_experts,
        "lasair_probe": lasair_probe,
        "lasair_dataset_probe": lasair_dataset_probe,
        "local_experts": {
            "parsnip": {
                "available": parsnip_meta.get("available"),
                "model_loaded": parsnip_meta.get("model_loaded"),
                "model_path": parsnip_meta.get("model_path"),
                "classifier_path": parsnip_meta.get("classifier_path"),
            },
            "supernnova": {
                "available": supernnova_meta.get("available"),
                "model_loaded": supernnova_meta.get("model_loaded"),
                "inference_implemented": supernnova_meta.get("inference_implemented"),
                "model_dir": supernnova_meta.get("model_dir"),
            },
        },
        "trust_training_blockers": trust_training_blockers,
        "full_phase1_blockers": full_phase1_blockers,
        "readiness_report": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check pre-ML readiness for DEBASS")
    parser.add_argument("--root", default=".")
    parser.add_argument("--labels", default=None)
    parser.add_argument("--lightcurves-dir", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    result = evaluate_preml_readiness(
        root=root,
        labels_path=Path(args.labels) if args.labels else None,
        lightcurve_dir=Path(args.lightcurves_dir) if args.lightcurves_dir else None,
    )

    out_path = Path(args.json_out) if args.json_out else root / "reports/summary/preml_readiness.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"Wrote pre-ML readiness report → {out_path}")
    print(f"ready_for_trust_training_now={result['ready_for_trust_training_now']}")
    print(f"ready_for_full_phase1_goal={result['ready_for_full_phase1_goal']}")

    if result["ready_for_trust_training_now"] and result["ready_for_full_phase1_goal"]:
        return
    raise SystemExit(1)


if __name__ == "__main__":
    main()
