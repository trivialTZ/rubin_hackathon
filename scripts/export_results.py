#!/usr/bin/env python3
"""Export publishable results from a completed pipeline run.

Copies key metrics, a sample score payload, and split metadata into
`published_results/` — a git-tracked directory suitable for public release.
Does NOT export full data, model weights, or lightcurves.

Usage (on SCC after pipeline completes):
    python3 scripts/export_results.py

Output:
    published_results/
        phase1_metrics.json           — all test-set metrics in one file
        sample_scores_ndet3.jsonl     — first 50 score payloads at n_det=3
        sample_scores_ndet5.jsonl     — first 50 score payloads at n_det=5
        split_summary.json            — train/cal/test counts (not IDs)
        expert_coverage.json          — per-expert availability at key epochs
        results_summary.txt           — human-readable summary
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS_DIR = Path("published_results")
SAMPLE_SIZE = 50  # score payloads to include


def _safe_load_json(path: Path) -> dict | None:
    if not path.exists():
        print(f"  SKIP {path} (not found)")
        return None
    with open(path) as fh:
        return json.load(fh)


def _safe_read_text(path: Path) -> str | None:
    if not path.exists():
        print(f"  SKIP {path} (not found)")
        return None
    return path.read_text()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Aggregate all metrics into one file ───────────────────────────
    print("Exporting metrics...")
    metrics = {}

    trust = _safe_load_json(Path("reports/metrics/expert_trust_metrics.json"))
    if trust:
        metrics["expert_trust"] = trust

    followup = _safe_load_json(Path("reports/metrics/followup_metrics.json"))
    if followup:
        metrics["followup"] = followup

    early = _safe_load_json(Path("reports/metrics/early_meta_metrics.json"))
    if early:
        metrics["baseline_early_meta"] = early

    # Add run metadata
    try:
        import pandas as pd
        snap_path = Path("data/gold/object_epoch_snapshots_trust.parquet")
        if not snap_path.exists():
            snap_path = Path("data/gold/object_epoch_snapshots.parquet")
        if snap_path.exists():
            df = pd.read_parquet(snap_path)
            metrics["dataset"] = {
                "n_objects": int(df["object_id"].nunique()),
                "n_epoch_rows": int(len(df)),
                "n_det_range": [int(df["n_det"].min()), int(df["n_det"].max())],
            }
            if "label_quality" in df.columns:
                quality = df.groupby("object_id")["label_quality"].first().value_counts()
                metrics["dataset"]["truth_quality"] = {
                    str(k): int(v) for k, v in quality.items()
                }
            if "target_class" in df.columns:
                classes = df.groupby("object_id")["target_class"].first().value_counts()
                metrics["dataset"]["class_distribution"] = {
                    str(k): int(v) for k, v in classes.items()
                }
    except Exception as exc:
        print(f"  Warning: could not read snapshots: {exc}")

    with open(RESULTS_DIR / "phase1_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"  → {RESULTS_DIR / 'phase1_metrics.json'}")

    # ── 2. Sample score payloads ─────────────────────────────────────────
    print("Exporting sample score payloads...")
    for n_det in [3, 5]:
        src = Path(f"reports/scores/scores_ndet{n_det}.jsonl")
        if not src.exists():
            print(f"  SKIP {src}")
            continue
        with open(src) as fh:
            lines = [fh.readline() for _ in range(SAMPLE_SIZE)]
            lines = [l for l in lines if l.strip()]
        dst = RESULTS_DIR / f"sample_scores_ndet{n_det}.jsonl"
        with open(dst, "w") as fh:
            fh.writelines(lines)
        print(f"  → {dst} ({len(lines)} payloads)")

    # ── 3. Split summary (counts only, not object IDs) ───────────────────
    print("Exporting split summary...")
    meta = _safe_load_json(Path("models/trust/metadata.json"))
    if meta:
        split_summary = {
            "n_train": len(meta.get("train_ids", [])),
            "n_cal": len(meta.get("cal_ids", [])),
            "n_test": len(meta.get("test_ids", [])),
            "experts_trained": meta.get("experts", []),
            "contains_weak_labels": meta.get("contains_weak_labels"),
            "allow_unsafe_alerce": meta.get("allow_unsafe_alerce"),
        }
        with open(RESULTS_DIR / "split_summary.json", "w") as fh:
            json.dump(split_summary, fh, indent=2)
        print(f"  → {RESULTS_DIR / 'split_summary.json'}")

    # ── 4. Expert coverage table ─────────────────────────────────────────
    print("Exporting expert coverage...")
    summary = _safe_load_json(Path("reports/results_summary.json"))
    if summary and "temporal_evolution" in summary:
        coverage = {}
        for n_det_str, entry in summary["temporal_evolution"].items():
            coverage[n_det_str] = {
                "n_objects": entry.get("n_objects"),
                "experts_available": entry.get("experts_available"),
                "mean_trust": entry.get("mean_trust"),
            }
        with open(RESULTS_DIR / "expert_coverage.json", "w") as fh:
            json.dump(coverage, fh, indent=2)
        print(f"  → {RESULTS_DIR / 'expert_coverage.json'}")

    # ── 5. Human-readable summary ────────────────────────────────────────
    print("Exporting summary text...")
    txt = _safe_read_text(Path("reports/results_summary.txt"))
    if txt:
        (RESULTS_DIR / "results_summary.txt").write_text(txt)
        print(f"  → {RESULTS_DIR / 'results_summary.txt'}")

    print(f"\nDone. Commit published_results/ and push to make public.")
    print(f"  git add published_results/")
    print(f"  git commit -m 'Add Phase 1 results (test-set metrics + sample scores)'")
    print(f"  git push origin main")


if __name__ == "__main__":
    main()
