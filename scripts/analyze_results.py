#!/usr/bin/env python3
"""Analyze DEBASS pipeline results and produce a science-ready summary.

Generates:
  reports/results_summary.json   — Machine-readable metrics digest
  reports/results_summary.txt    — Human-readable summary for the hackathon

Designed to run after the full pipeline completes on SCC:
  python3 scripts/analyze_results.py [--strict-labels]

Without --strict-labels, includes all truth tiers.
With --strict-labels, reports only spectroscopic-confirmed objects.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from debass_meta.projectors import PHASE1_EXPERT_KEYS, sanitize_expert_key


# ── helpers ──────────────────────────────────────────────────────────────────

def _safe_auc(y_true, y_prob):
    from sklearn.metrics import roc_auc_score
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return None


def _pct(num, denom):
    return f"{100 * num / denom:.1f}%" if denom > 0 else "n/a"


# ── Expert availability & trust per n_det ────────────────────────────────────

def expert_availability_table(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    """For each expert × n_det, count available rows and mean trust."""
    records = []
    for expert_key in PHASE1_EXPERT_KEYS:
        san = sanitize_expert_key(expert_key)
        avail_col = f"avail__{san}"
        trust_col = f"q__{san}"
        if avail_col not in snapshot_df.columns:
            continue
        for n_det, group in snapshot_df.groupby("n_det"):
            n_total = len(group)
            n_avail = int(group[avail_col].fillna(0).astype(bool).sum())
            mean_trust = None
            if trust_col in group.columns and n_avail > 0:
                trust_vals = group.loc[group[avail_col].fillna(0).astype(bool), trust_col].dropna()
                if len(trust_vals) > 0:
                    mean_trust = float(trust_vals.mean())
            records.append({
                "expert": expert_key,
                "n_det": int(n_det),
                "n_total": n_total,
                "n_available": n_avail,
                "coverage_pct": round(100 * n_avail / n_total, 1) if n_total > 0 else 0,
                "mean_trust": round(mean_trust, 4) if mean_trust is not None else None,
            })
    return pd.DataFrame(records)


# ── Classification accuracy per expert ───────────────────────────────────────

def expert_accuracy_table(helpfulness_df: pd.DataFrame) -> pd.DataFrame:
    """Per-expert accuracy on spectroscopic truth."""
    records = []
    for expert_key in sorted(helpfulness_df["expert_key"].dropna().unique()):
        rows = helpfulness_df[helpfulness_df["expert_key"] == expert_key]
        rows = rows[rows["is_topclass_correct"].notna()]
        if len(rows) == 0:
            continue
        n_correct = int(rows["is_topclass_correct"].sum())
        n_total = len(rows)
        # By class
        by_class = {}
        if "target_class" in rows.columns:
            for cls, cls_group in rows.groupby("target_class"):
                cls_n = len(cls_group)
                cls_correct = int(cls_group["is_topclass_correct"].sum())
                by_class[str(cls)] = {
                    "n": cls_n,
                    "accuracy": round(cls_correct / cls_n, 3) if cls_n > 0 else None,
                }
        records.append({
            "expert": expert_key,
            "n_evaluated": n_total,
            "accuracy": round(n_correct / n_total, 3),
            "n_correct": n_correct,
            "by_class": by_class,
        })
    return records


# ── Follow-up recommendation analysis ────────────────────────────────────────

def followup_analysis(scores_path: Path) -> dict:
    """Analyze follow-up recommendations from JSONL scores."""
    if not scores_path.exists():
        return {"error": "scores file not found"}
    payloads = []
    with open(scores_path) as fh:
        for line in fh:
            if line.strip():
                payloads.append(json.loads(line))
    if not payloads:
        return {"n_scored": 0}

    n_scored = len(payloads)
    p_follows = [p["ensemble"]["p_follow_proxy"] for p in payloads if p["ensemble"]["p_follow_proxy"] is not None]
    recommended = [p for p in payloads if p["ensemble"].get("recommended")]
    not_recommended = [p for p in payloads if p["ensemble"].get("recommended") is False]

    result = {
        "n_scored": n_scored,
        "n_with_followup_score": len(p_follows),
        "n_recommended": len(recommended),
        "n_not_recommended": len(not_recommended),
        "recommend_rate": round(len(recommended) / n_scored, 3) if n_scored > 0 else None,
    }
    if p_follows:
        result["p_follow_mean"] = round(float(np.mean(p_follows)), 4)
        result["p_follow_median"] = round(float(np.median(p_follows)), 4)
        result["p_follow_std"] = round(float(np.std(p_follows)), 4)
        # Distribution bins
        bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
        labels = ["<0.1", "0.1-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", ">0.9"]
        hist, _ = np.histogram(p_follows, bins=bins)
        result["p_follow_distribution"] = {label: int(count) for label, count in zip(labels, hist)}

    # Expert availability summary across all scored objects
    expert_avail_counts = defaultdict(int)
    expert_trust_sums = defaultdict(float)
    expert_trust_counts = defaultdict(int)
    for p in payloads:
        for expert_key, block in p.get("expert_confidence", {}).items():
            if block.get("available"):
                expert_avail_counts[expert_key] += 1
                if block.get("trust") is not None:
                    expert_trust_sums[expert_key] += block["trust"]
                    expert_trust_counts[expert_key] += 1

    result["expert_availability"] = {
        expert_key: {
            "n_available": expert_avail_counts[expert_key],
            "coverage_pct": round(100 * expert_avail_counts[expert_key] / n_scored, 1),
            "mean_trust": round(expert_trust_sums[expert_key] / expert_trust_counts[expert_key], 4)
                if expert_trust_counts[expert_key] > 0 else None,
        }
        for expert_key in PHASE1_EXPERT_KEYS
    }
    return result


# ── Truth quality breakdown ──────────────────────────────────────────────────

def truth_quality_summary(snapshot_df: pd.DataFrame) -> dict:
    """Summarize truth label quality distribution."""
    result = {"n_objects": int(snapshot_df["object_id"].nunique())}
    if "label_quality" in snapshot_df.columns:
        quality_counts = (
            snapshot_df.groupby("object_id")["label_quality"]
            .first()
            .value_counts()
            .to_dict()
        )
        result["by_quality"] = {str(k): int(v) for k, v in quality_counts.items()}
    if "label_source" in snapshot_df.columns:
        source_counts = (
            snapshot_df.groupby("object_id")["label_source"]
            .first()
            .value_counts()
            .to_dict()
        )
        result["by_source"] = {str(k): int(v) for k, v in source_counts.items()}
    if "target_class" in snapshot_df.columns:
        class_counts = (
            snapshot_df.groupby("object_id")["target_class"]
            .first()
            .value_counts()
            .to_dict()
        )
        result["by_class"] = {str(k): int(v) for k, v in class_counts.items()}
    return result


# ── Temporal evolution summary ───────────────────────────────────────────────

def temporal_evolution(snapshot_df: pd.DataFrame) -> dict:
    """How expert coverage and trust evolve with n_det."""
    n_det_summary = {}
    for n_det, group in snapshot_df.groupby("n_det"):
        n_det_key = int(n_det)
        entry = {"n_objects": len(group)}
        # Count experts available
        expert_counts = {}
        for expert_key in PHASE1_EXPERT_KEYS:
            san = sanitize_expert_key(expert_key)
            avail_col = f"avail__{san}"
            if avail_col in group.columns:
                expert_counts[expert_key] = int(group[avail_col].fillna(0).astype(bool).sum())
        entry["experts_available"] = expert_counts
        # Mean trust per expert
        trust_means = {}
        for expert_key in PHASE1_EXPERT_KEYS:
            san = sanitize_expert_key(expert_key)
            trust_col = f"q__{san}"
            avail_col = f"avail__{san}"
            if trust_col in group.columns and avail_col in group.columns:
                mask = group[avail_col].fillna(0).astype(bool)
                vals = group.loc[mask, trust_col].dropna()
                if len(vals) > 0:
                    trust_means[expert_key] = round(float(vals.mean()), 4)
        entry["mean_trust"] = trust_means
        n_det_summary[n_det_key] = entry
    return n_det_summary


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze DEBASS pipeline results")
    parser.add_argument("--snapshots", default="data/gold/object_epoch_snapshots_trust.parquet")
    parser.add_argument("--snapshots-raw", default="data/gold/object_epoch_snapshots.parquet")
    parser.add_argument("--helpfulness", default="data/gold/expert_helpfulness.parquet")
    parser.add_argument("--trust-metrics", default="reports/metrics/expert_trust_metrics.json")
    parser.add_argument("--followup-metrics", default="reports/metrics/followup_metrics.json")
    parser.add_argument("--scores", default=None, help="JSONL scores file (from score_nightly.py)")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--strict-labels", action="store_true",
                        help="Only report on spectroscopic-confirmed objects")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {"pipeline": "DEBASS trust-aware meta-classifier", "phase": "Phase 1 (CPU)"}

    # ── Load data ────────────────────────────────────────────────────────
    snapshot_path = Path(args.snapshots)
    if not snapshot_path.exists():
        # Fall back to raw snapshots if trust-augmented not available
        snapshot_path = Path(args.snapshots_raw)
    if not snapshot_path.exists():
        print(f"ERROR: No snapshot file found at {args.snapshots} or {args.snapshots_raw}")
        sys.exit(1)

    snapshot_df = pd.read_parquet(snapshot_path)
    print(f"Loaded {len(snapshot_df)} snapshot rows ({snapshot_df['object_id'].nunique()} objects)")

    if args.strict_labels and "label_quality" in snapshot_df.columns:
        snapshot_df = snapshot_df[snapshot_df["label_quality"] == "spectroscopic"]
        print(f"  Strict-labels filter: {len(snapshot_df)} rows ({snapshot_df['object_id'].nunique()} objects)")

    summary["snapshot_file"] = str(snapshot_path)
    summary["n_snapshot_rows"] = int(len(snapshot_df))
    summary["n_objects"] = int(snapshot_df["object_id"].nunique())
    summary["n_det_range"] = [int(snapshot_df["n_det"].min()), int(snapshot_df["n_det"].max())]

    # ── Truth quality ────────────────────────────────────────────────────
    summary["truth"] = truth_quality_summary(snapshot_df)
    print(f"\nTruth: {summary['truth']}")

    # ── Expert availability per n_det ────────────────────────────────────
    avail_df = expert_availability_table(snapshot_df)
    if len(avail_df) > 0:
        summary["expert_availability_sample"] = (
            avail_df[avail_df["n_det"].isin([3, 5, 10, 15, 20])]
            .to_dict(orient="records")
        )
        print(f"\nExpert availability at n_det=3:")
        for _, row in avail_df[avail_df["n_det"] == 3].iterrows():
            print(f"  {row['expert']:40s}  {row['coverage_pct']:5.1f}%  trust={row['mean_trust']}")

    # ── Expert accuracy (helpfulness) ────────────────────────────────────
    helpfulness_path = Path(args.helpfulness)
    if helpfulness_path.exists():
        helpfulness_df = pd.read_parquet(helpfulness_path)
        if args.strict_labels and "label_quality" in helpfulness_df.columns:
            helpfulness_df = helpfulness_df[helpfulness_df["label_quality"] == "spectroscopic"]
        accuracy_records = expert_accuracy_table(helpfulness_df)
        summary["expert_accuracy"] = accuracy_records
        print(f"\nExpert accuracy (on truth):")
        for rec in accuracy_records:
            print(f"  {rec['expert']:40s}  acc={rec['accuracy']:.3f}  (n={rec['n_evaluated']})")

    # ── Trust model metrics ──────────────────────────────────────────────
    trust_metrics_path = Path(args.trust_metrics)
    if trust_metrics_path.exists():
        with open(trust_metrics_path) as fh:
            trust_metrics = json.load(fh)
        summary["trust_metrics"] = trust_metrics
        print(f"\nExpert trust model metrics (test set):")
        for expert_key, m in trust_metrics.items():
            auc_str = f"{m['roc_auc']:.3f}" if m.get("roc_auc") is not None else "n/a"
            print(f"  {expert_key:40s}  ROC-AUC={auc_str}  Brier={m.get('brier', 'n/a'):.4f}  n_test={m.get('n_test_rows', '?')}")

    # ── Follow-up model metrics ──────────────────────────────────────────
    followup_metrics_path = Path(args.followup_metrics)
    if followup_metrics_path.exists():
        with open(followup_metrics_path) as fh:
            followup_metrics = json.load(fh)
        summary["followup_metrics"] = followup_metrics
        auc = followup_metrics.get("roc_auc")
        print(f"\nFollow-up model: ROC-AUC={auc:.3f}  Brier={followup_metrics.get('brier'):.4f}  n_test={followup_metrics.get('n_rows')}")

    # ── Temporal evolution ───────────────────────────────────────────────
    temporal = temporal_evolution(snapshot_df)
    summary["temporal_evolution"] = temporal
    print(f"\nTemporal evolution (objects per n_det):")
    for n_det in sorted(temporal.keys()):
        entry = temporal[n_det]
        experts_avail = sum(1 for v in entry["experts_available"].values() if v > 0)
        print(f"  n_det={n_det:2d}  objects={entry['n_objects']:5d}  experts_with_data={experts_avail}/7")

    # ── Score analysis (if JSONL available) ──────────────────────────────
    scores_path = None
    if args.scores:
        scores_path = Path(args.scores)
    else:
        # Auto-detect
        for candidate in [
            Path("reports/scores_ndet4.jsonl"),
            Path("reports/scores_ndet3.jsonl"),
            Path("reports/scores_ndet5.jsonl"),
        ]:
            if candidate.exists():
                scores_path = candidate
                break

    if scores_path and scores_path.exists():
        fu_analysis = followup_analysis(scores_path)
        summary["followup_scores"] = fu_analysis
        print(f"\nFollow-up score analysis ({scores_path.name}):")
        print(f"  Scored: {fu_analysis.get('n_scored', 0)}")
        print(f"  Recommended: {fu_analysis.get('n_recommended', 0)} ({_pct(fu_analysis.get('n_recommended', 0), fu_analysis.get('n_scored', 1))})")
        if "p_follow_distribution" in fu_analysis:
            print(f"  p_follow distribution: {fu_analysis['p_follow_distribution']}")

    # ── Write outputs ────────────────────────────────────────────────────
    json_out = output_dir / "results_summary.json"
    with open(json_out, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\nWrote JSON summary → {json_out}")

    # ── Human-readable summary ───────────────────────────────────────────
    txt_out = output_dir / "results_summary.txt"
    lines = []
    lines.append("=" * 72)
    lines.append("DEBASS Trust-Aware Meta-Classifier — Results Summary")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Objects:      {summary['n_objects']}")
    lines.append(f"Epoch rows:   {summary['n_snapshot_rows']}")
    lines.append(f"n_det range:  {summary['n_det_range'][0]} – {summary['n_det_range'][1]}")
    lines.append("")

    # Truth
    truth = summary.get("truth", {})
    lines.append("── Truth Labels ──")
    if "by_quality" in truth:
        for quality, count in sorted(truth["by_quality"].items()):
            lines.append(f"  {quality:20s}  {count:5d} objects")
    if "by_class" in truth:
        lines.append("")
        for cls, count in sorted(truth["by_class"].items()):
            lines.append(f"  {cls:20s}  {count:5d} objects")
    lines.append("")

    # Trust metrics
    if "trust_metrics" in summary:
        lines.append("── Expert Trust Models (test-set performance) ──")
        for expert_key, m in summary["trust_metrics"].items():
            auc_str = f"{m['roc_auc']:.3f}" if m.get("roc_auc") is not None else "  n/a"
            brier_str = f"{m['brier']:.4f}" if m.get("brier") is not None else " n/a"
            lines.append(f"  {expert_key:40s}  AUC={auc_str}  Brier={brier_str}  (n_test={m.get('n_test_rows', '?')})")
        lines.append("")

    # Follow-up
    if "followup_metrics" in summary:
        fm = summary["followup_metrics"]
        auc = fm.get("roc_auc")
        auc_str = f"{auc:.3f}" if auc is not None else "n/a"
        lines.append("── Follow-Up Recommendation Model ──")
        lines.append(f"  ROC-AUC:       {auc_str}")
        lines.append(f"  Brier:         {fm.get('brier', 'n/a')}")
        lines.append(f"  Positive rate: {fm.get('positive_rate', 'n/a')}")
        lines.append(f"  Test rows:     {fm.get('n_rows', 'n/a')}")
        lines.append("")

    # Key n_det availability
    lines.append("── Expert Coverage at Key Epochs ──")
    for n_det_target in [3, 5, 10]:
        if n_det_target in temporal:
            entry = temporal[n_det_target]
            lines.append(f"  n_det = {n_det_target}:")
            for expert_key in PHASE1_EXPERT_KEYS:
                n_avail = entry["experts_available"].get(expert_key, 0)
                n_total = entry["n_objects"]
                trust_val = entry["mean_trust"].get(expert_key)
                trust_str = f"trust={trust_val:.3f}" if trust_val is not None else "trust=n/a"
                lines.append(f"    {expert_key:38s}  {n_avail:5d}/{n_total}  ({_pct(n_avail, n_total):>6s})  {trust_str}")
            lines.append("")

    # Science significance
    lines.append("── Science Significance ──")
    lines.append("This pipeline demonstrates trust-aware fusion of heterogeneous")
    lines.append("transient classifiers for Rubin/LSST spectroscopic follow-up")
    lines.append("prioritisation. Key innovations:")
    lines.append("  1. Per-expert, per-epoch trust calibration (q_e,t)")
    lines.append("  2. No-leakage temporal snapshots (features at n_det=k use only k detections)")
    lines.append("  3. Native NaN handling — missing broker scores are not imputed")
    lines.append("  4. Three-way grouped split prevents object-level data leakage")
    lines.append("  5. Works with 3–5 detections — actionable before peak brightness")
    lines.append("")

    with open(txt_out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote text summary → {txt_out}")


if __name__ == "__main__":
    main()
