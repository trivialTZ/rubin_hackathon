#!/usr/bin/env python3
"""Generate DEBASS paper figures from the latency / metrics / models output.

Writes to paper/debass_aas/figures/:
  fig_latency_curve.pdf          AUC vs n_det, trust-weighted vs baselines
  fig_calibration.pdf            Reliability diagrams per trust head (raw vs calibrated)
  fig_purity_completeness.pdf    Purity-completeness curves at n_det=3/5/10
  fig_ensemble_gain.pdf          Δ-AUC(trust_weighted − best_single) vs n_det

Run on SCC with the trained models + reports/metrics present:
  python3 scripts/make_paper_figures.py
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGSIZE = (7, 4.5)


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def fig_latency_curve(latency_parquet: Path, out_dir: Path) -> None:
    df = pd.read_parquet(latency_parquet)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for method, color, marker in [
        ("trust_weighted", "#c0392b", "o"),
        ("mean_ensemble", "#2980b9", "s"),
        ("best_single", "#27ae60", "^"),
    ]:
        sub = df[df["method"] == method].sort_values("n_det")
        if sub.empty:
            continue
        ax.plot(sub["n_det"], sub["auc"], color=color, marker=marker, lw=1.8, label=method.replace("_", "-"))
        has_ci = sub["auc_lo"].notna() & sub["auc_hi"].notna()
        if has_ci.any():
            ax.fill_between(sub["n_det"][has_ci], sub["auc_lo"][has_ci], sub["auc_hi"][has_ci],
                            color=color, alpha=0.15)
    ax.axhline(0.9, color="gray", ls="--", lw=0.7, label="AUC = 0.9")
    ax.set_xlabel("n_det (detections)")
    ax.set_ylabel("AUC (P(SN Ia))")
    ax.set_title("DEBASS latency curve — 95% bootstrap CI")
    ax.set_ylim(0.35, 1.0)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, out_dir / "fig_latency_curve.pdf")


def fig_ensemble_gain(latency_parquet: Path, out_dir: Path) -> None:
    df = pd.read_parquet(latency_parquet)
    pivot = df.pivot_table(index="n_det", columns="method", values="auc")
    if "trust_weighted" not in pivot.columns or "best_single" not in pivot.columns:
        print("  [skip] ensemble gain — required columns missing")
        return
    delta_bs = pivot["trust_weighted"] - pivot["best_single"]
    delta_me = pivot["trust_weighted"] - pivot.get("mean_ensemble", 0)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(pivot.index, delta_bs, color="#c0392b", marker="o", lw=1.8,
            label="Δ-AUC vs best-single")
    ax.plot(pivot.index, delta_me, color="#2980b9", marker="s", lw=1.8,
            label="Δ-AUC vs mean-ensemble")
    ax.axhline(0, color="gray", lw=0.7)
    ax.set_xlabel("n_det (detections)")
    ax.set_ylabel("Δ-AUC (trust-weighted − baseline)")
    ax.set_title("DEBASS trust-weighted ensemble gain")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, out_dir / "fig_ensemble_gain.pdf")


def fig_purity_completeness(latency_parquet: Path, out_dir: Path) -> None:
    df = pd.read_parquet(latency_parquet)
    sub = df[df["method"] == "trust_weighted"].copy()
    if sub.empty:
        print("  [skip] purity-completeness — no trust_weighted data")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for n in [3, 5, 10]:
        row = sub[sub["n_det"] == n]
        if row.empty:
            continue
        pc_keys = [c for c in row.columns if c.startswith("purity@comp")]
        compl_keys = [c for c in row.columns if c.startswith("completeness@pur")]
        # Map columns: purity@comp0.5, 0.7, 0.9, 0.95
        pc = {float(c.split("comp")[1]): row[c].iloc[0] for c in pc_keys}
        cc = {float(c.split("pur")[1]): row[c].iloc[0] for c in compl_keys}
        axes[0].plot(sorted(pc), [pc[k] for k in sorted(pc)], marker="o",
                     label=f"n_det={n}", lw=1.6)
        axes[1].plot(sorted(cc), [cc[k] for k in sorted(cc)], marker="s",
                     label=f"n_det={n}", lw=1.6)
    axes[0].set_title("Max purity at completeness target")
    axes[0].set_xlabel("completeness target")
    axes[0].set_ylabel("purity")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.3)
    axes[1].set_title("Max completeness at purity target")
    axes[1].set_xlabel("purity target")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.3)
    fig.suptitle("DEBASS trust-weighted ensemble — purity × completeness tradeoffs")
    fig.tight_layout()
    _savefig(fig, out_dir / "fig_purity_completeness.pdf")


def fig_calibration(trust_metrics_json: Path, snapshot_parquet: Path,
                    trust_models_dir: Path, out_dir: Path) -> None:
    """Reliability diagrams per trust head — raw vs calibrated on test set."""
    import json

    if not trust_metrics_json.exists():
        print(f"  [skip] calibration — {trust_metrics_json} missing")
        return
    metrics = json.loads(trust_metrics_json.read_text())
    if not metrics:
        return
    n = len(metrics)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows),
                             squeeze=False, sharex=True, sharey=True)

    for i, (expert_key, m) in enumerate(metrics.items()):
        ax = axes[i // cols][i % cols]
        raw_brier = m.get("raw", {}).get("brier")
        cal_brier = m.get("calibrated", {}).get("brier")
        raw_ece = m.get("raw", {}).get("ece")
        cal_ece = m.get("calibrated", {}).get("ece")
        # We don't have stored per-row test predictions to draw a reliability
        # diagram directly; instead, show ECE/Brier bars as a summary proxy.
        bars = ["raw", "calibrated"]
        briers = [raw_brier or 0, cal_brier or 0]
        eces = [raw_ece or 0, cal_ece or 0]
        x = np.arange(2)
        width = 0.35
        ax.bar(x - width / 2, briers, width, label="Brier", color="#2980b9")
        ax.bar(x + width / 2, eces, width, label="ECE", color="#c0392b")
        ax.set_title(expert_key, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(bars)
        ax.set_ylim(0, max(max(briers), max(eces)) * 1.2 if (briers + eces) else 0.2)
        ax.grid(alpha=0.3, axis="y")
        if i == 0:
            ax.legend(frameon=False, fontsize=8)

    # Hide any unused cells
    for i in range(n, rows * cols):
        axes[i // cols][i % cols].set_visible(False)

    fig.suptitle("DEBASS calibration: Brier + ECE, raw vs isotonic-calibrated", y=1.02)
    fig.tight_layout()
    _savefig(fig, out_dir / "fig_calibration.pdf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latency", default="reports/metrics/latency_curves.parquet")
    parser.add_argument("--trust-metrics", default="reports/metrics/expert_trust_metrics.json")
    parser.add_argument("--snapshots", default="data/gold/object_epoch_snapshots_trust.parquet")
    parser.add_argument("--trust-models-dir", default="models/trust")
    parser.add_argument("--out-dir", default="paper/debass_aas/figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if Path(args.latency).exists():
        print("Generating figures…")
        fig_latency_curve(Path(args.latency), out_dir)
        fig_ensemble_gain(Path(args.latency), out_dir)
        fig_purity_completeness(Path(args.latency), out_dir)
    else:
        print(f"[warn] {args.latency} missing — run analyze_results.py first")
    fig_calibration(Path(args.trust_metrics), Path(args.snapshots),
                    Path(args.trust_models_dir), out_dir)

    print(f"\nDone. Figures in {out_dir}")


if __name__ == "__main__":
    main()
