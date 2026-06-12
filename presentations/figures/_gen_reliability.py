"""Generate a NeurIPS-style calibration summary for the v7 follow-up proxy."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from metadebass_plot_style import PALETTE, apply_neurips_style, clean_axis, save_figure

METRICS = ROOT / "reports" / "metrics" / "followup_metrics_safe_v7.json"
OUT = Path(__file__).resolve().parent / "reliability_v7.png"


def main() -> None:
    apply_neurips_style(base_font_size=10)
    metrics = json.loads(METRICS.read_text())
    raw = metrics["test_raw"]
    cal = metrics["test_calibrated"]

    labels = ["raw", "isotonic"]
    ece = np.array([raw["ece"], cal["ece"]], dtype=float)
    brier = np.array([raw["brier"], cal["brier"]], dtype=float)
    auc = np.array([raw["roc_auc"], cal["roc_auc"]], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.2, 1.0]})

    ax = axes[0]
    x = np.arange(len(labels))
    width = 0.34
    ax.bar(x - width / 2, ece, width, label="ECE", color=PALETTE["blue"], zorder=3)
    ax.bar(x + width / 2, brier, width, label="Brier", color=PALETTE["orange"], zorder=3)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Error (lower is better)")
    ax.set_title("Calibration improves probability quality", loc="left", pad=8)
    ax.legend(loc="upper right")
    clean_axis(ax, grid_axis="y")
    ymax = max(ece.max(), brier.max()) * 1.35
    ax.set_ylim(0, ymax)
    for xpos, value in zip(x - width / 2, ece):
        ax.text(xpos, value + ymax * 0.035, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    for xpos, value in zip(x + width / 2, brier):
        ax.text(xpos, value + ymax * 0.035, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    ax.plot(labels, auc, color=PALETTE["black"], marker="o", zorder=3)
    ax.set_ylim(min(auc) - 0.001, max(auc) + 0.001)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Ranking is preserved", loc="left", pad=8)
    clean_axis(ax, grid_axis="y")
    for i, value in enumerate(auc):
        ax.text(i, value + 0.00015, f"{value:.4f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        f"v7 follow-up proxy calibration on {raw['n_rows']:,} held-out epoch rows",
        x=0.02,
        y=1.02,
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=PALETTE["black"],
    )
    fig.tight_layout()
    save_figure(fig, OUT)
    plt.close(fig)
    print(f"Wrote {OUT} and {OUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
