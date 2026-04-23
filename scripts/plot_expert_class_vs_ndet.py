#!/usr/bin/env python3
"""Plot expert probability (per predicted class) as a function of n_det, split
by true class, for the 3 trained trust-head experts.

Writes two PDFs to reports/plots/:
  fig_expert_prob_vs_ndet.pdf  — P(snia|truth=snia) etc per expert
  fig_expert_acc_vs_ndet.pdf   — top-1 accuracy vs n_det per expert
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SNAPSHOT = Path("data/gold/object_epoch_snapshots_trust.parquet")
OUT = Path("reports/plots")
EXPERTS = ["fink__rf_ia", "fink__snn", "supernnova"]
CLASSES = ["snia", "nonIa_snlike", "other"]
CLASS_COLOR = {"snia": "#c0392b", "nonIa_snlike": "#2980b9", "other": "#27ae60"}


def main() -> None:
    df = pd.read_parquet(SNAPSHOT)
    # Restrict to objects with ternary truth
    df = df[df["target_class"].isin(CLASSES)].copy()
    df["n_det_int"] = df["n_det"].astype(int)

    # ── Fig 1: mean P(snia / nonIa / other) vs n_det, per expert, split by truth
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True, sharey=True)
    for i, expert in enumerate(EXPERTS):
        p_cols = {cls: f"proj__{expert}__p_{cls}" for cls in CLASSES}
        for j, truth in enumerate(CLASSES):
            ax = axes[i][j]
            sub = df[df["target_class"] == truth]
            for pred, color in CLASS_COLOR.items():
                col = p_cols[pred]
                if col not in sub.columns:
                    continue
                agg = sub.groupby("n_det_int")[col].agg(["mean", "std", "count"]).reset_index()
                agg = agg[agg["count"] >= 10]
                if agg.empty:
                    continue
                ax.plot(
                    agg["n_det_int"],
                    agg["mean"],
                    color=color,
                    marker="o",
                    ms=4,
                    lw=1.5,
                    label=f"P({pred})",
                )
                ax.fill_between(
                    agg["n_det_int"],
                    agg["mean"] - agg["std"] / np.sqrt(agg["count"]),
                    agg["mean"] + agg["std"] / np.sqrt(agg["count"]),
                    color=color,
                    alpha=0.15,
                )
            ax.axhline(0.5, color="gray", lw=0.5, ls="--")
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlim(0.5, 20.5)
            if i == 0:
                ax.set_title(f"truth = {truth}", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{expert}\nmean P(class)", fontsize=10)
            if i == 2:
                ax.set_xlabel("n_det")
            if i == 0 and j == 2:
                ax.legend(loc="center right", frameon=False, fontsize=8)
            ax.grid(alpha=0.3)

    fig.suptitle(
        "DEBASS trust-head experts — mean predicted probability vs detections\n"
        "rows: expert   ·   columns: true class   ·   shaded band = ±1 SE",
        y=1.00,
    )
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    out1 = OUT / "fig_expert_prob_vs_ndet.pdf"
    out2_png = OUT / "fig_expert_prob_vs_ndet.png"
    fig.savefig(out1, bbox_inches="tight", dpi=150)
    fig.savefig(out2_png, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"wrote {out1}  and  {out2_png}")

    # ── Fig 2: top-1 accuracy vs n_det per expert
    fig, ax = plt.subplots(figsize=(8, 5))
    for expert, marker in zip(EXPERTS, ["o", "s", "^"]):
        p_cols = [f"proj__{expert}__p_{cls}" for cls in CLASSES]
        sub = df.dropna(subset=p_cols).copy()
        if sub.empty:
            continue
        preds = np.asarray(sub[p_cols].values)
        idx = preds.argmax(axis=1)
        sub["pred"] = [CLASSES[k] for k in idx]
        sub["correct"] = (sub["pred"] == sub["target_class"]).astype(int)
        agg = sub.groupby("n_det_int")["correct"].agg(["mean", "count"]).reset_index()
        agg = agg[agg["count"] >= 20]
        ax.plot(
            agg["n_det_int"],
            agg["mean"],
            marker=marker,
            lw=1.8,
            ms=5,
            label=f"{expert} (n={int(agg['count'].sum())})",
        )
    ax.axhline(1 / 3, color="gray", ls=":", lw=0.8, label="random (3-class)")
    ax.set_xlabel("n_det")
    ax.set_ylabel("top-1 accuracy on truth-labelled objects")
    ax.set_title("Expert top-1 classification accuracy vs detections")
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out3 = OUT / "fig_expert_acc_vs_ndet.pdf"
    out4_png = OUT / "fig_expert_acc_vs_ndet.png"
    fig.savefig(out3, bbox_inches="tight", dpi=150)
    fig.savefig(out4_png, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"wrote {out3}  and  {out4_png}")


if __name__ == "__main__":
    main()
