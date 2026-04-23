#!/usr/bin/env python3
"""Per-object traces: how each trust-head expert's P(class) evolves with n_det
for individual objects.

Writes:
  reports/plots/fig_per_object_traces.pdf  — 3x3 grid (3 experts × 3 example objects)
  reports/plots/fig_per_object_heatmap.pdf — P(snia) heatmap (rows=objects, cols=n_det)
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


def _pick_example_objects(df: pd.DataFrame, truth: str, n: int = 1) -> list[str]:
    """Pick n objects with the given truth that have the most n_det coverage across all 3 experts."""
    sub = df[df["target_class"] == truth]
    all_p = [f"proj__{e}__p_snia" for e in EXPERTS]
    sub = sub.dropna(subset=all_p)
    coverage = sub.groupby("object_id")["n_det"].nunique().sort_values(ascending=False)
    return coverage.head(n).index.tolist()


def _pick_spec_examples(df: pd.DataFrame, truth: str, n: int = 1) -> list[str]:
    sub = df[(df["target_class"] == truth) & (df["label_quality"] == "spectroscopic")]
    all_p = [f"proj__{e}__p_snia" for e in EXPERTS]
    sub = sub.dropna(subset=all_p)
    coverage = sub.groupby("object_id")["n_det"].nunique().sort_values(ascending=False)
    return coverage.head(n).index.tolist()


def fig_traces(df: pd.DataFrame) -> None:
    examples = {truth: _pick_spec_examples(df, truth, 1) for truth in CLASSES}
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True, sharey=True)
    for i, expert in enumerate(EXPERTS):
        for j, truth in enumerate(CLASSES):
            ax = axes[i][j]
            for oid in examples[truth]:
                sub = df[df["object_id"] == oid].sort_values("n_det")
                for pred, color in CLASS_COLOR.items():
                    col = f"proj__{expert}__p_{pred}"
                    if col not in sub.columns:
                        continue
                    series = sub[["n_det", col]].dropna()
                    if series.empty:
                        continue
                    ax.plot(
                        series["n_det"],
                        series[col],
                        color=color,
                        marker="o",
                        ms=3.5,
                        lw=1.5,
                        label=f"P({pred})",
                    )
            ax.axhline(0.5, color="gray", ls="--", lw=0.5)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlim(0.5, 20.5)
            if i == 0:
                title = f"truth={truth}"
                if examples[truth]:
                    title += f"\n{examples[truth][0]}"
                ax.set_title(title, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{expert}\nP(class)", fontsize=10)
            if i == 2:
                ax.set_xlabel("n_det")
            if i == 0 and j == 2:
                ax.legend(loc="center right", frameon=False, fontsize=8)
            ax.grid(alpha=0.3)
    fig.suptitle(
        "DEBASS — per-object classification trace (one spec-typed example per class)\n"
        "rows: expert  ·  columns: true class",
        y=1.00,
    )
    fig.tight_layout()
    out = OUT / "fig_per_object_traces.pdf"
    out_png = OUT / "fig_per_object_traces.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(out_png, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"wrote {out} and {out_png}")
    return examples


def fig_heatmap(df: pd.DataFrame) -> None:
    """Heatmap: rows=objects (sorted by true class), cols=n_det, cell=P(snia) for fink/snn."""
    expert = "fink__snn"
    col = f"proj__{expert}__p_snia"
    sub = df[df["target_class"].isin(CLASSES)].dropna(subset=[col]).copy()
    sub["n_det_int"] = sub["n_det"].astype(int)
    # Pick up to 40 objects per class with the most n_det coverage
    picks = []
    for cls in CLASSES:
        cov = (
            sub[sub["target_class"] == cls]
            .groupby("object_id")["n_det"]
            .nunique()
            .sort_values(ascending=False)
        )
        picks.extend([(oid, cls) for oid in cov.head(30).index])
    oid_list = [p[0] for p in picks]
    mat = (
        sub[sub["object_id"].isin(oid_list)]
        .pivot_table(index="object_id", columns="n_det_int", values=col, aggfunc="mean")
        .reindex(oid_list)
        .reindex(columns=range(1, 21))
    )
    fig, ax = plt.subplots(figsize=(10, 14))
    im = ax.imshow(mat.values, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(0, 20, 2))
    ax.set_xticklabels([str(n) for n in range(1, 21, 2)])
    ax.set_xlabel("n_det")
    ax.set_ylabel("object (grouped by true class)")
    # Row-class boundaries
    boundaries = []
    prev = None
    for i, (_, cls) in enumerate(picks):
        if cls != prev:
            boundaries.append((i, cls))
            prev = cls
    for idx, (i, cls) in enumerate(boundaries):
        ax.axhline(i - 0.5, color="black", lw=1.0)
        y_end = boundaries[idx + 1][0] - 0.5 if idx + 1 < len(boundaries) else len(picks) - 0.5
        ax.text(
            20.5,
            (i + y_end) / 2,
            cls,
            va="center",
            fontsize=11,
            color="black",
            rotation=270,
        )
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, label=f"P(snia) from {expert}")
    ax.set_title(
        f"{expert} — P(snia) per object × n_det\n(up to 30 objects per true class, sorted within class by coverage)"
    )
    fig.tight_layout()
    out = OUT / "fig_per_object_heatmap.pdf"
    out_png = OUT / "fig_per_object_heatmap.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(out_png, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"wrote {out} and {out_png}")


def main() -> None:
    df = pd.read_parquet(SNAPSHOT)
    df = df[df["target_class"].isin(CLASSES)].copy()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_traces(df)
    fig_heatmap(df)


if __name__ == "__main__":
    main()
