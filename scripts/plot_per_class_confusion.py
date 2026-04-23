#!/usr/bin/env python3
"""Plot 3×3 confusion matrices (Ia/nonIa-SN/other) at selected n_det bins.

Uses the trust-weighted ternary argmax of each object's expert projections.

Writes:
  reports/plots/fig_per_class_confusion.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


CLASSES = ["snia", "nonIa_snlike", "other"]


def _ensemble_ternary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean P(class) over available expert projections per row."""
    out = pd.DataFrame(index=df.index)
    for cls in CLASSES:
        cols = [c for c in df.columns if c.endswith(f"__p_{cls}") and c.startswith("proj__")]
        if not cols:
            out[f"p_{cls}"] = np.nan
            continue
        m = df[cols]
        # Mean of non-null probabilities per row
        out[f"p_{cls}"] = m.mean(axis=1, skipna=True)
    # Normalise so probs sum to 1 (for rows with any non-null)
    s = out[[f"p_{c}" for c in CLASSES]].sum(axis=1)
    for c in CLASSES:
        out[f"p_{c}"] = out[f"p_{c}"] / s.replace(0, np.nan)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", default="data/gold/object_epoch_snapshots_trust.parquet")
    parser.add_argument("--n-det-bins", default="3,5,10,15")
    parser.add_argument("--out-dir", default="reports/plots")
    args = parser.parse_args()

    df = pd.read_parquet(args.snapshots)
    df = df[df["target_class"].isin(CLASSES)].copy()
    df["n_det_int"] = df["n_det"].astype(int)
    ens = _ensemble_ternary(df)
    df = pd.concat([df, ens], axis=1)
    p_cols = [f"p_{c}" for c in CLASSES]
    df = df.dropna(subset=p_cols, how="all").copy()
    df["pred_class"] = df[p_cols].fillna(0).idxmax(axis=1).str.replace("p_", "", regex=False)

    n_det_list = [int(x) for x in args.n_det_bins.split(",")]
    fig, axes = plt.subplots(1, len(n_det_list), figsize=(4 * len(n_det_list), 4), sharey=True)
    if len(n_det_list) == 1:
        axes = [axes]
    for ax, n in zip(axes, n_det_list):
        sub = df[df["n_det_int"] == n]
        if sub.empty:
            ax.text(0.5, 0.5, f"no rows at n_det={n}", ha="center", va="center", transform=ax.transAxes)
            continue
        cm = confusion_matrix(sub["target_class"], sub["pred_class"], labels=CLASSES, normalize="true")
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(CLASSES)))
        ax.set_yticks(range(len(CLASSES)))
        ax.set_xticklabels(CLASSES, rotation=30, ha="right")
        ax.set_yticklabels(CLASSES)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                v = cm[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=9)
        ax.set_title(f"n_det = {n}  (n={len(sub)})")
        ax.set_xlabel("predicted")
        if ax is axes[0]:
            ax.set_ylabel("true")
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, label="row-normalised rate")
    fig.suptitle("DEBASS per-class confusion — mean-ensemble argmax")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_per_class_confusion.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(out / "fig_per_class_confusion.png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"wrote {out}/fig_per_class_confusion.pdf + .png")


if __name__ == "__main__":
    main()
