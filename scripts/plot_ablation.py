#!/usr/bin/env python3
"""Plot ensemble AUC vs number of experts included (ablation study).

For each K in 1..N: sort experts by standalone test AUC, take top K, compute
trust-weighted ensemble AUC at fixed n_det bins. Shows how much each new
expert is worth and where the ensemble saturates.

Writes:
  reports/plots/fig_ablation.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _expert_ranking(trust_metrics_path: Path) -> list[str]:
    m = json.loads(Path(trust_metrics_path).read_text())
    ranked = sorted(
        m.items(),
        key=lambda kv: -(kv[1].get("raw", kv[1]).get("roc_auc") or 0),
    )
    return [k for k, _ in ranked]


def _ensemble_auc(df: pd.DataFrame, experts: list[str], truth_col: str) -> float:
    probs = []
    weights = []
    for e in experts:
        p_col = f"proj__{e.replace('/', '__')}__p_snia"
        if p_col not in df.columns:
            continue
        series = df[p_col]
        mask = series.notna()
        if mask.sum() < 10:
            continue
        probs.append(series.fillna(0.5).values)
        weights.append(mask.astype(float).values)
    if not probs:
        return float("nan")
    P = np.vstack(probs)
    W = np.vstack(weights)
    wsum = W.sum(axis=0)
    wsum[wsum == 0] = 1.0
    p_ens = (P * W).sum(axis=0) / wsum
    y = (df[truth_col] == "snia").astype(int).values
    if len(np.unique(y)) < 2:
        return float("nan")
    return roc_auc_score(y, p_ens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", default="data/gold/object_epoch_snapshots_trust.parquet")
    parser.add_argument("--trust-metrics", default="reports/metrics/expert_trust_metrics.json")
    parser.add_argument("--out-dir", default="reports/plots")
    parser.add_argument("--n-det-bins", default="3,5,7,10,15", help="Comma list")
    args = parser.parse_args()

    df = pd.read_parquet(args.snapshots)
    df = df[df["target_class"].notna()].copy()
    df["n_det_int"] = df["n_det"].astype(int)

    ranked_experts = _expert_ranking(Path(args.trust_metrics))
    print(f"Expert ranking: {ranked_experts}")

    n_det_list = [int(x) for x in args.n_det_bins.split(",")]
    rows = []
    for n in n_det_list:
        sub = df[df["n_det_int"] == n]
        if sub.empty:
            continue
        for k in range(1, len(ranked_experts) + 1):
            top_k = ranked_experts[:k]
            auc = _ensemble_auc(sub, top_k, truth_col="target_class")
            rows.append({"n_det": n, "k": k, "auc": auc, "experts": ",".join(top_k)})
    ab = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    for n, color in zip(n_det_list, ["#c0392b", "#2980b9", "#27ae60", "#8e44ad", "#e67e22"]):
        s = ab[ab["n_det"] == n]
        if s.empty:
            continue
        ax.plot(s["k"], s["auc"], marker="o", lw=1.6, color=color, label=f"n_det={n}")
    ax.set_xlabel("K (number of experts in ensemble, top-K by standalone AUC)")
    ax.set_ylabel("Trust-weighted ensemble AUC")
    ax.set_title("DEBASS ablation — does every new expert help?")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_ablation.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(out / "fig_ablation.png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"wrote {out}/fig_ablation.pdf + .png")

    # Also persist the table
    ab.to_parquet(out / "ablation.parquet", index=False)
    print(f"wrote {out}/ablation.parquet")


if __name__ == "__main__":
    main()
