"""Cross-check local rerun_exact experts against remote broker outputs.

For each (local, remote) pair of experts that classify the same thing,
compare their ternary projections at the latest epoch per object. Reads
already-built gold snapshots; no new broker fetches.

Example:
    python3 scripts/validate_local_vs_remote.py \\
        --snapshots data/gold/object_epoch_snapshots_safe_v5c.parquet \\
        --outdir    reports/local_vs_remote
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# (local_key, remote_key, survey_filter, note)
PAIRS: list[tuple[str, str, str | None, str]] = [
    ("supernnova", "fink/snn", "ZTF",
     "weights mismatch expected (local=ELAsTiCC, remote=real-ZTF)"),
    ("supernnova", "fink_lsst/snn", "LSST",
     "both ELAsTiCC-trained — tightest same-weights sanity check"),
    ("supernnova", "pittgoogle/supernnova_lsst", "LSST",
     "cross-broker SNN consistency on LSST"),
    # ALeRCE LC classifiers are latest_object_unsafe → excluded from safe snapshots;
    # no remote rows to compare against. Documented limitation.
]


def san(k: str) -> str:
    return k.replace("/", "__")


def extract_pair(
    df: pd.DataFrame, local: str, remote: str, survey: str | None
) -> pd.DataFrame:
    sl, sr = san(local), san(remote)
    needed = [
        f"proj__{sl}__p_snia",
        f"proj__{sr}__p_snia",
        f"proj__{sl}__p_nonIa_snlike",
        f"proj__{sr}__p_nonIa_snlike",
        f"proj__{sl}__p_other",
        f"proj__{sr}__p_other",
        f"mapped_pred_class__{sl}",
        f"mapped_pred_class__{sr}",
        f"avail__{sl}",
        f"avail__{sr}",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"  [SKIP] {local} vs {remote}: missing {len(missing)} cols "
              f"(first: {missing[0]})")
        return pd.DataFrame()
    sub = df[["object_id", "n_det", "survey"] + needed].copy()
    if survey is not None:
        sub = sub[sub["survey"] == survey]
    sub = sub[
        (sub[f"avail__{sl}"] == 1.0)
        & (sub[f"avail__{sr}"] == 1.0)
        & sub[f"proj__{sl}__p_snia"].notna()
        & sub[f"proj__{sr}__p_snia"].notna()
    ]
    # latest n_det per object
    sub = sub.sort_values("n_det").groupby("object_id", as_index=False).tail(1)
    return sub


def metrics_for_pair(sub: pd.DataFrame, local: str, remote: str) -> dict:
    if len(sub) < 5:
        return {"n_matched": int(len(sub)), "status": "insufficient"}
    sl, sr = san(local), san(remote)
    x = sub[f"proj__{sl}__p_snia"].astype(float).to_numpy()
    y = sub[f"proj__{sr}__p_snia"].astype(float).to_numpy()
    r_pearson, _ = pearsonr(x, y)
    r_spearman, _ = spearmanr(x, y)
    mad = float(np.mean(np.abs(x - y)))
    rmse = float(np.sqrt(np.mean((x - y) ** 2)))
    bias = float(np.mean(x - y))
    top_l = sub[f"mapped_pred_class__{sl}"].astype(str)
    top_r = sub[f"mapped_pred_class__{sr}"].astype(str)
    agree = float((top_l == top_r).mean())
    # disagreement matrix
    conf = (
        pd.crosstab(top_l, top_r, dropna=False)
        .reindex(index=["snia", "nonIa_snlike", "other"],
                 columns=["snia", "nonIa_snlike", "other"], fill_value=0)
    )
    return {
        "n_matched": int(len(sub)),
        "pearson_r": float(r_pearson),
        "spearman_r": float(r_spearman),
        "mean_abs_delta_p_snia": mad,
        "rmse_p_snia": rmse,
        "bias_local_minus_remote": bias,
        "top1_agreement": agree,
        "conf_matrix_local_rows_remote_cols": conf.to_dict(orient="split"),
    }


def plot_scatter(sub: pd.DataFrame, local: str, remote: str, out_path: Path) -> None:
    sl, sr = san(local), san(remote)
    x = sub[f"proj__{sl}__p_snia"].astype(float).to_numpy()
    y = sub[f"proj__{sr}__p_snia"].astype(float).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax = axes[0]
    ax.scatter(x, y, s=6, alpha=0.3)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(f"local: {local}\np_snia")
    ax.set_ylabel(f"remote: {remote}\np_snia")
    ax.set_title(f"n={len(sub):,}")
    ax.set_aspect("equal")
    ax = axes[1]
    delta = x - y
    ax.hist(delta, bins=50, edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("local.p_snia − remote.p_snia")
    ax.set_ylabel("count")
    ax.set_title(
        f"bias={delta.mean():+.3f}  MAD={np.mean(np.abs(delta)):.3f}"
    )
    fig.suptitle(f"{local}  vs  {remote}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--snapshots",
        default="data/gold/object_epoch_snapshots_safe_v5c.parquet",
    )
    ap.add_argument("--outdir", default="reports/local_vs_remote")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Reading {args.snapshots} ...")
    df = pd.read_parquet(args.snapshots)
    print(f"  {len(df):,} snapshot rows, "
          f"{df['object_id'].nunique():,} objects, "
          f"surveys={df['survey'].value_counts().to_dict()}")

    summary_rows: list[dict] = []
    for local, remote, survey, note in PAIRS:
        print(f"\n[PAIR] {local}  vs  {remote}  ({survey or 'any'})")
        print(f"       {note}")
        sub = extract_pair(df, local, remote, survey)
        row = {
            "local_expert": local,
            "remote_expert": remote,
            "survey": survey or "any",
            "note": note,
        }
        if len(sub) == 0:
            row["n_matched"] = 0
            row["status"] = "no_data"
            summary_rows.append(row)
            continue
        m = metrics_for_pair(sub, local, remote)
        # strip nested confusion matrix for flat CSV
        conf = m.pop("conf_matrix_local_rows_remote_cols", None)
        row.update(m)
        summary_rows.append(row)
        print(
            f"       n={m['n_matched']:,}  r={m['pearson_r']:+.3f}  "
            f"rho={m['spearman_r']:+.3f}  "
            f"MAD={m['mean_abs_delta_p_snia']:.3f}  "
            f"bias={m['bias_local_minus_remote']:+.3f}  "
            f"top1_agree={m['top1_agreement']:.1%}"
        )
        if conf is not None:
            print(f"       confusion (local rows / remote cols):")
            idx = conf["index"]
            cols = conf["columns"]
            data = conf["data"]
            header = "         " + "  ".join(f"{c:>14s}" for c in cols)
            print(header)
            for r_name, r_row in zip(idx, data):
                print(f"         {r_name:<14s}" + "  ".join(f"{v:>14d}" for v in r_row))

        # plots + mismatches
        pair_tag = f"{san(local)}__vs__{san(remote)}"
        plot_scatter(sub, local, remote, out / f"scatter_{pair_tag}.pdf")
        sl, sr = san(local), san(remote)
        sub = sub.assign(
            delta_p_snia=(
                sub[f"proj__{sl}__p_snia"].astype(float)
                - sub[f"proj__{sr}__p_snia"].astype(float)
            ).abs()
        )
        worst = sub.sort_values("delta_p_snia", ascending=False).head(30)
        worst[
            [
                "object_id",
                "n_det",
                "survey",
                f"proj__{sl}__p_snia",
                f"proj__{sr}__p_snia",
                f"mapped_pred_class__{sl}",
                f"mapped_pred_class__{sr}",
                "delta_p_snia",
            ]
        ].to_csv(out / f"mismatches_{pair_tag}.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "summary.csv", index=False)
    print(f"\nWrote {out}/summary.csv + scatter PDFs + mismatch CSVs")
    print("\n=== SUMMARY ===")
    show_cols = [
        "local_expert",
        "remote_expert",
        "survey",
        "n_matched",
        "pearson_r",
        "mean_abs_delta_p_snia",
        "top1_agreement",
    ]
    display = summary[[c for c in show_cols if c in summary.columns]]
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
