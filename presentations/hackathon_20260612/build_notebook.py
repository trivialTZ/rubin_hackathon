#!/usr/bin/env python3
"""Build the executed-results notebook for the metaDEBASS hackathon."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "metaDEBASS_hackathon_results.ipynb"
FACTS_PATH = HERE / "facts.json"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def main() -> None:
    facts = json.loads(FACTS_PATH.read_text())
    narrative = facts["narrative"]

    nb = nbf.v4.new_notebook()
    nb.metadata.update(
        {
            "kernelspec": {
                "display_name": "debass_py313",
                "language": "python",
                "name": "debass_py313",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        }
    )

    cells = [
        md(
            f"""
# metaDEBASS hackathon results

{narrative["problem"]}
{narrative["solution"]}
{narrative["honesty"]}

Verified numbers and provenance are loaded from [`facts.json`](facts.json), with SCC run artifacts under `reports_from_scc/fusion_v9c/`.
"""
        ),
        code(
            r"""
import contextlib
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display


def find_repo_root() -> Path:
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / "presentations/hackathon_20260612/facts.json").exists():
            return candidate
    raise RuntimeError("Run from the repository or a child directory.")


ROOT = find_repo_root()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from metadebass_plot_style import PALETTE, apply_neurips_style, clean_axis

apply_neurips_style(base_font_size=10)


@contextlib.contextmanager
def quiet_native_stderr():
    # Hide pyarrow CPU-probe messages emitted by this sandbox.
    fd = sys.__stderr__.fileno()
    saved = os.dup(fd)
    try:
        with open(os.devnull, "w") as null:
            os.dup2(null.fileno(), fd)
            yield
    finally:
        os.dup2(saved, fd)
        os.close(saved)


def read_parquet(path: Path, **kwargs) -> pd.DataFrame:
    with quiet_native_stderr():
        return pd.read_parquet(path, **kwargs)


def ci_text(metric: dict | None, value_key: str = "value") -> str:
    if not metric:
        return ""
    value = metric.get(value_key)
    lo = metric.get("lo")
    hi = metric.get("hi")
    if value is None:
        return ""
    if lo is None or hi is None:
        return f"{value:.3f}"
    return f"{value:.3f} [{lo:.3f}, {hi:.3f}]"


def conformal_set(row: pd.Series) -> str:
    labels = [
        name
        for name, flag in [
            ("Ia", row.get("set_snia")),
            ("non-Ia", row.get("set_nonia")),
            ("other", row.get("set_other")),
        ]
        if bool(flag)
    ]
    return "{" + ", ".join(labels) + "}"


FACTS = ROOT / "presentations/hackathon_20260612/facts.json"
HEADLINE_GUARDS = ROOT / "reports_from_scc/fusion_v9c/headline_guards.json"
facts = json.loads(FACTS.read_text())
headline_guards = json.loads(HEADLINE_GUARDS.read_text())

headline = facts["headline"]
summary = pd.DataFrame(
    [
        {
            "result": "locked test objects",
            "value": f"{headline['n_test_objects']:,}",
            "provenance": facts["provenance"]["headline"],
        },
        {
            "result": "macro OvR AUC at n_det=5",
            "value": ci_text(headline["v9c_macro_auc"]),
            "provenance": headline["protocol"],
        },
        {
            "result": "SN Ia AUC delta vs v6e2",
            "value": f"{headline['v9c_delta_vs_v6e2']['delta']:.3f} "
            f"[{headline['v9c_delta_vs_v6e2']['lo']:.3f}, {headline['v9c_delta_vs_v6e2']['hi']:.3f}]",
            "provenance": "paired object bootstrap",
        },
        {
            "result": "registered experts / trust-headed experts",
            "value": f"{facts['scale']['experts_registered']} / {facts['scale']['experts_trust_headed']}",
            "provenance": "facts.scale",
        },
        {
            "result": "gold object-epoch rows",
            "value": f"{facts['scale']['gold_rows']:,}",
            "provenance": "facts.scale",
        },
    ]
)

display(Markdown(f"**Pre-registered headline.** {headline_guards['preregistered']['headline']}"))
display(
    summary.style.hide(axis="index")
    .set_properties(**{"text-align": "left"})
    .set_table_styles(
        [
            {"selector": "th", "props": [("text-align", "left"), ("font-weight", "600")]},
            {"selector": "td", "props": [("padding", "6px 10px")]},
        ]
    )
)
"""
        ),
        md(
            """
## Early-epoch discrimination

This plot is regenerated from `tables_1.json`. It compares the live fusion scorer against the re-scored v6e2 baseline and simple ablations at fixed prefix lengths.
"""
        ),
        code(
            r"""
TABLE1 = ROOT / "reports_from_scc/fusion_v9c/tables_1.json"
tables_1 = json.loads(TABLE1.read_text())

auc_rows = []
for row in tables_1["rows"]:
    if "method" not in row or row.get("survey") != "all" or row.get("informational"):
        continue
    metric = row["metrics"]["auc_snia"]
    if metric["value"] is None:
        continue
    auc_rows.append(
        {
            "n_det": int(row["n_det"]),
            "method": row["method"],
            "auc_snia": float(metric["value"]),
            "lo": metric["lo"],
            "hi": metric["hi"],
            "n_objects": row["n_objects"],
        }
    )

auc_df = pd.DataFrame(auc_rows)
method_order = [
    "fusion_v8",
    "v6e2_rescored",
    "best_single_cal[alerce_lc]",
    "trust_weighted_pool",
    "mean_fusion",
]
labels = {
    "fusion_v8": "fusion v9c scorer\n(pre-reg key: fusion_v8)",
    "v6e2_rescored": "v6e2 re-scored",
    "best_single_cal[alerce_lc]": "best single expert",
    "trust_weighted_pool": "trust-weighted pool",
    "mean_fusion": "mean fusion",
}
colors = {
    "fusion_v8": PALETTE["blue"],
    "v6e2_rescored": PALETTE["vermillion"],
    "best_single_cal[alerce_lc]": PALETTE["green"],
    "trust_weighted_pool": PALETTE["purple"],
    "mean_fusion": PALETTE["gray"],
}

fig, ax = plt.subplots(figsize=(7.0, 4.1))
for method in method_order:
    part = auc_df[auc_df["method"] == method].sort_values("n_det")
    if part.empty:
        continue
    ax.plot(part["n_det"], part["auc_snia"], marker="o", label=labels[method], color=colors[method])
    ax.fill_between(
        part["n_det"].to_numpy(),
        part["lo"].to_numpy(dtype=float),
        part["hi"].to_numpy(dtype=float),
        color=colors[method],
        alpha=0.10,
        linewidth=0,
    )

ax.set_xlabel("detections available")
ax.set_ylabel("SN Ia OvR AUC")
ax.set_xticks([3, 5, 10, 20])
ax.set_ylim(0.70, 1.01)
ax.set_title("No-future-leakage performance by prefix length")
clean_axis(ax)
ax.legend(loc="lower right", ncol=1)
plt.show()

display(
    auc_df.pivot(index="n_det", columns="method", values="auc_snia")[method_order]
    .rename(columns=labels)
    .style.format("{:.3f}")
)
"""
        ),
        md(
            """
## One-night evidence packet

The scored parquet is the thing an observer would consume: calibrated class probabilities, conformal sets, priority score, and per-expert trust columns for every object epoch.
"""
        ),
        code(
            r"""
SCORES = ROOT / "data/scores/scc_fusion_v9c"
predictions = read_parquet(SCORES / "predictions_fusion_v9c.parquet")
priority_ia = read_parquet(SCORES / "priority_fusion_v9c_ia.parquet").sort_values("rank")

top_ia = priority_ia.head(10).copy()
top_ia["conformal_set"] = top_ia.apply(conformal_set, axis=1)
top_view = top_ia[
    [
        "rank",
        "object_id",
        "n_det",
        "survey",
        "target_class",
        "label_quality",
        "p_snia",
        "p_nonia",
        "p_other",
        "conformal_set",
        "priority_score",
    ]
]

display(Markdown("**Top 10 under the SN Ia utility vector.**"))
display(
    top_view.style.hide(axis="index").format(
        {
            "rank": "{:.0f}",
            "n_det": "{:.0f}",
            "p_snia": "{:.4f}",
            "p_nonia": "{:.4f}",
            "p_other": "{:.4f}",
            "priority_score": "{:.4f}",
        }
    )
)

interesting = top_ia.iloc[0]
match = predictions[
    (predictions["object_id"].astype(str) == str(interesting["object_id"]))
    & (predictions["n_det"].astype(float) == float(interesting["n_det"]))
]
if match.empty:
    raise RuntimeError("Could not find the rank-1 priority object in predictions parquet.")
rank1 = match.iloc[0]

q_cols = [c for c in predictions.columns if c.startswith("q__")]
trust_rows = []
for q_col in q_cols:
    q = rank1[q_col]
    if pd.isna(q):
        continue
    expert = q_col.removeprefix("q__")
    trust_rows.append(
        {
            "expert": expert.replace("__", "/"),
            "trust_q": float(q),
            "source": rank1.get("trust_source__" + expert, ""),
            "prior_q": rank1.get("q_prior__" + expert, np.nan),
        }
    )
trust_df = pd.DataFrame(trust_rows).sort_values("trust_q", ascending=False)

display(
    Markdown(
        f"**Which experts drive rank 1?** `{rank1['object_id']}` at n_det={int(rank1['n_det'])} "
        f"has p=({rank1['p_snia']:.4f}, {rank1['p_nonia']:.4f}, {rank1['p_other']:.4f}) "
        f"and conformal set {conformal_set(rank1)}."
    )
)
display(trust_df.head(12).style.hide(axis="index").format({"trust_q": "{:.3f}", "prior_q": "{:.3f}"}))
"""
        ),
        md(
            """
## Goal conditioning

The candidate pool is identical. Changing the utility vector from Ia to non-Ia to other changes the ranking and gives different communities different top follow-up lists.
"""
        ),
        code(
            r"""
priority = {
    "u_Ia": priority_ia,
    "u_nonIa": read_parquet(SCORES / "priority_fusion_v9c_nonia.parquet").sort_values("rank"),
    "u_other": read_parquet(SCORES / "priority_fusion_v9c_other.parquet").sort_values("rank"),
}

top5 = []
for goal, df in priority.items():
    view = df.head(5).copy()
    view["goal"] = goal
    view["conformal_set"] = view.apply(conformal_set, axis=1)
    top5.append(
        view[
            [
                "goal",
                "rank",
                "object_id",
                "n_det",
                "survey",
                "p_snia",
                "p_nonia",
                "p_other",
                "conformal_set",
                "priority_score",
            ]
        ]
    )

top5_df = pd.concat(top5, ignore_index=True)
display(
    top5_df.style.hide(axis="index").format(
        {
            "rank": "{:.0f}",
            "n_det": "{:.0f}",
            "p_snia": "{:.4f}",
            "p_nonia": "{:.4f}",
            "p_other": "{:.4f}",
            "priority_score": "{:.4f}",
        }
    )
)

sets = {goal: set(df.head(5)["object_id"].astype(str)) for goal, df in priority.items()}
overlap = pd.DataFrame(
    [
        {"pair": f"{a} vs {b}", "top5_overlap": len(sets[a] & sets[b])}
        for i, a in enumerate(sets)
        for b in list(sets)[i + 1 :]
    ]
)
display(Markdown("**Top-5 overlap between goals.**"))
display(overlap.style.hide(axis="index"))
"""
        ),
        md(
            """
## DP1 enrichment

DP1 is the LSST-like stress test. Lower EF is better for contaminants; higher EF is better for published supernovae. The v6e2 baselines come from `facts.json`, and the fusion values and CIs are read from `tables_4.json`.
"""
        ),
        code(
            r"""
TABLE4 = ROOT / "reports_from_scc/fusion_v9c/tables_4.json"
tables_4 = json.loads(TABLE4.read_text())
overall = tables_4["results"]["overall"]
v6e2 = facts["dp1_enrichment_top1pct"]["v6e2"]

dp1_rows = []
for name, baseline in v6e2.items():
    metric = overall["by_class"][name]["by_K"]["0.0100"]["p_follow_proxy"]
    dp1_rows.append(
        {
            "class": name,
            "v6e2_EF": float(baseline),
            "fusion_v9c_EF": float(metric["ef_point"]),
            "lo": float(metric["ef_ci95_lo"]),
            "hi": float(metric["ef_ci95_hi"]),
            "direction": "higher is better" if name == "Published SNe" else "lower is better",
        }
    )

dp1_df = pd.DataFrame(dp1_rows)
fig, ax = plt.subplots(figsize=(7.2, 4.0))
x = np.arange(len(dp1_df))
width = 0.36
ax.bar(x - width / 2, dp1_df["v6e2_EF"], width, label="v6e2 baseline", color=PALETTE["gray"], zorder=3)
after = ax.bar(x + width / 2, dp1_df["fusion_v9c_EF"], width, label="fusion v9c", color=PALETTE["blue"], zorder=3)
err_low = dp1_df["fusion_v9c_EF"] - dp1_df["lo"]
err_high = dp1_df["hi"] - dp1_df["fusion_v9c_EF"]
ax.errorbar(x + width / 2, dp1_df["fusion_v9c_EF"], yerr=[err_low, err_high], fmt="none", ecolor=PALETTE["black"], capsize=2, linewidth=0.8)
ax.axhline(1.0, color=PALETTE["black"], linewidth=0.8, linestyle=":", label="random EF=1")
ax.set_xticks(x)
ax.set_xticklabels(dp1_df["class"], rotation=25, ha="right")
ax.set_ylabel("enrichment factor at top 1%")
ax.set_title(f"DP1 enrichment, N={overall['n_pool']:,}")
clean_axis(ax)
ax.legend(loc="upper right")
plt.show()

display(
    dp1_df.style.hide(axis="index").format(
        {"v6e2_EF": "{:.2f}", "fusion_v9c_EF": "{:.2f}", "lo": "{:.2f}", "hi": "{:.2f}"}
    )
)
"""
        ),
        md(
            """
## Trust heads are the primary contract

The science product is not only a fused posterior. It is per-expert confidence at a specific object epoch, so judges can audit which experts were trusted for a call.
"""
        ),
        code(
            r"""
TABLE5 = ROOT / "reports_from_scc/fusion_v9c/tables_5.json"
tables_5 = json.loads(TABLE5.read_text())
trust = pd.DataFrame(tables_5["rows"])
trust_plot = trust[trust["pooled_cal_auc"].notna()].copy()
trust_plot = trust_plot.sort_values("pooled_cal_auc", ascending=True)

fig, ax = plt.subplots(figsize=(7.0, 4.8))
bar_colors = [
    PALETTE["orange"] if expert == "seq_v9" else PALETTE["blue"]
    for expert in trust_plot["expert"]
]
ax.barh(trust_plot["expert"], trust_plot["pooled_cal_auc"], color=bar_colors, zorder=3)
ax.set_xlim(0.78, 0.96)
ax.set_xlabel("calibrated trust-head AUC")
ax.set_title("Per-expert trust heads, pooled test rows")
clean_axis(ax)
for y, (_, row) in enumerate(trust_plot.iterrows()):
    if row["expert"] == "seq_v9":
        ax.text(row["pooled_cal_auc"] + 0.004, y, "seq_v9", va="center", color=PALETTE["orange"], fontweight="600")
plt.show()

display(
    trust_plot.sort_values("pooled_cal_auc", ascending=False)[
        ["expert", "pooled_cal_auc", "pooled_brier", "pooled_ece", "n_test", "calibrator_kind"]
    ]
    .style.hide(axis="index")
    .format({"pooled_cal_auc": "{:.3f}", "pooled_brier": "{:.3f}", "pooled_ece": "{:.3f}", "n_test": "{:,.0f}"})
)
"""
        ),
        md(
            f"""
## Live photometry-only expert

This cell loads the trained `SeqClassifierArtifact` from disk and scores a cached ZTF light curve prefix by prefix. `facts.json` records seq_v9 as a {facts["scale"]["seq_v9_params"]:,}-parameter expert; the demo shows how it answers from photometry alone as detections accumulate.
"""
        ),
        code(
            r"""
from debass_meta.features.sequence_dataset import sequence_arrays
from debass_meta.models.seq_classifier import CLASSES, SeqClassifierArtifact

LIVE_OBJECT_ID = "ZTF21abmfrad"
ARTIFACT = ROOT / "models_scc_backup/fusion_v9c_20260612/seq_classifier_v9"
LC_PATH = ROOT / "data/lightcurves" / f"{LIVE_OBJECT_ID}.json"

lightcurve = json.loads(LC_PATH.read_text())
cont, bands = sequence_arrays(lightcurve, max_len=20)
artifact = SeqClassifierArtifact.load(ARTIFACT, device="cpu")
proba = artifact.predict_proba_prefixes(cont, bands, device="cpu")

seq_df = pd.DataFrame(proba, columns=CLASSES)
seq_df.insert(0, "n_det", np.arange(1, len(seq_df) + 1))

fig, ax = plt.subplots(figsize=(7.0, 4.0))
class_colors = {"snia": PALETTE["blue"], "nonIa_snlike": PALETTE["vermillion"], "other": PALETTE["green"]}
for klass in CLASSES:
    ax.plot(seq_df["n_det"], seq_df[klass], marker="o", label=klass, color=class_colors[klass])
ax.set_xlabel("detections available to seq_v9")
ax.set_ylabel("P(class | detections 1..N)")
ax.set_ylim(-0.02, 1.02)
ax.set_xticks([1, 2, 3, 5, 10, 15, 20])
ax.set_title(f"{LIVE_OBJECT_ID}: causal prefix probabilities")
clean_axis(ax)
ax.legend(loc="center right")
plt.show()

summary_prefixes = seq_df[seq_df["n_det"].isin([1, 2, 3, 5, 10, 15, 20])]
display(
    Markdown(
        f"`{LIVE_OBJECT_ID}` has {len(lightcurve)} cached detections; the demo uses the first {len(seq_df)} positive detections. "
        f"SN Ia probability moves from {seq_df.iloc[0]['snia']:.3f} at detection 1 to {seq_df.iloc[-1]['snia']:.3f} at detection {len(seq_df)}."
    )
)
display(summary_prefixes.style.hide(axis="index").format({"snia": "{:.3f}", "nonIa_snlike": "{:.3f}", "other": "{:.3f}"}))
"""
        ),
        code(
            r"""
TRAIN = ROOT / "reports_from_scc/fusion_v9c/fusion_v9c_train.json"
train = json.loads(TRAIN.read_text())

gate_rows = []
for gate in train["gates"]:
    ci = gate.get("delta_macro_auc_ci95")
    purity_ci = gate.get("delta_purity50_ci95")
    gate_rows.append(
        {
            "component": gate["component"],
            "decision": gate["decision"],
            "delta_macro_auc": gate.get("delta_macro_auc"),
            "delta_macro_auc_ci95": "" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]",
            "delta_purity50": gate.get("delta_purity50"),
            "delta_purity50_ci95": "" if purity_ci is None else f"[{purity_ci[0]:.3f}, {purity_ci[1]:.3f}]",
            "reason": gate.get("reason", ""),
        }
    )

gate_df = pd.DataFrame(gate_rows)
display(Markdown("## Gate ledger"))
display(
    gate_df.style.hide(axis="index").format(
        {"delta_macro_auc": "{:.4f}", "delta_purity50": "{:.3f}"}, na_rep=""
    )
)
"""
        ),
        md(
            """
## Closeout and rerun path

Limitations: LSST labels here remain weak where they are not spectroscopic, there is no local LSST spectroscopic truth slice yet, and ELAsTiCC2 is the next external truth stress test. The notebook is still useful evidence because every headline number is loaded from verified local artifacts, every displayed table or plot is regenerated live, and the sequence demo loads the trained model from disk.

Reproduction command:

```bash
bash jobs/submit_fusion_v9_chain.sh
```
"""
        ),
    ]

    nb.cells = cells
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(f"Wrote {OUT.relative_to(ROOT)} with {len(cells)} cells")


if __name__ == "__main__":
    main()
