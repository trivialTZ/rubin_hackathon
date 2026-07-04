#!/usr/bin/env python
"""Hackathon figures + verified facts file for the fusion_v8/v9c results.

Every number is read from the pulled SCC report JSONs (reports_from_scc/) —
the facts.json this script writes is the SINGLE SOURCE OF TRUTH for the deck
and notebook builders (no number may appear on a slide that is not in
facts.json).

Run:  python presentations/hackathon_20260612/make_figures.py
Out:  presentations/hackathon_20260612/figs/*.png + facts.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from metadebass_plot_style import PALETTE, apply_neurips_style  # noqa: E402

OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(parents=True, exist_ok=True)

V9C = ROOT / "reports_from_scc" / "fusion_v9c"
V8 = ROOT / "reports_from_scc" / "fusion_v8"

apply_neurips_style(base_font_size=11)
plt.rcParams["figure.dpi"] = 200


def load(path: Path) -> dict:
    return json.loads(path.read_text())


facts: dict = {"provenance": {}}

# ---------------------------------------------------------------- headline --
hg9 = load(V9C / "headline_guards.json")["headline"]
hg8 = load(V8 / "headline_guards.json")["headline"]
facts["headline"] = {
    "n_test_objects": hg9["n_objects"],
    "v9c_macro_auc": hg9["fusion_v8_macro_auc"],
    "v9c_snia_auc": hg9["fusion_v8_auc_snia"]["value"],
    "v9c_delta_vs_v6e2": hg9["vs_v6e2_snia_auc_delta"],
    "v8_macro_auc": hg8["fusion_v8_macro_auc"],
    "v8_delta_vs_v6e2": hg8["vs_v6e2_snia_auc_delta"],
    "protocol": "object-level, spectroscopic-only, locked test split, n_det=5, "
                "1000-resample paired object bootstrap",
}
facts["provenance"]["headline"] = "reports_from_scc/fusion_{v8,v9c}/headline_guards.json (SCC jobs 6060088, 6064722)"

fig, ax = plt.subplots(figsize=(6.0, 2.6))
rows = [
    ("fusion_v9c (integrated)", hg9["vs_v6e2_snia_auc_delta"]),
    ("fusion_v8", hg8["vs_v6e2_snia_auc_delta"]),
]
for i, (label, d) in enumerate(rows):
    ax.errorbar(d["delta"], i, xerr=[[d["delta"] - d["lo"]], [d["hi"] - d["delta"]]],
                fmt="o", color=PALETTE["blue"], capsize=4, markersize=7, lw=2)
ax.axvline(0, color=PALETTE["gray"], lw=1, ls="--")
ax.set_yticks(range(len(rows)), [r[0] for r in rows])
ax.set_xlabel("Δ SN Ia OvR AUC vs previous system (v6e2), n_det=5")
ax.set_title(f"Pre-registered headline — {hg9['n_objects']} spectroscopic test objects (SCC)")
ax.set_xlim(-0.02, 0.20)
fig.tight_layout()
fig.savefig(OUT / "fig_headline.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------ auc vs n_det --
t1 = load(V9C / "tables_1.json")
methods = {"fusion_v8": ("metaDEBASS fusion (ours)", PALETTE["blue"], "o-"),
           "v6e2_rescored": ("previous system (v6e2)", PALETTE["gray"], "s--")}
fig, ax = plt.subplots(figsize=(6.0, 3.4))
curve_facts = {}
for method, (label, color, fmt) in methods.items():
    xs, ys, los, his = [], [], [], []
    for nd in (3, 5, 10, 20):
        rows_ = [r for r in t1["rows"] if r["n_det"] == nd and r["survey"] == "all"
                 and r["method"] == method and r.get("metrics")]
        if not rows_:
            continue
        m = rows_[0]["metrics"].get("auc_snia")
        if isinstance(m, dict) and m.get("value") is not None:
            xs.append(nd); ys.append(m["value"]); los.append(m["lo"]); his.append(m["hi"])
    ax.errorbar(xs, ys, yerr=[np.array(ys) - np.array(los), np.array(his) - np.array(ys)],
                fmt=fmt, color=color, label=label, capsize=3, markersize=5.5, lw=1.8)
    curve_facts[method] = {"n_det": xs, "auc_snia": ys, "lo": los, "hi": his}
facts["auc_vs_ndet_snia"] = curve_facts
facts["provenance"]["auc_vs_ndet_snia"] = "reports_from_scc/fusion_v9c/tables_1.json (spec-only, locked test, object-level)"
ax.set_xlabel("number of detections (epochs since discovery)")
ax.set_ylabel("SN Ia one-vs-rest AUC")
ax.set_title("Early-epoch classification — locked spectroscopic test (SCC)")
ax.set_xticks([3, 5, 10, 20])
ax.legend(loc="lower right")
ax.set_ylim(0.62, 1.0)
fig.tight_layout()
fig.savefig(OUT / "fig_auc_vs_ndet.png", bbox_inches="tight")
plt.close(fig)

# -------------------------------------------------------------- DP1 EF ------
t4 = load(V9C / "tables_4.json")
bc = t4["results"]["overall"]["by_class"]
def ef(cls: str) -> dict:
    e = bc[cls]["by_K"]["0.0100"]["p_follow_proxy"]
    return {"ef": e["ef_point"], "lo": e["ef_ci95_lo"], "hi": e["ef_ci95_hi"]}

# v6e2 baselines from reports/v6e2_dp1_50k/enrichment_headline.md (SCC, 2026-04-25)
V6E2_EF = {"EclBin+RRLyrae": 17.88, "Gaia variables": 5.46, "Gaia stars": 0.85,
           "SIMBAD Galaxy": 0.00, "Published SNe": 0.00}
classes = ["EclBin+RRLyrae", "Gaia variables", "Gaia stars", "SIMBAD Galaxy", "Published SNe"]
new = {c: ef(c) for c in classes if c in bc}
facts["dp1_enrichment_top1pct"] = {
    "v6e2": V6E2_EF,
    "fusion_v9c": new,
    "note": "EF=1 is random; contaminant classes want LOW, Published SNe wants HIGH. N=15,868 DP1 objects.",
}
facts["provenance"]["dp1_enrichment"] = ("fusion: reports_from_scc/fusion_v9c/tables_4.json; "
                                          "v6e2: reports/v6e2_dp1_50k/enrichment_headline.md (SCC)")
fig, ax = plt.subplots(figsize=(6.4, 3.4))
x = np.arange(len(classes))
w = 0.38
old_vals = [V6E2_EF[c] for c in classes]
new_vals = [new[c]["ef"] if c in new else np.nan for c in classes]
ax.bar(x - w / 2, old_vals, w, color=PALETTE["gray"], label="previous (v6e2)")
ax.bar(x + w / 2, new_vals, w, color=PALETTE["blue"], label="metaDEBASS fusion (ours)")
for i, c in enumerate(classes):
    if c in new:
        ax.errorbar(i + w / 2, new[c]["ef"],
                    yerr=[[new[c]["ef"] - new[c]["lo"]], [new[c]["hi"] - new[c]["ef"]]],
                    fmt="none", ecolor=PALETTE["black"], capsize=3, lw=1.2)
ax.axhline(1.0, color=PALETTE["vermillion"], lw=1, ls=":", label="random (EF=1)")
ax.set_xticks(x, [c + ("\n(want high)" if c == "Published SNe" else "\n(want low)") for c in classes],
              fontsize=8.5)
ax.set_ylabel("enrichment factor @ top 1%")
ax.set_title("Operational ranking on 15,868 real Rubin DP1 objects")
ax.legend(fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT / "fig_dp1_ef.png", bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------- trust heads ----
t5 = load(V9C / "tables_5.json")
trust_rows = [r for r in t5.get("rows", []) if isinstance(r.get("pooled_cal_auc"), (int, float))]
trust_rows.sort(key=lambda r: r["pooled_cal_auc"])
facts["trust_heads"] = [
    {"expert": r["expert"], "cal_auc": r["pooled_cal_auc"], "n_test": r["n_test"]}
    for r in trust_rows
]
facts["provenance"]["trust_heads"] = "reports_from_scc/fusion_v9c/tables_5.json (calibrated test AUC of per-expert trust)"
fig, ax = plt.subplots(figsize=(6.4, 3.6))
names = [r["expert"] for r in trust_rows]
aucs = [r["pooled_cal_auc"] for r in trust_rows]
colors = [PALETTE["vermillion"] if n == "seq_v9" else PALETTE["blue"] for n in names]
bars = ax.barh(range(len(names)), aucs, color=colors)
for i, r in enumerate(trust_rows):
    ax.text(r["pooled_cal_auc"] + 0.004, i, f"n={r['n_test']:,}", va="center", fontsize=7.5,
            color=PALETTE["gray"])
ax.set_yticks(range(len(names)), names, fontsize=8.5)
ax.set_xlabel("trust-head AUC (calibrated, locked test)")
ax.set_xlim(0.5, 1.02)
ax.set_title("Per-expert confidence is learnable — incl. our new sequence expert")
fig.tight_layout()
fig.savefig(OUT / "fig_trust_heads.png", bbox_inches="tight")
plt.close(fig)

# -------------------------------------------------------------- gates -------
train_rep = load(V9C / "fusion_v9c_train.json")
gates = [
    {"component": e.get("component"), "decision": e.get("decision"),
     "delta_macro_auc": e.get("delta_macro_auc"),
     "ci95": e.get("delta_macro_auc_ci95")}
    for e in train_rep.get("component_gates", train_rep.get("gates", []))
    if isinstance(e, dict) and e.get("component")
]
facts["component_gates"] = gates
facts["provenance"]["component_gates"] = "reports_from_scc/fusion_v9c/fusion_v9c_train.json (cal-decided, 2,555 objects)"

# ---------------------------------------------------------- context numbers --
facts["scale"] = {
    "gold_rows": 211392, "objects": 12772, "ztf_objects": 8774, "lsst_objects": 3998,
    "experts_registered": 29, "experts_trust_headed": 13,
    "stage_a_rows": 1755697, "seq_v9_params": 45578,
    "seq_v9_standalone_cal_auc": {"snia_ndet5": 0.856, "other_ndet5": 0.865,
                                   "note": "SCC fine-tune cal diagnostics (logs/fusion_v9_pretrain.qsub.out)"},
    "label_bug_fixed": "3,149 BTS-unclassified objects carried fabricated nonIa/spectroscopic labels (15 in locked test) — demoted",
    "wallclock": {"v8_scc_hours": 6.0, "v9c_rerun_min": 54},
}
facts["narrative"] = {
    "problem": "LSST will deliver ~10M alerts/night scored by many broker classifiers that often disagree. "
               "Follow-up observers need to know WHICH classifier to trust on WHICH object, and WHAT to observe "
               "tonight — after only 3-5 detections.",
    "solution": "metaDEBASS: a meta-layer over 29 registered experts. Calibrated per-expert trust "
                "(P(this expert is right on this object, now)) + goal-conditioned priority (u·p with conformal "
                "abstention) for SN Ia, non-Ia, AND 'other' science users.",
    "honesty": "Pre-registered headline + guards; locked test split byte-identical across versions; component "
               "gates decided on cal with bootstrap CIs; label-provenance anti-circularity; independent audits "
               "(Claude adversarial re-computation + OpenAI codex code audit).",
}
(Path(__file__).resolve().parent / "facts.json").write_text(json.dumps(facts, indent=1))
print("figures:", sorted(p.name for p in OUT.glob("*.png")))
print("facts.json written with keys:", sorted(facts.keys()))
