"""Adversarial verifier probe 1 — anchor/blend edge cases.

(a) scorer no-BlendSpec fallback: does deployed == model as the warn claims?
(b) q=NaN with q__ column present: does the expert fire at weight 1.0
    (divergence from trust_weighted_p_snia)?
(c) single-expert row, LSST row with zero Ia-capable experts, all-NaN LC row.
(d) sum-to-1 after blending with a mixed-alpha spec.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from debass_meta.models import anchor_blend
from debass_meta.models.anchor_blend import BlendSpec, apply, compute_anchor
from debass_meta.projectors.base import sanitize_expert_key
from scripts.score_fusion_v8 import trust_weighted_p_snia

san = sanitize_expert_key

print("=== (a) scorer degraded path: default BlendSpec on an anchored row ===")
row = {"survey": "ztf", "target_class": "snia",
       "p_snia": 0.70, "p_nonia": 0.20, "p_other": 0.10}
k = san("fink/snn")
row[f"proj__{k}__p_snia"] = 0.10
row[f"proj__{k}__p_nonIa_snlike"] = 0.10
row[f"proj__{k}__p_other"] = 0.80
row[f"avail__{k}"] = True
row[f"q__{k}"] = 0.9
df = pd.DataFrame([row])
df = compute_anchor(df)          # mirrors score_fusion_v11.py:156
out = apply(df, BlendSpec())     # mirrors score_fusion_v11.py:157
print("alpha:", out['alpha'].iloc[0], "level:", out['alpha_fallback_level'].iloc[0])
print("model p_snia:", out['p_snia_model'].iloc[0],
      "anchor p_snia:", round(out['p_snia_anchor'].iloc[0], 4),
      "deployed p_snia:", round(out['p_snia'].iloc[0], 4))
print("deployed == model ?", np.isclose(out['p_snia'].iloc[0], 0.70))

print("\n=== (b) q=NaN with q__ column present ===")
row2 = {"survey": "ztf"}
row2[f"proj__{k}__p_snia"] = 0.72
row2[f"proj__{k}__p_nonIa_snlike"] = 0.18
row2[f"proj__{k}__p_other"] = 0.10
row2[f"avail__{k}"] = True
row2[f"q__{k}"] = np.nan          # trust head failed / not scored
df2 = pd.DataFrame([row2])
a2 = compute_anchor(df2)
print("compute_anchor: n_experts_fired =", int(a2['n_experts_fired'].iloc[0]),
      " p_snia_anchor =", a2['p_snia_anchor'].iloc[0])
print("trust_weighted_p_snia (v8 semantics):", trust_weighted_p_snia(df2))

print("\n=== (c1) single-expert LSST row, zero Ia-capable fired ===")
k2 = san("alerce/stamp_classifier_rubin_beta")   # non-Ia-capable, LSST
row3 = {"survey": "LSST"}
row3[f"proj__{k2}__p_snia"] = 0.0
row3[f"proj__{k2}__p_nonIa_snlike"] = 0.9
row3[f"proj__{k2}__p_other"] = 0.1
row3[f"avail__{k2}"] = True
row3[f"q__{k2}"] = 0.8
df3 = pd.DataFrame([row3])
a3 = compute_anchor(df3, base_rate_by_survey={"lsst": 0.62}, default_base_rate=0.5)
tot = a3[[c for c in a3.columns if c.endswith('_anchor')]].iloc[0].sum()
print("n_fired:", int(a3['n_experts_fired'].iloc[0]),
      "anchor:", a3[['p_snia_anchor','p_nonia_anchor','p_other_anchor']].iloc[0].round(4).tolist(),
      "sum:", tot)
print("expected p_snia = 0.9*0.62 =", 0.9*0.62)

print("\n=== (c2) survey string 'LSST' vs base_rate key case ===")
a3b = compute_anchor(df3.assign(survey="LSST"), base_rate_by_survey={"LSST": 0.62})
print("uppercase key routed?", round(a3b['p_snia_anchor'].iloc[0], 4))

print("\n=== (c3) all-NaN LC feature row -> cov bucket + blend ===")
from debass_meta.features.lightcurve import FEATURE_NAMES
row4 = dict(row)  # anchored ZTF row
for c in FEATURE_NAMES:
    row4[c] = np.nan
df4 = pd.DataFrame([row4])
df4 = compute_anchor(df4)
spec = BlendSpec(
    alpha_cells={"ztf|1|<0.25": {"alpha": 0.25, "n": 60}},
    alpha_survey={"ztf": {"alpha": 0.75, "n": 120}},
    alpha_global={"alpha": 1.0, "n": 200},
)
o4 = apply(df4, spec)
print("cov bucket routed to cell? alpha:", o4['alpha'].iloc[0],
      "level:", o4['alpha_fallback_level'].iloc[0])
dep = o4[['p_snia','p_nonia','p_other']].to_numpy(float)
print("deployed sum:", dep.sum(axis=1))

print("\n=== (d) sum-to-1 after blending across random alphas/rows ===")
rng = np.random.default_rng(0)
rows = []
for i in range(200):
    r = {"survey": "ztf" if i % 2 else "lsst", "target_class": "snia"}
    m = rng.dirichlet([1, 1, 1])
    r["p_snia"], r["p_nonia"], r["p_other"] = m
    if i % 3:
        p = rng.dirichlet([1, 1, 1])
        r[f"proj__{k}__p_snia"], r[f"proj__{k}__p_nonIa_snlike"], r[f"proj__{k}__p_other"] = p
        r[f"avail__{k}"] = True
        r[f"q__{k}"] = rng.uniform(0.1, 1.0)
    for j, c in enumerate(FEATURE_NAMES):
        r[c] = 1.0 if i % 4 else np.nan
    rows.append(r)
dfr = pd.DataFrame(rows)
dfr = compute_anchor(dfr)
spec2 = BlendSpec(
    alpha_cells={"ztf|1|>=0.25": {"alpha": 0.5, "n": 60}},
    alpha_survey={"ztf": {"alpha": 0.25, "n": 100}, "lsst": {"alpha": 0.0, "n": 80}},
    alpha_global={"alpha": 0.75, "n": 999},
)
outr = apply(dfr, spec2)
depr = outr[['p_snia','p_nonia','p_other']].to_numpy(float)
print("max |sum-1| deployed:", np.abs(depr.sum(axis=1) - 1).max())
anc = outr[['p_snia_anchor','p_nonia_anchor','p_other_anchor']].to_numpy(float)
fin = np.isfinite(anc).all(axis=1)
print("max |sum-1| anchor (finite rows):", np.abs(anc[fin].sum(axis=1) - 1).max())
