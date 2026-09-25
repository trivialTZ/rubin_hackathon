"""Adversarial verifier probe 2 — hierarchical head + alpha-fit internals.

(e) 1-SE selection property: chosen alpha never worse than alpha=0 on the fit slice
    (random stress, 500 draws).
(f) _label_index dtype: fit_alpha on a frame containing one non-ternary
    target_class row.
(g) isotonic-vs-Platt threshold counts ROWS not OBJECTS (multi-epoch LSST cal).
(h) predict_proba on degenerate rows: all-NaN features, unknown survey, empty df.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from debass_meta.models.anchor_blend import ALPHA_GRID, _fit_alpha_1se, fit_alpha

print("=== (e) 1-SE property: means[chosen] <= means[0.0] + 1e-12, 500 random draws ===")
rng = np.random.default_rng(123)
viol = 0
for t in range(500):
    n = rng.integers(2, 120)
    model = rng.dirichlet(np.ones(3), size=n)
    anchor = rng.dirichlet(np.ones(3), size=n)
    y = rng.integers(0, 3, size=n)
    a, info = _fit_alpha_1se(model, anchor, y)
    if info["loss_blend"] > info["loss_anchor"] + 1e-12:
        viol += 1
        print("VIOLATION", t, a, info)
print("violations:", viol, "/500")

print("\n=== (f) fit_alpha with one non-ternary target_class row ===")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from test_anchor_blend import _cal_frame  # reuse the suite's frame builder

df = _cal_frame(120, anchor_good=True, model_good=False, seed=9)
df.loc[0, "target_class"] = "unknown_class"   # e.g. an unlabeled row slipping in
try:
    spec = fit_alpha(df)
    print("fit_alpha survived; cells:", list(spec.alpha_cells))
except Exception as exc:
    print(f"fit_alpha CRASHED: {type(exc).__name__}: {exc}")

print("\n=== (g) survey_cal_min counts rows, not objects ===")
from test_hierarchical_followup import SEED, _add_expert_block, _class_for, _IA_SHIFT, _OTHER_CENTER, _SN_CENTER
from debass_meta.models.hierarchical_followup import HierarchicalFollowup

rng = np.random.default_rng(SEED)
rows, split = [], {}

def emit(object_id, survey, is_sn, is_ia, quality, n_epochs, phase):
    cls = _class_for(is_sn, is_ia)
    for n_det in range(1, n_epochs + 1):
        center = _SN_CENTER if is_sn else _OTHER_CENTER
        feats = center + (_IA_SHIFT if is_ia else 0.0) + rng.normal(0, 1.1, 4)
        subtype = "SN Ia" if is_ia else "SN II"
        row = {
            "object_id": object_id, "n_det": n_det, "survey": survey,
            "target_class": cls, "label_quality": quality,
            "label_source": "tns" if quality == "spectroscopic" else "weak_stamp",
            "tns_type": subtype if quality == "spectroscopic" else None,
            "bts_type": subtype if quality == "spectroscopic" else "-",
            "alert_jd": 2460000.0 + n_det,
            "survey_is_lsst": 1.0 if survey == "LSST" else 0.0,
        }
        for j, v in enumerate(feats):
            row[f"lcf_{j}"] = float(v)
        _add_expert_block(row, rng, 0.8 if is_sn else 0.15)
        rows.append(row)
    split[object_id] = phase

zi = 0
for phase, count in (("train", 120), ("cal", 60)):
    for _ in range(count):
        is_sn = int(rng.random() < 0.6)
        is_ia = int(is_sn and rng.random() < 0.55)
        emit(f"ZTF{zi:05d}", "ZTF", is_sn, is_ia, "spectroscopic", int(rng.integers(4, 9)), phase)
        zi += 1
# LSST: only SIX cal objects, but 8 epochs each -> 48 rows >= survey_cal_min(40)
li = 0
for phase, count in (("train", 40), ("cal", 6)):
    for kk in range(count):
        is_sn = int(kk % 2 == 0)
        is_ia = int(is_sn and rng.random() < 0.5)
        emit(f"LSST{li:05d}", "LSST", is_sn, is_ia,
             "spectroscopic" if is_sn else "context", 8, phase)
        li += 1

df2 = pd.DataFrame(rows)
tr = {k for k, v in split.items() if v == "train"}
ca = {k for k, v in split.items() if v == "cal"}
m = HierarchicalFollowup(survey_cal_min=40, n_jobs=2, seed=SEED).fit(df2, tr, ca)
n_lsst_cal_rows = int((df2["object_id"].isin(ca) & (df2["survey"] == "LSST")).sum())
n_lsst_cal_obj = df2.loc[df2["object_id"].isin(ca) & (df2["survey"] == "LSST"), "object_id"].nunique()
print(f"LSST cal: {n_lsst_cal_obj} objects / {n_lsst_cal_rows} rows; "
      f"head1 calibrator kind = {m.head1_calibrator_kinds.get('lsst')}")

print("\n=== (h) degenerate predict rows ===")
probe = pd.DataFrame([
    {"object_id": "X1", "survey": "ZTF"},                 # everything missing
    {"object_id": "X2", "survey": "atlas"},               # unknown survey
    {"object_id": "X3"},                                  # no survey at all
])
p = m.predict_proba(probe)
print("proba:\n", np.round(p, 4), "\nrow sums:", p.sum(axis=1))
p0 = m.predict_proba(probe.iloc[0:0])
print("empty df shape:", p0.shape)
