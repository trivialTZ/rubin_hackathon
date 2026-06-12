"""Real Option A retrain: fink_lsst/cats and fink_lsst/snn with target=is_sn.

For each of the 2 SN-filter experts:
  1. Train LightGBM with target = int(target_class != 'other')
  2. Evaluate on held-out test set
  3. Report per-slice AUC: overall, per-target_class, per-label_quality
  4. Compare to v5c baseline (target=is_topclass_correct)

This tests the principled claim: 'these experts are SN filters, so we should
train them as SN filters, not Ia classifiers'.

Invariants:
  (A) SN-filter AUC on test set >= 0.7 (reasonable SN-vs-other discrimination)
  (B) AUC on is_sn target reflects expert capability honestly (no longer
      inflated by easy-negative rejection on the wrong task)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from debass_meta.projectors import sanitize_expert_key


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(y) < 5 or len(np.unique(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return None


def train_and_report(expert: str, help_df: pd.DataFrame, *,
                     train_ids: set, test_ids: set, target_col: str,
                     target_builder=None):
    from lightgbm import LGBMClassifier
    san = sanitize_expert_key(expert)
    rows = help_df[help_df.expert_key == expert].copy()
    rows = rows[rows.target_class.notna()]
    if target_builder is not None:
        rows["_y"] = rows.apply(target_builder, axis=1)
    else:
        rows["_y"] = rows[target_col]
    rows = rows[rows["_y"].notna()]
    rows["_y"] = rows["_y"].astype(int)

    # feature cols same as _expert_feature_cols
    blocked = {
        "object_id", "expert_key", "target_class", "target_follow_proxy",
        "mapped_pred_class", "prediction_type", "reason", "available",
        "label_source", "label_quality", "is_topclass_correct",
        "is_helpful_for_follow_proxy", "mapped_p_true_class",
        "temporal_exactness", "_y",
    }
    feats = []
    for c in rows.columns:
        if c in blocked:
            continue
        if c == "alert_jd" or c.startswith("source_event_time_jd__"):
            continue
        if c.startswith("proj__") and not c.startswith(f"proj__{san}__"):
            continue
        if c.startswith("q__") or c.startswith("trust_source__"):
            continue
        if c.startswith("mapped_pred_class__") and c != f"mapped_pred_class__{san}":
            continue
        if rows[c].dtype == object and c not in {f"mapped_pred_class__{san}", f"prediction_type__{san}"}:
            continue
        if pd.api.types.is_numeric_dtype(rows[c]):
            if rows[c].isna().all() or rows[c].nunique(dropna=True) <= 1:
                continue
            feats.append(c)

    tr = rows[rows.object_id.isin(train_ids)]
    te = rows[rows.object_id.isin(test_ids)]
    if len(tr) == 0 or len(te) == 0 or len(np.unique(tr["_y"])) < 2:
        print(f"  {expert}: SKIP (train={len(tr)}, test={len(te)}, n_classes={len(np.unique(tr['_y'])) if len(tr) else 0})")
        return None
    X_tr = tr[feats].apply(pd.to_numeric, errors="coerce")
    y_tr = tr["_y"].to_numpy()
    X_te = te[feats].apply(pd.to_numeric, errors="coerce")
    y_te = te["_y"].to_numpy()

    model = LGBMClassifier(
        objective="binary", n_estimators=500, learning_rate=0.05,
        num_leaves=31, min_child_samples=5,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        reg_alpha=0.1, reg_lambda=0.1,
        is_unbalance=True, random_state=42, n_jobs=4, verbose=-1,
    )
    model.fit(X_tr, y_tr)
    p_te = model.predict_proba(X_te)[:, list(model.classes_).index(1)]

    result = {
        "expert": expert, "target": target_col,
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "train_pos_rate": float(np.mean(y_tr)),
        "overall_auc": _safe_auc(y_te, p_te),
        "overall_pos_rate": float(np.mean(y_te)),
    }
    # By target_class
    tc = te["target_class"].astype(str).to_numpy()
    for cls in ["snia", "nonIa_snlike", "other"]:
        m = tc == cls
        if m.sum() >= 5:
            result[f"auc_tc_{cls}"] = _safe_auc(y_te[m], p_te[m])
            result[f"pos_tc_{cls}"] = float(np.mean(y_te[m]))
            result[f"n_tc_{cls}"] = int(m.sum())
    # By label_quality
    lq = te["label_quality"].astype(str).to_numpy()
    for lqv in ["spectroscopic", "weak", "context"]:
        m = lq == lqv
        if m.sum() >= 5:
            result[f"auc_lq_{lqv}"] = _safe_auc(y_te[m], p_te[m])
            result[f"pos_lq_{lqv}"] = float(np.mean(y_te[m]))
            result[f"n_lq_{lqv}"] = int(m.sum())
    # Spec × snia
    spec = lq == "spectroscopic"
    for cls in ["snia", "nonIa_snlike", "other"]:
        m = spec & (tc == cls)
        if m.sum() >= 5:
            result[f"auc_spec_{cls}"] = _safe_auc(y_te[m], p_te[m])
            result[f"n_spec_{cls}"] = int(m.sum())

    print(f"\n=== {expert} | target={target_col} ===")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    return result


def main() -> None:
    help_df = pd.read_parquet("data/gold/expert_helpfulness_safe_v5c.parquet")
    help_df["object_id"] = help_df["object_id"].astype(str)

    md = json.loads(Path("models/trust_safe_v5c/metadata.json").read_text())
    train_ids = set(str(x) for x in md["train_ids"])
    test_ids = set(str(x) for x in md["test_ids"])
    print(f"Split: train={len(train_ids):,} test={len(test_ids):,}")

    results = []
    # For cats and snn: train under both targets and compare
    for expert in ["fink_lsst/cats", "fink_lsst/snn"]:
        # Baseline: target = is_topclass_correct
        r_base = train_and_report(expert, help_df, train_ids=train_ids,
                                  test_ids=test_ids, target_col="is_topclass_correct")
        if r_base: results.append(r_base)
        # Option A: target = is_sn
        def is_sn(row):
            tc = row.get("target_class")
            if pd.isna(tc) or tc is None:
                return None
            return int(str(tc) != "other")
        r_optA = train_and_report(expert, help_df, train_ids=train_ids,
                                  test_ids=test_ids, target_col="is_sn",
                                  target_builder=is_sn)
        if r_optA: results.append(r_optA)

    # Summary comparison
    print("\n\n=== SUMMARY: Option A effect on trust-head metrics ===")
    print(f"{'expert':<25s}  {'target':<22s}  {'overall':>8s}  {'tc_snia':>8s}  {'tc_nonIa':>8s}  {'tc_other':>8s}  {'spec':>8s}  {'weak':>8s}")
    for r in results:
        print(f"{r['expert']:<25s}  {r['target']:<22s}  "
              f"{(r.get('overall_auc') or 0):>8.3f}  "
              f"{(r.get('auc_tc_snia') or 0):>8.3f}  "
              f"{(r.get('auc_tc_nonIa_snlike') or 0):>8.3f}  "
              f"{(r.get('auc_tc_other') or 0):>8.3f}  "
              f"{(r.get('auc_lq_spectroscopic') or 0):>8.3f}  "
              f"{(r.get('auc_lq_weak') or 0):>8.3f}")

    out = Path("reports/sn_filter_ab_comparison.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
