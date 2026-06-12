"""Small-sample preflight: does class-balanced sample_weight actually
fix the LSST trust-head pathology?

Mini-experiment on ONE expert (fink_lsst/cats):
  (1) Verify the actual (target_class × mapped_pred_class) crosstab.
  (2) Split into train / test with object-level GroupShuffleSplit.
  (3) Fit LightGBM trust head (a) without weight, (b) with inverse-frequency
      class-balanced sample_weight.
  (4) Report per-slice AUC for BOTH — the decisive test for whether class
      weighting is the right lever.

Invariants (pass = hypothesis confirmed):
  (A) baseline reproduces v5c pattern: overall AUC ≥ 0.97, target=snia AUC < 0.3
  (B) class-balanced: target=snia AUC rises above 0.5 (even at some cost
      to overall AUC)

If (B) fails, class weighting is NOT the fix and we need a different target
redefinition (e.g., is_ia_correct). Don't retrain the full v5d until (B)
passes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

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


def report_slices(y: np.ndarray, p: np.ndarray, tc: np.ndarray, lq: np.ndarray, tag: str) -> dict:
    result = {"tag": tag, "n": int(len(y)), "overall_auc": _safe_auc(y, p),
              "overall_pos_rate": float(np.mean(y))}
    for cls in ["snia", "nonIa_snlike", "other"]:
        m = tc == cls
        if m.sum() >= 5:
            auc = _safe_auc(y[m], p[m])
            pr = float(np.mean(y[m]))
            result[f"auc_tc_{cls}"] = auc
            result[f"pos_rate_tc_{cls}"] = pr
            result[f"n_tc_{cls}"] = int(m.sum())
    for lqv in ["spectroscopic", "weak", "context"]:
        m = lq == lqv
        if m.sum() >= 5:
            auc = _safe_auc(y[m], p[m])
            pr = float(np.mean(y[m]))
            result[f"auc_lq_{lqv}"] = auc
            result[f"pos_rate_lq_{lqv}"] = pr
            result[f"n_lq_{lqv}"] = int(m.sum())
    # Cross: LSST-spec-snia is the holy grail
    spec = lq == "spectroscopic"
    for cls in ["snia", "nonIa_snlike", "other"]:
        m = spec & (tc == cls)
        if m.sum() >= 5:
            auc = _safe_auc(y[m], p[m])
            result[f"auc_spec_{cls}"] = auc
            result[f"n_spec_{cls}"] = int(m.sum())
    print(f"\n=== {tag} ===")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--helpfulness", default="data/gold/expert_helpfulness_safe_v5c.parquet")
    ap.add_argument("--expert", default="fink_lsst/cats")
    ap.add_argument("--survey", default="LSST",
                    help="Only train/eval on this survey (since LSST experts are LSST-only)")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--target", default="is_topclass_correct",
                    choices=["is_topclass_correct", "is_helpful_for_follow_proxy"],
                    help="Training target: legacy top-1 match, or Ia-vs-non-Ia helpfulness")
    args = ap.parse_args()

    TARGET_COL = args.target
    print(f"TARGET_COL = {TARGET_COL}")
    help_df = pd.read_parquet(args.helpfulness)
    help_df["object_id"] = help_df["object_id"].astype(str)
    expert_df = help_df[help_df["expert_key"] == args.expert].copy()
    print(f"Loaded {len(expert_df):,} rows for expert {args.expert}")

    # Keep only rows where TARGET_COL is defined
    expert_df = expert_df[expert_df[TARGET_COL].notna()]
    expert_df[TARGET_COL] = expert_df[TARGET_COL].astype(int)
    print(f"  with valid target: {len(expert_df):,}")

    # Infer survey from object_id (LSST = numeric long digit, ZTF = starts with 'ZTF')
    def _is_lsst(oid: str) -> bool:
        return oid.isdigit() and len(oid) > 10
    expert_df["survey"] = expert_df["object_id"].apply(lambda o: "LSST" if _is_lsst(o) else "ZTF")
    if args.survey:
        expert_df = expert_df[expert_df["survey"] == args.survey]
    print(f"  after survey={args.survey} filter: {len(expert_df):,}")

    # --- Sanity: (target_class x mapped_pred_class) crosstab
    print("\n=== (target_class × mapped_pred_class) crosstab ===")
    san = sanitize_expert_key(args.expert)
    mp_col = f"mapped_pred_class__{san}"
    if mp_col not in expert_df.columns:
        # The helpfulness table uses a generic column name 'mapped_pred_class'
        # (see build_expert_helpfulness.py line 46).
        mp_col = "mapped_pred_class"
    tab = pd.crosstab(
        expert_df["target_class"].fillna("NA"),
        expert_df[mp_col].fillna("NA"),
        dropna=False,
    )
    print(tab.to_string())

    # --- Pos rate per target_class
    print("\npos_rate (TARGET_COL == 1) per target_class:")
    print(expert_df.groupby("target_class")[TARGET_COL].agg(["count", "mean"]).to_string())

    # --- Select features: same logic as expert_trust._expert_feature_cols ---
    blocked = {
        "object_id", "expert_key", "target_class", "target_follow_proxy",
        "mapped_pred_class", "prediction_type", "reason", "available",
        "label_source", "label_quality",
        "is_topclass_correct", "is_helpful_for_follow_proxy",
        "mapped_p_true_class", "temporal_exactness",
        "survey",
    }
    feature_cols = []
    for c in expert_df.columns:
        if c in blocked:
            continue
        if c == "alert_jd" or c.startswith("source_event_time_jd__"):
            continue
        if c.startswith(f"proj__") and not c.startswith(f"proj__{san}__"):
            continue
        if c.startswith("q__") or c.startswith("trust_source__"):
            continue
        if c.startswith(f"mapped_pred_class__") and c != f"mapped_pred_class__{san}":
            continue
        if expert_df[c].dtype == object and c not in {
            f"mapped_pred_class__{san}", f"prediction_type__{san}"
        }:
            continue
        if pd.api.types.is_numeric_dtype(expert_df[c]):
            if expert_df[c].isna().all() or expert_df[c].nunique(dropna=True) <= 1:
                continue
            feature_cols.append(c)
    print(f"\n  {len(feature_cols)} features")

    # --- Train/test object-level split ---
    rng = np.random.default_rng(42)
    object_ids = expert_df["object_id"].to_numpy()
    unique_objs = np.unique(object_ids)
    rng.shuffle(unique_objs)
    n_test_objs = int(len(unique_objs) * args.test_frac)
    test_objs = set(unique_objs[:n_test_objs])
    mask_test = np.isin(object_ids, list(test_objs))
    train_df = expert_df[~mask_test].reset_index(drop=True)
    test_df = expert_df[mask_test].reset_index(drop=True)
    print(f"  train={len(train_df):,}  test={len(test_df):,}  test_objs={n_test_objs}")

    y_train = train_df[TARGET_COL].to_numpy().astype(int)
    y_test = test_df[TARGET_COL].to_numpy().astype(int)
    tc_test = test_df["target_class"].astype(str).to_numpy()
    lq_test = test_df["label_quality"].astype(str).to_numpy()

    X_train = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_test = test_df[feature_cols].apply(pd.to_numeric, errors="coerce")

    from lightgbm import LGBMClassifier

    def _fit_and_eval(sample_weight, tag):
        model = LGBMClassifier(
            objective="binary", n_estimators=500, learning_rate=0.05,
            num_leaves=31, min_child_samples=5,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            reg_alpha=0.1, reg_lambda=0.1,
            is_unbalance=True, random_state=42, n_jobs=4, verbose=-1,
        )
        model.fit(X_train, y_train, sample_weight=sample_weight)
        p_test = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
        return report_slices(y_test, p_test, tc_test, lq_test, tag)

    # (a) baseline, no class weights
    baseline = _fit_and_eval(None, "baseline (no class weight)")

    # (b) class-balanced by target_class
    tc_train = train_df["target_class"].astype(str).to_numpy()
    freq = pd.Series(tc_train).value_counts().to_dict()
    # w = n / (n_classes * freq) — sklearn's "balanced" mode
    n_classes = sum(1 for v in freq.values() if v > 0)
    w = np.array([len(tc_train) / (n_classes * freq.get(c, 1)) for c in tc_train])
    print(f"\nClass-balanced weights per target_class:")
    for c, f in freq.items():
        print(f"  {c}: count={f}, weight={len(tc_train) / (n_classes * f):.2f}")
    classbal = _fit_and_eval(w, "class-balanced (by target_class)")

    # (c) STRONGER — inv-freq linear
    w2 = np.array([1.0 / freq.get(c, 1) for c in tc_train])
    w2 = w2 * (len(tc_train) / w2.sum())
    strongbal = _fit_and_eval(w2, "normalized inv-frequency (strong)")

    # --- Invariants ---
    print("\n=== INVARIANT CHECK ===")
    overall_baseline = baseline.get("overall_auc") or 0
    snia_baseline = baseline.get("auc_tc_snia") or 0
    snia_classbal = classbal.get("auc_tc_snia") or 0
    snia_strongbal = strongbal.get("auc_tc_snia") or 0
    print(f"(A) baseline overall={overall_baseline:.3f} (expect >= 0.9)")
    print(f"(A) baseline snia   ={snia_baseline:.3f} (expect <= 0.5 confirming pathology)")
    print(f"(B) classbal  snia  ={snia_classbal:.3f}")
    print(f"(B) strongbal snia  ={snia_strongbal:.3f}")
    print(f"(B) best snia lift  : {max(snia_classbal, snia_strongbal) - snia_baseline:+.3f}")
    if max(snia_classbal, snia_strongbal) > 0.5 and max(snia_classbal, snia_strongbal) - snia_baseline > 0.1:
        print("[B] PASS — class weighting meaningfully improves snia AUC")
    else:
        print("[B] INCONCLUSIVE/FAIL — may need different target redef, not just weights")


if __name__ == "__main__":
    main()
