"""Real A/B/C test: which LSST trust-head fix actually improves followup.

Variants (all on the same v5c train/cal/test split):
  V0 — baseline (v5c as-is)
  VA — Option A: retrain fink_lsst/snn and fink_lsst/cats with target=is_sn
       (bool: target_class != 'other'). Their job is SN-vs-other, not Ia.
       Ia-classifier experts (early_snia, pittgoogle/supernnova_lsst) keep
       their is_topclass_correct target.
  VB — Option B: modify p_snia for fink_lsst/snn and cats (double it, since
       projection was × 0.5). Retrain trust with existing target on new
       features. Retrain followup on new features.
  VC1 — Option C minus q: zero q__{4 LSST experts} for everyone (effectively
       remove trust gating for those experts; proj__ and avail__ still used).
  VC2 — Option C minus both: drop q__ AND proj__ columns for the 4 LSST
       experts entirely.

Evaluation:
  For each variant:
    - followup test AUC (overall, ZTF spec, LSST spec, LSST weak)
    - bootstrap CI for LSST spec AUC (resample objects 200×)
    - expected calibration error

Uses existing v5c artifacts on SCC — no gold rebuild required.
"""
from __future__ import annotations

import argparse
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

LSST_EXPERTS = [
    "fink_lsst/snn",
    "fink_lsst/cats",
    "fink_lsst/early_snia",
    "pittgoogle/supernnova_lsst",
]
SN_FILTER_EXPERTS = ["fink_lsst/snn", "fink_lsst/cats"]  # Option A target = is_sn


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(y) < 5 or len(np.unique(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return None


def _bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, groups: np.ndarray,
                      n_boot: int = 200, seed: int = 42) -> tuple[float | None, float | None]:
    if len(y) < 5 or len(np.unique(y)) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    aucs = []
    for _ in range(n_boot):
        picked = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        mask = np.isin(groups, picked)
        if mask.sum() < 5 or len(np.unique(y[mask])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y[mask], p[mask]))
        except Exception:
            continue
    if not aucs:
        return None, None
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def _lgbm_fit(X, y, *, sample_weight=None, n_estimators=200, n_jobs=4):
    from lightgbm import LGBMClassifier
    if len(np.unique(y)) < 2:
        return None, float(np.mean(y))
    model = LGBMClassifier(
        objective="binary", n_estimators=n_estimators, learning_rate=0.05,
        num_leaves=31, min_child_samples=5,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        reg_alpha=0.1, reg_lambda=0.1,
        is_unbalance=True, random_state=42, n_jobs=n_jobs, verbose=-1,
    )
    if sample_weight is not None:
        model.fit(X, y, sample_weight=sample_weight)
    else:
        model.fit(X, y)
    return model, None


def _lgbm_predict(model, X, const=None):
    if model is None:
        return np.full(len(X), float(const or 0.5))
    probs = model.predict_proba(X)
    if probs.shape[1] == 1:
        return np.full(len(X), float(model.classes_[0]))
    idx = list(model.classes_).index(1)
    return probs[:, idx]


def _numeric_feature_cols(df: pd.DataFrame, blocklist: set[str] | None = None) -> list[str]:
    blocked = {"object_id", "target_follow_proxy", "target_class", "label_source", "label_quality"}
    if blocklist:
        blocked = blocked | blocklist
    cols = []
    for c in df.columns:
        if c in blocked:
            continue
        if c == "alert_jd" or c.startswith("source_event_time_jd__"):
            continue
        if c == "lightcurve_association_sep_arcsec":
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if df[c].isna().all() or df[c].nunique(dropna=True) <= 1:
            continue
        cols.append(c)
    return sorted(cols)


def retrain_trust_for_experts(
    help_df: pd.DataFrame, snapshots_df: pd.DataFrame, *,
    experts: list[str], target_fn, train_ids: set, cal_ids: set, test_ids: set,
    label: str,
) -> pd.DataFrame:
    """Retrain trust for the given experts with target_fn, update q__ column
    in snapshots_df. Returns a MODIFIED copy of snapshots_df."""
    out = snapshots_df.copy()
    for expert in experts:
        san = sanitize_expert_key(expert)
        q_col = f"q__{san}"
        rows = help_df[help_df["expert_key"] == expert].copy()
        if len(rows) == 0:
            print(f"  [{label}] {expert}: no rows, skipping")
            continue
        # Filter rows where target_class is defined (needed for target_fn)
        rows = rows[rows["target_class"].notna()]
        # Compute target
        rows["_new_y"] = rows.apply(target_fn, axis=1)
        rows = rows[rows["_new_y"].notna()]
        rows["_new_y"] = rows["_new_y"].astype(int)
        # Feature columns: same pattern as expert_trust._expert_feature_cols
        blocked = {
            "object_id", "expert_key", "target_class", "target_follow_proxy",
            "mapped_pred_class", "prediction_type", "reason", "available",
            "label_source", "label_quality", "is_topclass_correct",
            "is_helpful_for_follow_proxy", "mapped_p_true_class",
            "temporal_exactness", "_new_y",
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
        train_mask = rows["object_id"].isin(train_ids)
        test_mask = rows["object_id"].isin(test_ids | cal_ids)
        if train_mask.sum() == 0:
            print(f"  [{label}] {expert}: no train rows, skipping")
            continue
        X_tr = rows.loc[train_mask, feats].apply(pd.to_numeric, errors="coerce")
        y_tr = rows.loc[train_mask, "_new_y"].to_numpy()
        pos = int(y_tr.sum()); neg = len(y_tr) - pos
        print(f"  [{label}] {expert}: train n={len(y_tr)} pos={pos} neg={neg} features={len(feats)}")
        if len(np.unique(y_tr)) < 2:
            print(f"    [{label}] {expert}: single-class target, skipping")
            continue
        model, const = _lgbm_fit(X_tr, y_tr)
        # Score ALL rows (train + test), update q in snapshots
        X_all = rows[feats].apply(pd.to_numeric, errors="coerce")
        q_new = _lgbm_predict(model, X_all, const=const)
        rows["_q_new"] = q_new
        # Merge into snapshots on (object_id, n_det)
        q_map = rows.set_index(["object_id", "n_det"])["_q_new"].to_dict()
        out[q_col] = [
            q_map.get((oid, nd), out.loc[i, q_col])
            for i, (oid, nd) in enumerate(zip(out["object_id"].astype(str), out["n_det"]))
        ]
    return out


def modify_proj_for_uncapped_p_snia(snapshots_df: pd.DataFrame, experts: list[str]) -> pd.DataFrame:
    """For Option B — the projections of fink_lsst/snn (×0.5) and fink_lsst/cats
    (×0.5 when class=SN-like) cap p_snia at 0.5. Double the p_snia column and
    subtract from p_nonIa_snlike to keep normalization. This assumes the
    expert's 'SN' signal is maximally consistent with 'Ia' — an optimistic
    interpretation that unlocks p_snia > 0.5."""
    out = snapshots_df.copy()
    for expert in experts:
        san = sanitize_expert_key(expert)
        p_snia_col = f"proj__{san}__p_snia"
        p_nonia_col = f"proj__{san}__p_nonIa_snlike"
        if p_snia_col in out.columns:
            orig = out[p_snia_col].astype(float).fillna(0.0)
            nonia = out[p_nonia_col].astype(float).fillna(0.0) if p_nonia_col in out.columns else pd.Series(np.zeros(len(out)))
            # Double p_snia, deduct from p_nonia (where both were equal in projection)
            new_p_snia = (orig * 2).clip(0.0, 1.0)
            new_p_nonia = (nonia - orig).clip(0.0, 1.0)  # removes the 0.5 redundancy
            out[p_snia_col] = new_p_snia
            if p_nonia_col in out.columns:
                out[p_nonia_col] = new_p_nonia
    return out


def drop_features_for_experts(df: pd.DataFrame, experts: list[str], which: str) -> pd.DataFrame:
    """which: 'q' (zero out q__), 'q_proj' (drop q__+proj__)."""
    out = df.copy()
    for expert in experts:
        san = sanitize_expert_key(expert)
        if which == "q":
            if f"q__{san}" in out.columns:
                out[f"q__{san}"] = 0.0  # neutralize trust
        elif which == "q_proj":
            drop_cols = [c for c in out.columns if c == f"q__{san}" or c.startswith(f"proj__{san}__")]
            out = out.drop(columns=drop_cols)
    return out


def train_and_eval_followup(
    snapshots_df: pd.DataFrame, *, train_ids: set, cal_ids: set, test_ids: set,
    variant_tag: str,
) -> dict:
    labelled = snapshots_df[snapshots_df["target_follow_proxy"].notna()].copy()
    feat_cols = _numeric_feature_cols(labelled)
    train_df = labelled[labelled["object_id"].isin(train_ids)]
    test_df = labelled[labelled["object_id"].isin(test_ids)]
    if len(train_df) == 0 or len(test_df) == 0:
        return {"tag": variant_tag, "error": "empty train or test"}

    X_tr = train_df[feat_cols].apply(pd.to_numeric, errors="coerce")
    y_tr = train_df["target_follow_proxy"].astype(int).to_numpy()
    X_te = test_df[feat_cols].apply(pd.to_numeric, errors="coerce")
    y_te = test_df["target_follow_proxy"].astype(int).to_numpy()

    # Weak-weight 0.25 like v5c
    lq_tr = train_df["label_quality"].astype(str)
    sw = np.where(lq_tr.isin(["weak", "context"]), 0.25, 1.0).astype(float)

    model, const = _lgbm_fit(X_tr, y_tr, sample_weight=sw)
    p_te = _lgbm_predict(model, X_te, const=const)

    result = {"tag": variant_tag, "n_train": int(len(train_df)), "n_test": int(len(test_df)),
              "n_features": int(len(feat_cols))}
    result["overall_auc"] = _safe_auc(y_te, p_te)

    # Slice by survey × label_quality
    survey = test_df.get("survey", pd.Series(["?"] * len(test_df))).astype(str).to_numpy()
    lq = test_df["label_quality"].astype(str).to_numpy()
    groups = test_df["object_id"].astype(str).to_numpy()
    tc = test_df["target_class"].astype(str).to_numpy()

    for sv in ["ZTF", "LSST"]:
        for lqv in ["spectroscopic", "weak", "context"]:
            m = (survey == sv) & (lq == lqv)
            if m.sum() >= 5:
                auc = _safe_auc(y_te[m], p_te[m])
                key = f"auc_{sv}_{lqv}"
                result[key] = auc
                result[f"n_{sv}_{lqv}"] = int(m.sum())
                n_obj = len(np.unique(groups[m]))
                result[f"nobj_{sv}_{lqv}"] = n_obj
                # Bootstrap CI for LSST spec
                if sv == "LSST" and lqv == "spectroscopic":
                    lo, hi = _bootstrap_auc_ci(y_te[m], p_te[m], groups[m])
                    result[f"{key}_ci_lo"] = lo
                    result[f"{key}_ci_hi"] = hi

    # LSST × spec × snia slice (the hardest subset)
    m = (survey == "LSST") & (lq == "spectroscopic") & (tc == "snia")
    if m.sum() >= 5:
        auc = _safe_auc(y_te[m], p_te[m])
        result["auc_LSST_spec_snia"] = auc
        result["n_LSST_spec_snia"] = int(m.sum())
        result["nobj_LSST_spec_snia"] = len(np.unique(groups[m]))

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--helpfulness", default="data/gold/expert_helpfulness_safe_v5c.parquet")
    ap.add_argument("--trust-snapshots",
                    default="data/gold/object_epoch_snapshots_trust_safe_v5c.parquet")
    ap.add_argument("--models-dir", default="models/trust_safe_v5c")
    ap.add_argument("--output", default="reports/trust_target_ablation.json")
    ap.add_argument("--skip", default="", help="Comma-separated variants to skip: V0,VA,VB,VC1,VC2")
    args = ap.parse_args()

    skip = set(args.skip.split(",")) if args.skip else set()

    print(f"Loading helpfulness from {args.helpfulness} ...")
    help_df = pd.read_parquet(args.helpfulness)
    help_df["object_id"] = help_df["object_id"].astype(str)

    print(f"Loading trust snapshots from {args.trust_snapshots} ...")
    snap = pd.read_parquet(args.trust_snapshots)
    snap["object_id"] = snap["object_id"].astype(str)

    # Recover train/cal/test from per-expert metadata (lc_features_bv is a reliable pick)
    meta_path = Path(args.models_dir) / "lc_features_bv" / "metadata.json"
    # Fallback: use any expert's metadata or infer from trust_source column
    split_info = None
    for cand in [Path(args.models_dir) / "lc_features_bv" / "metadata.json",
                 Path(args.models_dir) / "fink__snn" / "metadata.json"]:
        if cand.exists():
            md = json.loads(cand.read_text())
            if "train_ids" not in md or "test_ids" not in md:
                continue
            split_info = md
            print(f"  recovered split from {cand}")
            break
    if split_info is None:
        # Infer from trust_source column
        ts_col = next((c for c in snap.columns if c.startswith("trust_source__")), None)
        if ts_col is None:
            raise RuntimeError("No train/test split available")
        train_obj = set(snap[snap[ts_col] == "oof"]["object_id"].astype(str).unique())
        non_train = set(snap["object_id"].astype(str).unique()) - train_obj
        # Split non-train 50/50 by hash
        unique_nt = sorted(non_train)
        np.random.default_rng(42).shuffle(unique_nt)
        mid = len(unique_nt) // 2
        cal_obj = set(unique_nt[:mid])
        test_obj = set(unique_nt[mid:])
        split_info = {"train_ids": list(train_obj), "cal_ids": list(cal_obj), "test_ids": list(test_obj)}
        print(f"  inferred split: train={len(train_obj)} cal={len(cal_obj)} test={len(test_obj)}")

    train_ids = set(str(x) for x in split_info["train_ids"])
    cal_ids = set(str(x) for x in split_info["cal_ids"])
    test_ids = set(str(x) for x in split_info["test_ids"])
    print(f"Split: train={len(train_ids):,}, cal={len(cal_ids):,}, test={len(test_ids):,}")

    results: list[dict] = []

    # --- V0 baseline ---
    if "V0" not in skip:
        print("\n=== V0 baseline ===")
        r = train_and_eval_followup(snap, train_ids=train_ids, cal_ids=cal_ids, test_ids=test_ids,
                                    variant_tag="V0_baseline")
        results.append(r)
        print(json.dumps(r, indent=2, default=str))

    # --- VA Option A ---
    if "VA" not in skip:
        print("\n=== VA Option A — retrain cats/snn with target=is_sn ===")
        def target_is_sn(row):
            tc = row.get("target_class")
            if pd.isna(tc) or tc is None:
                return None
            return int(str(tc) != "other")
        snap_VA = retrain_trust_for_experts(
            help_df, snap, experts=SN_FILTER_EXPERTS, target_fn=target_is_sn,
            train_ids=train_ids, cal_ids=cal_ids, test_ids=test_ids, label="VA",
        )
        r = train_and_eval_followup(snap_VA, train_ids=train_ids, cal_ids=cal_ids, test_ids=test_ids,
                                    variant_tag="VA_sn_target_cats_snn")
        results.append(r)
        print(json.dumps(r, indent=2, default=str))

    # --- VB Option B — uncap p_snia for cats+snn ---
    if "VB" not in skip:
        print("\n=== VB Option B — uncap p_snia for cats/snn + retrain trust ===")
        snap_VB = modify_proj_for_uncapped_p_snia(snap, SN_FILTER_EXPERTS)
        # Retrain trust for cats/snn with their OLD target (is_topclass_correct) on new features
        def target_topclass(row):
            y = row.get("is_topclass_correct")
            if pd.isna(y) or y is None:
                return None
            return int(y)
        # First we need to also modify the helpfulness parquet's proj columns
        help_VB = help_df.copy()
        for expert in SN_FILTER_EXPERTS:
            san = sanitize_expert_key(expert)
            if f"proj__{san}__p_snia" in help_VB.columns:
                orig = help_VB[f"proj__{san}__p_snia"].astype(float).fillna(0.0)
                nonia_col = f"proj__{san}__p_nonIa_snlike"
                nonia = help_VB[nonia_col].astype(float).fillna(0.0) if nonia_col in help_VB.columns else pd.Series(np.zeros(len(help_VB)))
                help_VB[f"proj__{san}__p_snia"] = (orig * 2).clip(0.0, 1.0)
                if nonia_col in help_VB.columns:
                    help_VB[nonia_col] = (nonia - orig).clip(0.0, 1.0)
        snap_VB2 = retrain_trust_for_experts(
            help_VB, snap_VB, experts=SN_FILTER_EXPERTS, target_fn=target_topclass,
            train_ids=train_ids, cal_ids=cal_ids, test_ids=test_ids, label="VB",
        )
        r = train_and_eval_followup(snap_VB2, train_ids=train_ids, cal_ids=cal_ids, test_ids=test_ids,
                                    variant_tag="VB_uncapped_projection")
        results.append(r)
        print(json.dumps(r, indent=2, default=str))

    # --- VC1 Option C — zero q for 4 LSST experts ---
    if "VC1" not in skip:
        print("\n=== VC1 Option C — zero q__{4 LSST experts} ===")
        snap_VC1 = drop_features_for_experts(snap, LSST_EXPERTS, which="q")
        r = train_and_eval_followup(snap_VC1, train_ids=train_ids, cal_ids=cal_ids, test_ids=test_ids,
                                    variant_tag="VC1_zero_q_4lsst")
        results.append(r)
        print(json.dumps(r, indent=2, default=str))

    # --- VC2 Option C — drop q AND proj for 4 LSST experts ---
    if "VC2" not in skip:
        print("\n=== VC2 Option C — drop q AND proj__{4 LSST experts} ===")
        snap_VC2 = drop_features_for_experts(snap, LSST_EXPERTS, which="q_proj")
        r = train_and_eval_followup(snap_VC2, train_ids=train_ids, cal_ids=cal_ids, test_ids=test_ids,
                                    variant_tag="VC2_drop_q_proj_4lsst")
        results.append(r)
        print(json.dumps(r, indent=2, default=str))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))

    # Print comparison table
    print("\n\n=== COMPARISON TABLE ===")
    headers = ["variant", "overall", "ZTF_spec", "LSST_spec", "LSST_spec_CI",
               "LSST_weak", "n_LSST_spec", "nobj_LSST_spec"]
    print("  ".join(f"{h:>22s}" for h in headers))
    for r in results:
        row = [
            r.get("tag", "")[:22],
            f"{r.get('overall_auc') or 0:.3f}",
            f"{r.get('auc_ZTF_spectroscopic') or 0:.3f}",
            f"{r.get('auc_LSST_spectroscopic') or 0:.3f}",
            (f"[{r.get('auc_LSST_spectroscopic_ci_lo', 0):.2f},{r.get('auc_LSST_spectroscopic_ci_hi', 0):.2f}]"
             if r.get('auc_LSST_spectroscopic_ci_lo') is not None else "n/a"),
            f"{r.get('auc_LSST_weak') or 0:.3f}",
            str(r.get('n_LSST_spectroscopic', 0)),
            str(r.get('nobj_LSST_spectroscopic', 0)),
        ]
        print("  ".join(f"{c:>22s}" for c in row))

    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
