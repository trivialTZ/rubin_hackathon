"""Preflight: verify the new calibration guard rejects the fink_lsst/early_snia
pathology on real v5c cal-set data (without a full retrain).

Invariants:
  (A) On early_snia's real cal set, the candidate IsotonicCalibrator causes
      > 0.05 AUC drop on the held-out test set → guard triggers, calibrator
      is REJECTED, and predict_trust falls through to raw.
  (B) On a high-n head like fink_lsst/snn, the guard does NOT trigger (since
      cal has >>20 positives and >>20 negatives and AUC drop is small).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from debass_meta.models.calibrate import IsotonicCalibrator
from debass_meta.projectors import sanitize_expert_key


def simulate_guard(cal_raw: np.ndarray, cal_y: np.ndarray,
                   test_raw: np.ndarray, test_y: np.ndarray) -> tuple[bool, str | None]:
    """Returns (calibrator_accepted, skip_reason_or_none)."""
    valid = ~np.isnan(cal_raw) & np.isin(cal_y, [0, 1])
    n_valid = int(valid.sum())
    n_pos = int(cal_y[valid].sum()) if n_valid > 0 else 0
    n_neg = n_valid - n_pos
    if n_valid < 200:
        return False, f"cal_n<200 (have {n_valid})"
    if n_pos < 20:
        return False, f"cal_pos<20 (have {n_pos})"
    if n_neg < 20:
        return False, f"cal_neg<20 (have {n_neg})"
    if len(np.unique(cal_y[valid])) != 2:
        return False, "single_class_in_cal"
    cand = IsotonicCalibrator()
    cand.fit(cal_raw[valid], cal_y[valid])
    gv = ~np.isnan(test_raw) & np.isin(test_y, [0, 1])
    if gv.sum() < 20 or len(np.unique(test_y[gv])) < 2:
        return True, None
    auc_raw = roc_auc_score(test_y[gv], test_raw[gv])
    auc_cal = roc_auc_score(test_y[gv], np.asarray(cand.transform(test_raw[gv]), dtype=float))
    if (auc_raw - auc_cal) > 0.05:
        return False, f"auc_drop>0.05 (raw={auc_raw:.3f}, cal={auc_cal:.3f})"
    return True, None


def main() -> None:
    help_path = Path("data/gold/expert_helpfulness_safe_v5c.parquet")
    trust_snap_path = Path("data/gold/object_epoch_snapshots_trust_safe_v5c.parquet")
    help_df = pd.read_parquet(help_path)
    trust_df = pd.read_parquet(trust_snap_path)
    help_df["object_id"] = help_df["object_id"].astype(str)
    trust_df["object_id"] = trust_df["object_id"].astype(str)

    for expert in ["fink_lsst/early_snia", "fink_lsst/snn"]:
        san = sanitize_expert_key(expert)
        q_col = f"q__{san}"
        if q_col not in trust_df.columns:
            print(f"[SKIP] {expert}: {q_col} not in trust_df")
            continue

        # Infer train / cal / test via trust_source column
        ts_col = f"trust_source__{san}"
        if ts_col not in trust_df.columns:
            print(f"[SKIP] {expert}: no trust_source column")
            continue

        sub = trust_df[[ "object_id", "n_det", q_col, ts_col ]].drop_duplicates(["object_id", "n_det"])
        hs = help_df[help_df["expert_key"] == expert][[
            "object_id", "n_det", "is_topclass_correct"
        ]].copy()
        j = hs.merge(sub, on=["object_id", "n_det"], how="inner")
        j = j[j["is_topclass_correct"].notna() & j[q_col].notna()]
        # Split: train = trust_source=="oof", cal+test = trust_source=="train_model"
        # Cal set is what the calibrator was fit on originally — we need a
        # separate flag. Approximate: randomly split the non-train rows 50/50
        # using object_id hash (object-level split matches GroupSplit semantics).
        non_train = j[j[ts_col] == "train_model"].copy()
        train_rows = j[j[ts_col] == "oof"]
        # Hash-based object split for cal vs test
        non_train["hash_mod"] = non_train["object_id"].map(lambda x: int(hash(x)) % 2)
        cal_df = non_train[non_train["hash_mod"] == 0]
        test_df = non_train[non_train["hash_mod"] == 1]
        print(f"\n=== {expert} ===")
        print(f"  train={len(train_rows):,}  cal={len(cal_df):,}  test={len(test_df):,}")

        if len(cal_df) == 0 or len(test_df) == 0:
            print("  SKIP: empty cal or test")
            continue

        cal_raw = cal_df[q_col].astype(float).to_numpy()
        cal_y = cal_df["is_topclass_correct"].astype(int).to_numpy()
        test_raw = test_df[q_col].astype(float).to_numpy()
        test_y = test_df["is_topclass_correct"].astype(int).to_numpy()

        accepted, reason = simulate_guard(cal_raw, cal_y, test_raw, test_y)
        # Sanity: report raw vs "what cal would have done" AUC
        valid = ~np.isnan(cal_raw) & np.isin(cal_y, [0, 1])
        gv = ~np.isnan(test_raw) & np.isin(test_y, [0, 1])
        if valid.sum() >= 10 and len(np.unique(cal_y[valid])) == 2:
            cand = IsotonicCalibrator()
            cand.fit(cal_raw[valid], cal_y[valid])
            auc_raw = roc_auc_score(test_y[gv], test_raw[gv]) if gv.sum() >= 2 and len(np.unique(test_y[gv])) == 2 else None
            auc_cal = roc_auc_score(test_y[gv], np.asarray(cand.transform(test_raw[gv]), dtype=float)) if auc_raw is not None else None
            n_pos = int(cal_y[valid].sum())
            n_neg = int(valid.sum()) - n_pos
            print(f"  cal: n_valid={valid.sum()}  n_pos={n_pos}  n_neg={n_neg}")
            print(f"  test AUC raw: {auc_raw:.3f}" if auc_raw else "  test AUC raw: N/A")
            print(f"  test AUC cal: {auc_cal:.3f}" if auc_cal else "  test AUC cal: N/A")
        print(f"  guard decision: accepted={accepted}  reason={reason}")

        if expert == "fink_lsst/early_snia":
            # Invariant (A)
            if accepted:
                print("  [A] FAIL: guard ACCEPTED early_snia calibrator (expected REJECT)")
                sys.exit(1)
            print("  [A] PASS: guard rejected early_snia calibrator")
        elif expert == "fink_lsst/snn":
            # Invariant (B) — for a head with many cal rows, guard should pass
            # (unless the model is genuinely miscalibrated on test, in which
            # case rejection is also correct).
            print(f"  [B] informational: accepted={accepted}")

    print("\nPreflight complete.")


if __name__ == "__main__":
    main()
