"""Per-class slice of trust-head AUCs — diagnose suspected easy-negative inflation.

For each trust head, compute test AUC sliced by:
  - survey (ZTF / LSST)
  - label_quality (spectroscopic / consensus / weak / context)
  - target_class (snia / nonIa_snlike / other)
  - (survey × label_quality) cross

Answers: does fink_lsst/cats 0.996 reflect "clean negative rejection"
(easy: most objects are non-Ia, predict low; AGN-heavy LSST sample means AUC
is dominated by that skew) or genuine snia discrimination on hard-truth rows?

Example:
    python3 scripts/analyze_trust_heads_by_slice.py \\
        --helpfulness data/gold/expert_helpfulness_safe_v5c.parquet \\
        --trust-snapshots data/gold/object_epoch_snapshots_trust_safe_v5c.parquet \\
        --trust-metadata models/trust_safe_v5c/split_metadata.json \\
        --experts fink_lsst/cats fink_lsst/snn fink_lsst/early_snia pittgoogle/supernnova_lsst \\
        --output reports/trust_head_slices_v5c.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from debass_meta.projectors import sanitize_expert_key
from debass_meta.models.expert_trust import trust_target_col


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """AUC or None if only one class present (common on small slices)."""
    if len(y_true) < 5 or len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def _resolve_target(expert_key: str, joined: pd.DataFrame, override: str | None) -> str | None:
    """Pick the y column to evaluate q against.

    Priority:
      1. --target-col override if given and present in joined.
      2. Per-expert trust target (is_sn for SN-filter experts, else is_topclass_correct).
      3. Fallback to is_topclass_correct.
    """
    if override and override in joined.columns:
        return override
    col = trust_target_col(expert_key)
    if col in joined.columns and joined[col].notna().any():
        return col
    return "is_topclass_correct" if "is_topclass_correct" in joined.columns else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--helpfulness", default="data/gold/expert_helpfulness_safe_v5c.parquet")
    ap.add_argument("--trust-snapshots", default="data/gold/object_epoch_snapshots_trust_safe_v5c.parquet")
    ap.add_argument("--trust-metadata", default=None,
                    help="JSON with {train_ids, cal_ids, test_ids}; if missing, per-model metadata.json files used")
    ap.add_argument("--models-dir", default="models/trust_safe_v5c")
    ap.add_argument("--experts", nargs="+", default=[
        "fink_lsst/cats", "fink_lsst/snn", "fink_lsst/early_snia",
        "pittgoogle/supernnova_lsst", "fink/snn", "lc_features_bv",
    ])
    ap.add_argument("--output", default="reports/trust_head_slices_v5c.csv")
    ap.add_argument("--target-col", default=None,
                    help="Override target column for all experts. Default: auto per expert (is_sn for SN-filter heads).")
    args = ap.parse_args()

    help_df = pd.read_parquet(args.helpfulness)
    trust_df = pd.read_parquet(args.trust_snapshots)
    print(f"helpfulness rows: {len(help_df):,}")
    print(f"trust snapshots: {len(trust_df):,} rows × {len(trust_df.columns)} cols")

    # Derive test_ids: read models_dir metadata if no explicit file
    test_ids: set[str] = set()
    if args.trust_metadata and Path(args.trust_metadata).exists():
        md = json.loads(Path(args.trust_metadata).read_text())
        test_ids = set(str(x) for x in md.get("test_ids", []))
    else:
        # Fall back: check trust_df for object_ids in a "trust_source__*" column tagged "train_model" (= cal|test)
        # Actually easier: use the per-expert metadata.json if present
        any_meta = next(Path(args.models_dir).glob("*/metadata.json"), None)
        if any_meta is None:
            print(f"  no metadata — inferring test set from trust_source columns")
        # For now, assume ALL non-train rows are test-worthy; use first trust_source column to infer
        first_ts = next((c for c in trust_df.columns if c.startswith("trust_source__")), None)
        if first_ts:
            # Objects NEVER tagged "oof" are not in training — they are cal or test
            train_obj = set(trust_df[trust_df[first_ts] == "oof"]["object_id"].astype(str).unique())
            non_train = set(trust_df["object_id"].astype(str).unique()) - train_obj
            test_ids = non_train
            print(f"  inferred test+cal: {len(test_ids):,} objects")

    # Build lookup: truth columns (target_class, label_quality, survey) per row
    # Join help_df with trust_df on (object_id, n_det) to get q__<expert> cols
    # Only keep rows in test_ids
    help_df["object_id"] = help_df["object_id"].astype(str)
    trust_df["object_id"] = trust_df["object_id"].astype(str)
    if test_ids:
        help_df = help_df[help_df["object_id"].isin(test_ids)]
        trust_df = trust_df[trust_df["object_id"].isin(test_ids)]
        print(f"  after test-id filter: help={len(help_df):,}, trust={len(trust_df):,}")

    # Pull survey from trust_df
    if "survey" in trust_df.columns:
        survey_map = (
            trust_df[["object_id", "n_det", "survey"]]
            .drop_duplicates(subset=["object_id", "n_det"])
            .set_index(["object_id", "n_det"])["survey"]
            .to_dict()
        )
    else:
        survey_map = {}

    rows: list[dict] = []
    for expert_key in args.experts:
        san = sanitize_expert_key(expert_key)
        q_col = f"q__{san}"
        avail_col = f"avail__{san}"
        if q_col not in trust_df.columns:
            print(f"[SKIP] {expert_key}: no {q_col} column in trust snapshots")
            continue

        # Subset of helpfulness for this expert
        sub = help_df[help_df["expert_key"] == expert_key].copy()
        if len(sub) == 0:
            print(f"[SKIP] {expert_key}: no helpfulness rows")
            continue

        # Join with trust_df to get q__<expert>
        joined = sub.merge(
            trust_df[["object_id", "n_det", q_col]].drop_duplicates(["object_id", "n_det"]),
            on=["object_id", "n_det"],
            how="left",
        )
        joined["survey"] = joined.apply(
            lambda r: survey_map.get((r["object_id"], int(r["n_det"])), None), axis=1
        )
        target_col = _resolve_target(expert_key, joined, args.target_col)
        if target_col is None:
            print(f"[SKIP] {expert_key}: no target column resolvable")
            continue
        joined = joined[joined[target_col].notna() & joined[q_col].notna()]
        print(f"\n=== {expert_key} ===  (y={target_col}, n_usable={len(joined):,})")
        if len(joined) < 10:
            continue

        y_all = joined[target_col].astype(int).to_numpy()
        q_all = joined[q_col].astype(float).to_numpy()
        auc_all = safe_auc(y_all, q_all)
        pos_rate = float(np.mean(y_all))
        print(f"  overall test: n={len(joined):,}  AUC={auc_all}  pos_rate={pos_rate:.3f}")
        rows.append({
            "expert": expert_key, "target_col": target_col,
            "slice_type": "overall", "slice_value": "all",
            "n": len(joined), "pos_rate": pos_rate, "auc": auc_all,
        })

        # Slice by survey
        for sv in ["ZTF", "LSST"]:
            m = joined["survey"] == sv
            if m.sum() < 10:
                continue
            y = joined.loc[m, target_col].astype(int).to_numpy()
            q = joined.loc[m, q_col].astype(float).to_numpy()
            auc = safe_auc(y, q)
            pr = float(np.mean(y))
            print(f"  survey={sv:4s}: n={m.sum():>6,}  AUC={auc}  pos_rate={pr:.3f}")
            rows.append({
                "expert": expert_key, "target_col": target_col,
                "slice_type": "survey", "slice_value": sv,
                "n": int(m.sum()), "pos_rate": pr, "auc": auc,
            })

        # Slice by label_quality
        for lq in joined["label_quality"].dropna().unique():
            m = joined["label_quality"] == lq
            if m.sum() < 10:
                continue
            y = joined.loc[m, target_col].astype(int).to_numpy()
            q = joined.loc[m, q_col].astype(float).to_numpy()
            auc = safe_auc(y, q)
            pr = float(np.mean(y))
            print(f"  lq={lq:14s}: n={m.sum():>6,}  AUC={auc}  pos_rate={pr:.3f}")
            rows.append({
                "expert": expert_key, "target_col": target_col,
                "slice_type": "label_quality", "slice_value": str(lq),
                "n": int(m.sum()), "pos_rate": pr, "auc": auc,
            })

        # Slice by target_class
        for tc in ["snia", "nonIa_snlike", "other"]:
            m = joined["target_class"] == tc
            if m.sum() < 10:
                continue
            y = joined.loc[m, target_col].astype(int).to_numpy()
            q = joined.loc[m, q_col].astype(float).to_numpy()
            auc = safe_auc(y, q)
            pr = float(np.mean(y))
            print(f"  tc={tc:14s}: n={m.sum():>6,}  AUC={auc}  pos_rate={pr:.3f}")
            rows.append({
                "expert": expert_key, "target_col": target_col,
                "slice_type": "target_class", "slice_value": tc,
                "n": int(m.sum()), "pos_rate": pr, "auc": auc,
            })

        # Cross (survey × label_quality) — hardest slice: LSST × spectroscopic
        for sv in ["ZTF", "LSST"]:
            for lq in ["spectroscopic", "consensus", "weak", "context"]:
                m = (joined["survey"] == sv) & (joined["label_quality"] == lq)
                if m.sum() < 10:
                    continue
                y = joined.loc[m, target_col].astype(int).to_numpy()
                q = joined.loc[m, q_col].astype(float).to_numpy()
                auc = safe_auc(y, q)
                pr = float(np.mean(y))
                print(f"  {sv:4s}×{lq:14s}: n={m.sum():>6,}  AUC={auc}  pos_rate={pr:.3f}")
                rows.append({
                    "expert": expert_key, "target_col": target_col,
                    "slice_type": "survey_x_label_quality",
                    "slice_value": f"{sv}_{lq}",
                    "n": int(m.sum()), "pos_rate": pr, "auc": auc,
                })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {len(rows)} slice rows → {out}")


if __name__ == "__main__":
    main()
