#!/usr/bin/env python3
"""Train the LightGBM head for the Bazin/Villar local expert.

Iterates gold snapshot objects with truth labels, loads each lightcurve from
data/lightcurves/, computes Bazin + Villar features per band using
`light-curve-python`, and fits a 3-way LightGBM classifier (snia /
nonIa_snlike / other).

Output: artifacts/local_experts/lc_features/model.pkl (pickled sklearn-compat
LGBMClassifier) + metadata.json.

Usage:
    python3 scripts/train_lc_features_head.py
    python3 scripts/train_lc_features_head.py --max-objects 500  # smoke test
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from debass_meta.experts.local.lc_features import LcFeaturesExpert


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", default="data/truth/object_truth.parquet")
    parser.add_argument("--lightcurves-dir", default="data/lightcurves")
    parser.add_argument("--output-dir", default="artifacts/local_experts/lc_features")
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()

    truth = pd.read_parquet(args.truth)
    # Drop rows without a ternary label
    truth = truth[truth["final_class_ternary"].notna()].copy()
    if args.max_objects:
        truth = truth.head(args.max_objects)
    print(f"Training on {len(truth)} labelled objects")

    lc_dir = Path(args.lightcurves_dir)
    expert = LcFeaturesExpert(model_path=Path("/dev/null"))  # force head=None

    X_rows: list[list[float]] = []
    y_rows: list[str] = []
    skipped = 0
    for i, row in truth.iterrows():
        oid = str(row["object_id"])
        ternary = str(row["final_class_ternary"])
        lc_path = lc_dir / f"{oid}.json"
        if not lc_path.exists():
            skipped += 1
            continue
        try:
            payload = json.loads(lc_path.read_text())
        except Exception:
            skipped += 1
            continue
        dets = payload if isinstance(payload, list) else (
            payload.get("detections") or payload.get("lightcurve") or []
        )
        if len(dets) < 5:
            skipped += 1
            continue
        # Group by band and pick the best band (most dets)
        from debass_meta.experts.local.lc_features import _band, _mjd, _flux

        by_band: dict[str, list[tuple[float, float, float, str]]] = {}
        for det in dets:
            band = _band(det)
            mjd = _mjd(det)
            flux_pair = _flux(det)
            if not (band and mjd and flux_pair):
                continue
            f, ferr = flux_pair
            by_band.setdefault(band, []).append((mjd, f, ferr, band))
        if not by_band:
            skipped += 1
            continue
        best = max(by_band, key=lambda k: len(by_band[k]))
        feats = expert._compute_features(sorted(by_band[best]))
        X_rows.append([feats[n] for n in expert.FEATURE_NAMES])
        y_rows.append(ternary)
        if (len(X_rows) + 1) % 200 == 0:
            print(f"  processed {len(X_rows)} / {len(truth)} (skipped {skipped})")

    print(f"Kept {len(X_rows)} usable rows (skipped {skipped})")
    if len(X_rows) < 100:
        print("Too few rows — aborting")
        sys.exit(1)

    from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=args.n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=5,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=4,
        verbose=-1,
    )
    X = np.asarray(X_rows, dtype=float)
    y = np.asarray(y_rows)
    print(f"Fitting LGBMClassifier on X={X.shape}, y classes={np.unique(y, return_counts=True)}")
    model.fit(X, y)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model.pkl", "wb") as fh:
        pickle.dump(model, fh)
    meta = {
        "feature_names": expert.FEATURE_NAMES,
        "n_classes": 3,
        "classes": list(model.classes_),
        "n_train_rows": int(len(X)),
        "n_skipped": int(skipped),
        "n_estimators": args.n_estimators,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out_dir}/model.pkl and metadata.json")


if __name__ == "__main__":
    main()
