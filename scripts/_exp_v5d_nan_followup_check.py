#!/usr/bin/env python3
"""Critical check: can v5d followup model score with NaN for the 4 LSST-broker expert columns?"""
from __future__ import annotations
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    # Find v5d followup model locally
    candidates = [
        REPO / "models" / "followup_safe_v5d" / "model.pkl",
        REPO / "models" / "followup_safe_v5d" / "followup.pkl",
    ]
    model_path = None
    for c in candidates:
        if c.exists():
            model_path = c
            break
    if model_path is None:
        # Check elsewhere
        print("v5d followup model not found locally. Candidates tried:")
        for c in candidates:
            print(f"  {c}")
        found = list((REPO / "models").glob("followup*safe*v5*/**/*.pkl"))
        print(f"Globbed results: {found}")
        return 1

    print(f"Loading {model_path}")
    import pickle
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"Model type: {type(model).__name__}")

    meta_path = model_path.parent / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        features = meta.get("features") or meta.get("feature_columns") or meta.get("feature_names")
        print(f"\nExpected features (n={len(features) if features else 'unknown'}):")
        if features:
            for i, f in enumerate(features):
                is_lsst = "fink_lsst" in f or "pittgoogle/supernnova_lsst" in f
                print(f"  [{i:3d}] {'LSST-BROKER' if is_lsst else '           '}  {f}")

        # Build a test row: all zeros, NaN the LSST-broker cols
        import numpy as np
        n = len(features)
        X = np.zeros((1, n))
        for i, f in enumerate(features):
            if "fink_lsst" in f or "pittgoogle/supernnova_lsst" in f:
                X[0, i] = np.nan
        print(f"\nTest row shape: {X.shape}, NaN count: {np.isnan(X).sum()}")

        # Try prediction
        try:
            pred = model.predict_proba(X)
            print(f"predict_proba output: {pred}")
            print("✓ v5d followup accepts NaN LSST-broker columns")
        except Exception as e:
            print(f"✗ predict_proba failed: {type(e).__name__}: {e}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
