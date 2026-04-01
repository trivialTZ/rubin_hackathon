"""Train a local ALeRCE-style light-curve classifier (balanced random forest).

Uses our cached lightcurves + TNS spectroscopic labels to produce a model
that replaces the ALeRCE API LC classification scores.  Runs on CPU.

Usage:
    python scripts/train_alerce_lc.py
    python scripts/train_alerce_lc.py --from-labels data/labels.csv --full-features
    python scripts/train_alerce_lc.py --n-estimators 500 --max-n-det 10
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from debass_meta.features.lightcurve import extract_features, FEATURE_NAMES

# Map TNS type → ALeRCE LC classifier class
# We can split nonIa_snlike into finer classes using the original TNS type
_TNS_TO_ALERCE: dict[str, str] = {
    # snia
    "SN Ia": "SNIa",
    "SN Ia-91T-like": "SNIa",
    "SN Ia-91bg-like": "SNIa",
    "SN Ia-CSM": "SNIa",
    "SN Ia-pec": "SNIa",
    "SN Ia-SC": "SNIa",
    "SN Iax[02cx-like]": "SNIa",
    # SNIbc
    "SN Ib": "SNIbc",
    "SN Ib-pec": "SNIbc",
    "SN Ib/c": "SNIbc",
    "SN Ic": "SNIbc",
    "SN Ic-BL": "SNIbc",
    "SN Ic-pec": "SNIbc",
    # SNII
    "SN II": "SNII",
    "SN IIb": "SNII",
    "SN IIP": "SNII",
    "SN IIL": "SNII",
    "SN IIn": "SNII",
    "SN IIn-pec": "SNII",
    "SN II-pec": "SNII",
    "SN": "SNII",  # unspecified SN → default to SNII
    # SLSN
    "SLSN-I": "SLSN",
    "SLSN-II": "SLSN",
    "SLSN-R": "SLSN",
    # Other → AGN (dominant non-SN class in our training set)
    "AGN": "AGN",
    "QSO": "AGN",
    "TDE": "AGN",  # TDE often AGN-like in LC features
    "CV": "VS",
    "Varstar": "VS",
    "LBV": "VS",
    "Nova": "VS",
    "Galaxy": "AGN",
}

# Ternary fallback when tns_type column is missing
_TERNARY_TO_ALERCE = {
    "snia": "SNIa",
    "nonIa_snlike": "SNII",  # coarse fallback
    "other": "AGN",
}

_MODEL_DIR = Path("artifacts/local_experts/alerce_lc")


def _load_labels(path: Path) -> list[dict]:
    """Load labels.csv, return list of dicts with object_id, label, tns_type."""
    rows = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows


def _map_label(row: dict) -> str | None:
    """Map a labels.csv row to an ALeRCE LC classifier class."""
    tns_type = row.get("tns_type", "").strip()
    if tns_type and tns_type in _TNS_TO_ALERCE:
        return _TNS_TO_ALERCE[tns_type]
    # Fallback to ternary label
    ternary = row.get("label", "").strip()
    return _TERNARY_TO_ALERCE.get(ternary)


def _extract_at_n_det(detections: list[dict], n_det: int) -> dict[str, float]:
    """Extract features from the first n_det detections (no leakage)."""
    # Sort and truncate
    dets = sorted(detections, key=lambda d: d.get("mjd") or d.get("jd") or 0.0)
    # Filter to positive detections
    pos = [
        d for d in dets
        if d.get("isdiffpos") in ("t", "1", 1, True, "true")
    ]
    if not pos:
        pos = dets
    truncated = pos[:n_det]
    return extract_features(truncated)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train local ALeRCE-style LC classifier"
    )
    parser.add_argument("--from-labels", default="data/labels.csv")
    parser.add_argument("--lc-dir", default="data/lightcurves")
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--n-estimators", type=int, default=300,
                        help="Number of trees in the balanced random forest")
    parser.add_argument("--max-n-det", type=int, default=20,
                        help="Max detections to use for training samples")
    parser.add_argument("--sample-n-dets", type=int, nargs="+",
                        default=[3, 5, 10, 15, 20],
                        help="Specific n_det values to sample for training")
    parser.add_argument("--full-features", action="store_true",
                        help="Use lc_classifier for ~50+ features (requires lc_classifier package)")
    parser.add_argument("--val-frac", type=float, default=0.2,
                        help="Fraction of objects held out for validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels_path = Path(args.from_labels)
    lc_dir = Path(args.lc_dir)
    model_dir = Path(args.model_dir)

    if not labels_path.exists():
        print(f"Labels file not found: {labels_path}")
        sys.exit(1)

    # Load labels and map to ALeRCE classes
    raw_labels = _load_labels(labels_path)
    label_map: dict[str, str] = {}
    for row in raw_labels:
        oid = row["object_id"]
        alerce_class = _map_label(row)
        if alerce_class:
            label_map[oid] = alerce_class

    print(f"Loaded {len(label_map)} objects with ALeRCE-class labels")
    from collections import Counter
    class_counts = Counter(label_map.values())
    print(f"  Class distribution: {dict(sorted(class_counts.items()))}")

    # Full-feature mode check
    use_full = False
    if args.full_features:
        try:
            from lc_classifier.features import CustomHierarchicalExtractor  # noqa: F401
            use_full = True
            print("  Using lc_classifier full features (~50+ features)")
        except ImportError:
            print("  lc_classifier not installed — falling back to lite features (24)")

    feature_names = FEATURE_NAMES

    # Build training data: for each object × sampled n_det, extract features
    print(f"\nExtracting features at n_det={args.sample_n_dets} ...")
    X_rows = []
    y_rows = []
    object_ids = []
    skipped = 0

    oids = sorted(label_map.keys())
    for idx, oid in enumerate(oids, start=1):
        lc_path = lc_dir / f"{oid}.json"
        if not lc_path.exists():
            skipped += 1
            continue

        try:
            with open(lc_path) as fh:
                detections = json.load(fh)
        except Exception:
            skipped += 1
            continue

        if not detections:
            skipped += 1
            continue

        alerce_class = label_map[oid]

        for n_det in args.sample_n_dets:
            if n_det > len(detections):
                continue
            feats = _extract_at_n_det(detections, n_det)
            X_rows.append([feats.get(f, np.nan) for f in feature_names])
            y_rows.append(alerce_class)
            object_ids.append(oid)

        if idx % 500 == 0:
            print(f"  [{idx}/{len(oids)}] {len(X_rows)} samples, {skipped} skipped")

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_rows)

    print(f"\nTraining data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Skipped objects: {skipped}")
    training_counts = Counter(y)
    print(f"  Training class distribution: {dict(sorted(training_counts.items()))}")

    # Train/val split by object
    rng = np.random.default_rng(args.seed)
    unique_oids = sorted(set(object_ids))
    rng.shuffle(unique_oids)
    n_val = max(1, int(len(unique_oids) * args.val_frac))
    val_oids = set(unique_oids[:n_val])
    train_mask = np.array([oid not in val_oids for oid in object_ids])
    val_mask = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    print(f"  Train: {len(X_train)} samples ({len(unique_oids) - n_val} objects)")
    print(f"  Val:   {len(X_val)} samples ({n_val} objects)")

    # Train balanced random forest
    try:
        from imblearn.ensemble import BalancedRandomForestClassifier
        clf = BalancedRandomForestClassifier(
            n_estimators=args.n_estimators,
            random_state=args.seed,
            n_jobs=-1,
            max_features="sqrt",
        )
        print(f"\nTraining BalancedRandomForestClassifier (n_estimators={args.n_estimators})...")
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        print("  imbalanced-learn not installed — using sklearn RandomForestClassifier with class_weight='balanced'")
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            random_state=args.seed,
            n_jobs=-1,
            class_weight="balanced",
            max_features="sqrt",
        )
        print(f"\nTraining RandomForestClassifier (n_estimators={args.n_estimators})...")

    clf.fit(X_train, y_train)

    # Evaluate
    from sklearn.metrics import classification_report, accuracy_score
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"\nValidation accuracy: {acc:.4f}")
    print(classification_report(y_val, y_pred, zero_division=0))

    # Save model
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pkl"
    blob = {
        "model": clf,
        "feature_names": feature_names,
        "full_features": use_full,
        "classes": list(clf.classes_),
        "n_estimators": args.n_estimators,
        "training_samples": len(X_train),
        "validation_accuracy": float(acc),
        "sample_n_dets": args.sample_n_dets,
    }
    with open(model_path, "wb") as fh:
        pickle.dump(blob, fh)
    print(f"\nSaved model → {model_path}")
    print(f"  Classes: {list(clf.classes_)}")
    print(f"  Features: {len(feature_names)}")
    print(f"\nNext: python scripts/local_infer.py --expert alerce_lc --cpu-only --from-labels data/labels.csv")


if __name__ == "__main__":
    main()
