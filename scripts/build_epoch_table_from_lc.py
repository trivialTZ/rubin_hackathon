"""scripts/build_epoch_table_from_lc.py

Build the (object_id, n_det) epoch feature table directly from cached lightcurves.

This is the NO-LEAKAGE version:
  - For each object, lightcurve is truncated to exactly n_det detections
  - Features are computed ONLY from detections 1..n_det
  - ALeRCE/Fink broker scores are fetched separately and merged in
    (they provide the same object-level score at every epoch, but
     the LC features genuinely change with n_det)
  - Local expert scores (SNN, ParSNIP) are NaN until SCC GPU job fills them in

Output: data/epochs/epoch_features.parquet

Usage:
    python scripts/build_epoch_table_from_lc.py
    python scripts/build_epoch_table_from_lc.py --max-n-det 10
    python scripts/build_epoch_table_from_lc.py --lc-dir data/lightcurves --labels data/labels.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from debass_meta.features.lightcurve import extract_features_at_each_epoch, FEATURE_NAMES

# Broker score columns to merge from silver parquet (object-level, same at all epochs)
_BROKER_SCORE_COLS = [
    ("alerce", "lc_classifier_transient_SNIa",   "alerce_SNIa"),
    ("alerce", "lc_classifier_transient_SNIbc",  "alerce_SNIbc"),
    ("alerce", "lc_classifier_transient_SNII",   "alerce_SNII"),
    ("alerce", "lc_classifier_transient_SLSN",   "alerce_SLSN"),
    ("alerce", "lc_classifier_top_Transient",    "alerce_top_Transient"),
    ("alerce", "stamp_classifier_SN",            "alerce_stamp_SN"),
    ("fink",   "rf_snia_vs_nonia",               "fink_rf_snia"),
    ("fink",   "snn_snia_vs_nonia",              "fink_snn_snia"),
    ("fink",   "snn_sn_vs_all",                  "fink_snn_sn"),
    # NOTE: local expert scores (snn_prob_ia, parsnip_prob_ia) are NOT in silver
    # parquet — they are written to data/silver/local_expert_outputs.json by
    # local_infer.py. They are merged separately via _load_local_expert_json().
]

_REMOTE_BROKERS = ["alerce", "fink", "lasair", "ampel"]


def _load_labels(path: Path) -> dict[str, str]:
    labels = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            labels[row["object_id"]] = row["label"]
    return labels


def _build_silver_lookup(silver_dir: Path) -> dict[tuple, float | None]:
    """Return (object_id, broker, field) -> canonical_projection from silver parquet."""
    silver_path = silver_dir / "broker_outputs.parquet"
    if not silver_path.exists():
        return {}
    try:
        import pandas as pd
        df = pd.read_parquet(silver_path)
        lookup = {}
        for _, row in df.iterrows():
            key = (str(row["object_id"]), str(row["broker"]), str(row["field"]))
            lookup[key] = row.get("canonical_projection")
        return lookup
    except Exception:
        return {}


def _load_local_expert_json(
    silver_dir: Path,
) -> dict[tuple[str, int], dict[str, float | None]]:
    """Load per-(object_id, n_det) local expert scores from local_expert_outputs.json.

    Returns {(object_id, n_det): {"snn_prob_ia": float|None, "parsnip_prob_ia": float|None}}.
    Written by scripts/local_infer.py (run on SCC GPU node).
    Returns empty dict when the file doesn't exist (pre-GPU run).
    """
    path = silver_dir / "local_expert_outputs.json"
    if not path.exists():
        return {}
    try:
        with open(path) as fh:
            records = json.load(fh)
        out: dict[tuple[str, int], dict[str, float | None]] = {}
        for rec in records:
            oid = rec.get("object_id")
            n_det = rec.get("n_det")
            if not oid or n_det is None:
                continue
            key = (str(oid), int(n_det))
            if key not in out:
                out[key] = {"snn_prob_ia": None, "parsnip_prob_ia": None}
            expert = rec.get("expert", "")
            probs = rec.get("class_probabilities", {})
            # SuperNNova / ParSNIP both output "SN Ia" as the Ia class key
            ia_prob = probs.get("SN Ia") if probs.get("SN Ia") is not None else probs.get("prob_ia")
            if expert == "supernnova":
                out[key]["snn_prob_ia"] = ia_prob
            elif expert == "parsnip":
                out[key]["parsnip_prob_ia"] = ia_prob
        return out
    except Exception:
        return {}


def _build_remote_broker_availability(
    silver_lookup: dict[tuple[str, str, str], float | None],
) -> dict[tuple[str, str], bool]:
    availability: dict[tuple[str, str], bool] = {}
    for (object_id, broker, _field), value in silver_lookup.items():
        key = (object_id, broker)
        availability[key] = availability.get(key, False) or value is not None
    return availability


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build no-leakage epoch feature table from lightcurves"
    )
    parser.add_argument("--lc-dir",     default="data/lightcurves")
    parser.add_argument("--silver-dir", default="data/silver")
    parser.add_argument("--epochs-dir", default="data/epochs")
    parser.add_argument("--labels",     default="data/labels.csv")
    parser.add_argument("--max-n-det",  type=int, default=20)
    parser.add_argument("--min-n-det",  type=int, default=1,
                        help="Only include epochs with n_det >= this (filter very sparse LCs)")
    args = parser.parse_args()

    lc_dir     = Path(args.lc_dir)
    silver_dir = Path(args.silver_dir)
    epochs_dir = Path(args.epochs_dir)
    labels_path = Path(args.labels)

    # Load labels
    label_map: dict[str, str] = {}
    if labels_path.exists():
        label_map = _load_labels(labels_path)
        print(f"Loaded {len(label_map)} labels from {labels_path}")
    else:
        print(f"No labels file at {labels_path} — epoch table will have no target_label")

    # Load silver broker scores for merging
    silver_lookup = _build_silver_lookup(silver_dir)
    print(f"Silver score lookup: {len(silver_lookup)} entries")
    remote_broker_availability = _build_remote_broker_availability(silver_lookup)

    # Load local expert scores (written by local_infer.py on SCC GPU node)
    local_expert_lookup = _load_local_expert_json(silver_dir)
    if local_expert_lookup:
        print(f"Local expert scores loaded for {len(local_expert_lookup)} objects")
    else:
        print("No local expert scores found (snn_prob_ia/parsnip_prob_ia will be NaN — run local_infer_gpu.sh on SCC)")

    # Process each lightcurve file
    lc_files = sorted(lc_dir.glob("*.json"))
    if not lc_files:
        print(f"No lightcurve files found in {lc_dir}")
        print("Run: python scripts/download_alerce_training.py")
        sys.exit(1)

    print(f"Processing {len(lc_files)} lightcurves (max_n_det={args.max_n_det})...")
    print(f"PROGRESS build phase=process current=0 total={len(lc_files)} rows=0 skipped=0")

    import pandas as pd
    import numpy as np

    all_rows = []
    skipped = 0

    for idx, lc_path in enumerate(lc_files, start=1):
        oid = lc_path.stem
        try:
            with open(lc_path) as fh:
                detections = json.load(fh)
        except Exception as e:
            print(f"  {oid}: skip (read error: {e})")
            skipped += 1
            continue

        if not detections:
            skipped += 1
            continue

        # Extract per-epoch LC features (no leakage — truncated at each n_det)
        epoch_feats = extract_features_at_each_epoch(
            detections, max_n_det=args.max_n_det
        )

        if not epoch_feats:
            skipped += 1
            continue

        for feats in epoch_feats:
            n_det = int(feats["n_det"])
            if n_det < args.min_n_det:
                continue

            row: dict = {
                "object_id": oid,
                "n_det": n_det,
                "alert_mjd": feats.pop("alert_mjd", np.nan),
                "target_label": label_map.get(oid),
            }

            # Add LC features
            for fname in FEATURE_NAMES:
                row[fname] = feats.get(fname, np.nan)

            # Merge broker scores from silver parquet (object-level, same at all epochs)
            for broker, field, col_name in _BROKER_SCORE_COLS:
                key = (oid, broker, field)
                row[col_name] = silver_lookup.get(key)  # None -> NaN

            # Merge local expert scores from JSON (written by local_infer.py on GPU)
            # Keyed by (object_id, n_det) so each epoch gets the score from the
            # correctly truncated lightcurve — no leakage from future detections.
            expert_scores = local_expert_lookup.get((oid, n_det), {})
            row["snn_prob_ia"]     = expert_scores.get("snn_prob_ia")
            row["parsnip_prob_ia"] = expert_scores.get("parsnip_prob_ia")

            # Availability flags.
            # Remote brokers are object-level features from the silver parquet.
            # Local experts are per-(object_id, n_det), so the availability flag
            # must reflect whether that epoch-specific score was actually written.
            for b in _REMOTE_BROKERS:
                row[f"{b}_available"] = float(remote_broker_availability.get((oid, b), False))
            row["supernnova_available"] = float(expert_scores.get("snn_prob_ia") is not None)
            row["parsnip_available"] = float(expert_scores.get("parsnip_prob_ia") is not None)

            all_rows.append(row)

        if idx % 25 == 0 or idx == len(lc_files):
            print(
                f"PROGRESS build phase=process current={idx} total={len(lc_files)} "
                f"rows={len(all_rows)} skipped={skipped}"
            )

    if not all_rows:
        print("No epoch rows built. Check lightcurve files and labels.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)

    # Summary
    print(f"\nEpoch table summary:")
    print(f"  Total rows:    {len(df)}")
    print(f"  Objects:       {df['object_id'].nunique()}")
    print(f"  Skipped LCs:   {skipped}")
    print(f"  n_det range:   {df['n_det'].min()} – {df['n_det'].max()}")
    if df['target_label'].notna().any():
        print(f"  Label counts:  {df[df['target_label'].notna()]['target_label'].value_counts().to_dict()}")
    early = df[df['n_det'] <= 5]
    print(f"  Rows n_det≤5:  {len(early)}")

    # Verify no leakage: mag_std and mag_range at n_det=1 should be 0
    n1 = df[df['n_det'] == 1]
    n5 = df[df['n_det'] == 5]
    for col in ['mag_std', 'mag_range', 'dmag_dt', 'color_gr']:
        if col not in df.columns:
            continue
        v1 = n1[col].describe()[['mean','min','max']].to_dict()
        v5 = n5[col].describe()[['mean','min','max']].to_dict() if len(n5) > 0 else {}
        print(f"  {col}: n_det=1 → {v1}  |  n_det=5 → {v5}")

    # Write
    epochs_dir.mkdir(parents=True, exist_ok=True)
    out_path = epochs_dir / "epoch_features.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path}")
    print(
        f"PROGRESS build phase=complete current={len(lc_files)} total={len(lc_files)} "
        f"rows={len(df)} skipped={skipped}"
    )
    print(f"Next: python scripts/train_early.py")


if __name__ == "__main__":
    main()
