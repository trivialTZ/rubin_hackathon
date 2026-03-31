"""Build (object_id, n_det) epoch feature table from lightcurves + broker history.

This is the central data structure for the early-epoch meta-classifier.
Each row = one (object_id, n_det) pair with:
  - All available broker scores at that epoch
  - Availability mask (which brokers had data)
  - Lightcurve summary features at that n_det
  - Ground-truth label (if available)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_LC_DIR = Path("data/lightcurves")
_SILVER_DIR = Path("data/silver")
_EPOCHS_DIR = Path("data/epochs")

# Broker score fields to extract into flat columns
# Maps column_name -> (broker, field_suffix)
_FEATURE_MAP = {
    # ALeRCE transient classifier
    "alerce_SNIa":          ("alerce", "lc_classifier_transient_SNIa"),
    "alerce_SNIbc":         ("alerce", "lc_classifier_transient_SNIbc"),
    "alerce_SNII":          ("alerce", "lc_classifier_transient_SNII"),
    "alerce_SLSN":          ("alerce", "lc_classifier_transient_SLSN"),
    # ALeRCE top-level
    "alerce_top_Transient": ("alerce", "lc_classifier_top_Transient"),
    # ALeRCE stamp
    "alerce_stamp_SN":      ("alerce", "stamp_classifier_SN"),
    # Fink probability scores
    "fink_rf_snia":         ("fink",   "rf_snia_vs_nonia"),
    "fink_snn_snia":        ("fink",   "snn_snia_vs_nonia"),
    "fink_snn_sn":          ("fink",   "snn_sn_vs_all"),
    # Local experts (filled in by SCC GPU job, NaN until then)
    "snn_prob_ia":          ("supernnova", "prob_ia"),
    "parsnip_prob_ia":      ("parsnip",    "prob_ia"),
}

_ALL_BROKERS = ["alerce", "fink", "lasair", "supernnova", "parsnip", "ampel"]


def build_epoch_table(
    lc_dir: Path = _LC_DIR,
    silver_dir: Path = _SILVER_DIR,
    epochs_dir: Path = _EPOCHS_DIR,
    label_map: dict[str, str] | None = None,
    max_n_det: int = 20,
) -> Path:
    """Build epoch feature table and write to epochs_dir/epoch_features.parquet.

    Returns path of written file.
    """
    try:
        import pandas as pd
        import numpy as np
    except ImportError as e:
        raise RuntimeError("pandas/numpy required") from e

    label_map = label_map or {}

    # Load silver broker outputs for score lookup
    silver_path = silver_dir / "broker_outputs.parquet"
    silver_df = pd.read_parquet(silver_path) if silver_path.exists() else pd.DataFrame()

    # Build a lookup: (object_id, broker, field) -> canonical_projection
    score_lookup: dict[tuple, Any] = {}
    if not silver_df.empty:
        for _, row in silver_df.iterrows():
            key = (str(row["object_id"]), str(row["broker"]), str(row["field"]))
            score_lookup[key] = row.get("canonical_projection")

    # Collect epoch rows
    rows = []
    lc_files = sorted(lc_dir.glob("*.json"))

    for lc_path in lc_files:
        object_id = lc_path.stem
        with open(lc_path) as fh:
            detections = json.load(fh)

        # Sort by MJD/JD (ALeRCE uses mjd), keep only positive detections
        def _t(d):
            return d.get("mjd") or d.get("jd") or 0

        dets = sorted(
            [d for d in detections if d.get("isdiffpos") in ("t", "1", 1, True, "true")],
            key=_t
        )
        if not dets:
            dets = sorted(detections, key=_t)

        for i in range(min(len(dets), max_n_det)):
            n_det = i + 1
            alert_mjd = float(_t(dets[i]))

            row: dict[str, Any] = {
                "object_id": object_id,
                "n_det": n_det,
                "alert_mjd": alert_mjd,
                "target_label": label_map.get(object_id),
            }

            # Lightcurve summary features at this epoch
            lc_slice = dets[: n_det]
            mags = [d.get("magpsf") for d in lc_slice if d.get("magpsf") is not None]
            row["lc_n_det"] = n_det
            row["lc_mean_mag"] = float(np.mean(mags)) if mags else np.nan
            row["lc_std_mag"] = float(np.std(mags)) if len(mags) > 1 else np.nan
            row["lc_mag_range"] = float(max(mags) - min(mags)) if len(mags) > 1 else np.nan

            # Broker scores — object-level (ALeRCE/Fink) attached at every epoch
            for col, (broker, field) in _FEATURE_MAP.items():
                key = (object_id, broker, field)
                row[col] = score_lookup.get(key)  # None → NaN in DataFrame

            # Availability mask
            for b in _ALL_BROKERS:
                # A broker is "available" if any score for it is non-null
                b_keys = [k for k in score_lookup if k[0] == object_id and k[1] == b]
                row[f"{b}_available"] = float(any(
                    score_lookup[k] is not None for k in b_keys
                ))

            rows.append(row)

    if not rows:
        raise ValueError("No epoch rows built — run fetch_lightcurves.py first")

    epochs_dir.mkdir(parents=True, exist_ok=True)
    out_path = epochs_dir / "epoch_features.parquet"
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    return out_path
