"""Build unified truth table from multiple sources.

Merges truth from:
  1. TNS crossmatch (spectroscopic — strongest)
  2. Fink LSST crossmatch fields (xm_tns_type, xm_simbad_otype)
  3. Broker consensus (≥2 experts agree on ternary class)
  4. Host galaxy context (SIMBAD=G + photo-z + Gaia non-stellar)
  5. ALeRCE stamp class (weakest)

Science note: SIMBAD type 'G' (Galaxy) means HOST GALAXY association.
A transient at a known galaxy position is MORE likely a real SN.

Usage:
    python3 scripts/build_truth_multisource.py
    python3 scripts/build_truth_multisource.py --labels data/labels.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

# TNS type string → ternary class mapping
_TNS_TYPE_MAP = {
    "SN Ia": "snia", "SN Ia-91T-like": "snia", "SN Ia-91bg-like": "snia",
    "SN Ia-CSM": "snia", "SN Ia-pec": "snia", "SN Ia-SC": "snia",
    "SN Iax[02cx-like]": "snia",
    "SN II": "nonIa_snlike", "SN IIP": "nonIa_snlike", "SN IIL": "nonIa_snlike",
    "SN IIn": "nonIa_snlike", "SN IIb": "nonIa_snlike",
    "SN Ib": "nonIa_snlike", "SN Ic": "nonIa_snlike", "SN Ic-BL": "nonIa_snlike",
    "SN Ibc": "nonIa_snlike", "SN I": "nonIa_snlike",
    "SLSN-I": "nonIa_snlike", "SLSN-II": "nonIa_snlike", "SLSN-R": "nonIa_snlike",
}


def _classify_tns_type(tns_type: str | None) -> str | None:
    """Map TNS type string to ternary class."""
    if not tns_type or str(tns_type) in ("None", "nan", ""):
        return None
    tns_type = str(tns_type).strip()
    # Direct match
    if tns_type in _TNS_TYPE_MAP:
        return _TNS_TYPE_MAP[tns_type]
    # Prefix match
    for prefix, cls in _TNS_TYPE_MAP.items():
        if tns_type.startswith(prefix):
            return cls
    # Non-SN TNS types → other
    if tns_type.startswith("AGN") or tns_type.startswith("TDE"):
        return "other"
    return None


def build_truth_multisource(
    *,
    labels_path: Path,
    silver_dir: Path,
    existing_truth_path: Path | None,
    output_path: Path,
) -> Path:
    """Build unified truth table from multiple sources."""

    # Load existing TNS truth (strongest source)
    truth: dict[str, dict[str, Any]] = {}
    if existing_truth_path and existing_truth_path.exists():
        df = pd.read_parquet(existing_truth_path)
        for _, row in df.iterrows():
            oid = str(row["object_id"])
            truth[oid] = row.to_dict()
        print(f"  Loaded {len(truth)} existing truth rows from TNS crossmatch")

    # Load labels for object list
    with open(labels_path) as fh:
        labels = list(csv.DictReader(fh))
    all_oids = [row["object_id"] for row in labels]
    print(f"  Objects in labels: {len(all_oids)}")

    # --- Source 2: Fink LSST crossmatch fields ---
    fink_xm_count = 0
    broker_events_path = silver_dir / "broker_events.parquet"
    if broker_events_path.exists():
        events_df = pd.read_parquet(broker_events_path)
        xm_events = events_df[events_df["expert_key"] == "fink_lsst/crossmatch"]
        for _, row in xm_events.iterrows():
            oid = str(row["object_id"])
            if oid in truth and truth[oid].get("label_quality") == "spectroscopic":
                continue  # don't downgrade existing spectroscopic truth
            try:
                xm = json.loads(row["raw_label_or_score"])
            except (json.JSONDecodeError, TypeError):
                continue

            tns_type = xm.get("fink_xm_tns_type")
            tns_name = xm.get("fink_xm_tns_fullname")
            simbad = xm.get("fink_xm_simbad_otype")
            pstar = xm.get("fink_xm_legacydr8_pstar")
            zphot = xm.get("fink_xm_legacydr8_zphot")
            plx = xm.get("fink_xm_gaiadr3_Plx")

            # Tier 1: TNS spectroscopic type from Fink crossmatch
            ternary = _classify_tns_type(tns_type)
            if ternary:
                truth[oid] = {
                    "object_id": oid,
                    "final_class_ternary": ternary,
                    "follow_proxy": int(ternary == "snia"),
                    "label_source": f"fink_xm_tns ({tns_name})",
                    "label_quality": "spectroscopic",
                    "tns_name": tns_name,
                    "tns_type": tns_type,
                    "redshift": xm.get("fink_xm_tns_redshift"),
                }
                fink_xm_count += 1
                continue

            # Tier 2: TNS name without type (AT prefix = astronomical transient)
            if tns_name and str(tns_name).startswith("AT"):
                if oid not in truth:
                    truth[oid] = {
                        "object_id": oid,
                        "final_class_ternary": None,  # unknown subtype
                        "follow_proxy": None,
                        "label_source": f"fink_xm_tns ({tns_name})",
                        "label_quality": "tns_untyped",
                        "tns_name": tns_name,
                    }
                    fink_xm_count += 1
                continue

            # Tier 4: Host galaxy context
            # SIMBAD=G means HOST GALAXY (SN come from galaxies!)
            # Combined with pstar < 0.1 → extragalactic context
            if simbad == "G" and pstar is not None:
                try:
                    if float(pstar) < 0.1 and oid not in truth:
                        truth[oid] = {
                            "object_id": oid,
                            "final_class_ternary": None,  # galaxy host, class unknown
                            "follow_proxy": None,
                            "label_source": "fink_xm_host_context",
                            "label_quality": "context",
                            "host_simbad": simbad,
                            "host_zphot": zphot,
                            "host_pstar": pstar,
                        }
                        fink_xm_count += 1
                except (TypeError, ValueError):
                    pass

            # Gaia stellar rejection
            if plx is not None:
                try:
                    if float(plx) > 1.0:  # significant parallax → star
                        if oid not in truth or truth[oid].get("label_quality") in ("weak", "context"):
                            truth[oid] = {
                                "object_id": oid,
                                "final_class_ternary": "other",
                                "follow_proxy": 0,
                                "label_source": "fink_xm_gaia_stellar",
                                "label_quality": "context",
                                "gaia_parallax": float(plx),
                            }
                            fink_xm_count += 1
                except (TypeError, ValueError):
                    pass

    print(f"  Fink crossmatch: {fink_xm_count} truth rows added/updated")

    # --- Source 5: ALeRCE stamp labels (weakest) ---
    stamp_count = 0
    for row in labels:
        oid = row["object_id"]
        if oid in truth:
            continue  # don't overwrite better truth
        label = row.get("label", "").strip()
        if label:
            ternary = label if label in ("snia", "nonIa_snlike", "other") else None
            truth[oid] = {
                "object_id": oid,
                "final_class_ternary": ternary,
                "follow_proxy": int(ternary == "snia") if ternary else None,
                "label_source": "labels_csv",
                "label_quality": "weak",
            }
            stamp_count += 1

    print(f"  Labels CSV (weak): {stamp_count} truth rows added")

    # Build output DataFrame
    rows = list(truth.values())
    if not rows:
        print("  WARNING: No truth rows generated")
        rows = [{"object_id": "placeholder"}]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)

    # Summary
    quality_counts = df["label_quality"].value_counts().to_dict() if "label_quality" in df.columns else {}
    print(f"\n  Truth table: {len(df)} objects → {output_path}")
    for q, n in sorted(quality_counts.items(), key=lambda x: -x[1]):
        print(f"    {q}: {n}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multi-source truth table")
    parser.add_argument("--labels", default="data/labels.csv")
    parser.add_argument("--silver-dir", default="data/silver")
    parser.add_argument("--existing-truth", default="data/truth/object_truth.parquet",
                        help="Existing TNS crossmatch truth (merged, not overwritten)")
    parser.add_argument("--output", default="data/truth/object_truth.parquet")
    args = parser.parse_args()

    existing = Path(args.existing_truth) if Path(args.existing_truth).exists() else None

    build_truth_multisource(
        labels_path=Path(args.labels),
        silver_dir=Path(args.silver_dir),
        existing_truth_path=existing,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
