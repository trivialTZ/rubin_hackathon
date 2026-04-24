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

# Use the canonical mapping from tns.py — single source of truth
from debass_meta.access.tns import (
    TNS_TERNARY_MAP,
    TNS_AMBIGUOUS_TYPES,
    map_tns_type_to_ternary as _classify_tns_type,
    is_ambiguous_type,
)
from debass_meta.access.identifiers import infer_identifier_kind

# LSST stamp_only Tier 2 (v5c circularity fix, ported 2026-04-24).
# NEVER maps stamp=="SN" to snia — all snia labels must come from spec truth.
_STAMP_OTHER_CLASSES = {"AGN", "VS", "asteroid", "star", "bogus"}
_STAMP_MIN_PROB = 0.70


def build_truth_multisource(
    *,
    labels_path: Path,
    silver_dir: Path,
    existing_truth_path: Path | None,
    output_path: Path,
    ztf_bts_path: Path | None = None,
    yse_dr1_path: Path | None = None,
    mpc_exclusion_path: Path | None = None,
    lsst_stamp_path: Path | None = None,
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

    # --- Source 1.5 (NEW Phase 2): ZTF BTS spectroscopic subtypes ---
    # BTS has fine-grained subtypes (Ia, Ia-CSM, Iax, II, IIn, IIb, ...) and
    # is spec-confirmed — we treat it as Tier 1 "spectroscopic" for any ZTF
    # object whose TNS truth is missing or whose TNS type is "SN" (ambiguous).
    bts_count = 0
    if ztf_bts_path and Path(ztf_bts_path).exists():
        bts_df = pd.read_parquet(ztf_bts_path)
        for _, row in bts_df.iterrows():
            oid = str(row.get("object_id") or "")
            if not oid:
                continue
            existing = truth.get(oid, {})
            existing_quality = existing.get("label_quality")
            ternary = row.get("ternary")
            # Only override if existing is weaker (or no ternary known)
            if existing_quality == "spectroscopic" and existing.get("final_class_ternary"):
                continue
            if ternary is None:
                continue
            truth[oid] = {
                "object_id": oid,
                "final_class_ternary": str(ternary),
                "follow_proxy": int(ternary == "snia"),
                "label_source": "ztf_bts",
                "label_quality": "spectroscopic",
                "bts_type": row.get("bts_type"),
                "tns_name": row.get("tns_name") or existing.get("tns_name"),
                "redshift": row.get("redshift") or existing.get("redshift"),
            }
            bts_count += 1
        print(f"  ZTF BTS: {bts_count} truth rows added/refined")

    # --- Source 1.6 (NEW Phase 2): YSE DR1 low-z SNe (cross-check) ---
    yse_count = 0
    if yse_dr1_path and Path(yse_dr1_path).exists():
        yse_df = pd.read_parquet(yse_dr1_path)
        for _, row in yse_df.iterrows():
            key = str(row.get("tns_name") or row.get("object_id") or "")
            if not key:
                continue
            ternary = row.get("ternary")
            if ternary is None:
                continue
            # YSE keyed by TNS name — look up any matching tns_name in existing
            matched = False
            for oid, entry in truth.items():
                if str(entry.get("tns_name")) == key and entry.get("label_quality") != "spectroscopic":
                    truth[oid]["yse_ternary"] = str(ternary)
                    truth[oid]["label_source"] = (entry.get("label_source", "") + "+yse_dr1").strip("+")
                    if entry.get("final_class_ternary") is None:
                        truth[oid]["final_class_ternary"] = str(ternary)
                        truth[oid]["follow_proxy"] = int(ternary == "snia")
                        truth[oid]["label_quality"] = "spectroscopic"
                    matched = True
                    yse_count += 1
                    break
            if not matched:
                pass  # YSE-only objects (not in labels) — skip
        print(f"  YSE DR1: {yse_count} truth rows cross-matched")

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
            # Note: Fink xm_tns_type comes from the TNS classification,
            # but we don't know the source group. Treat as credible IF
            # the type is specific (not ambiguous "SN").
            ternary = _classify_tns_type(tns_type)
            if ternary:
                if is_ambiguous_type(tns_type):
                    # Generic "SN" — ambiguous, downgrade to consensus
                    quality = "consensus"
                    label_source = f"fink_xm_tns_ambiguous ({tns_name})"
                else:
                    quality = "spectroscopic"
                    label_source = f"fink_xm_tns ({tns_name})"
                truth[oid] = {
                    "object_id": oid,
                    "final_class_ternary": ternary,
                    "follow_proxy": int(ternary == "snia"),
                    "label_source": label_source,
                    "label_quality": quality,
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

    # --- Source 4.5 (ported 2026-04-24, v5c stamp_only Tier 2) ---
    # LSST stamp-based weak labels. Circularity-safe: SN-stamp rows become
    # nonIa_snlike (NEVER snia — snia labels must come from spec truth).
    # Skips any object already labeled spectroscopic.
    stamp_count = 0
    if lsst_stamp_path and Path(lsst_stamp_path).exists():
        stamp_df = pd.read_parquet(lsst_stamp_path)
        stamp_df["object_id"] = stamp_df["object_id"].astype(str)
        for _, srow in stamp_df.iterrows():
            oid = str(srow["object_id"])
            # Only apply to LSST ids
            if infer_identifier_kind(oid) != "lsst_dia_object_id":
                continue
            existing = truth.get(oid)
            if existing and existing.get("label_quality") == "spectroscopic":
                continue
            stamp_class = srow.get("stamp_class")
            try:
                stamp_prob = float(srow.get("stamp_prob") or 0.0)
            except (TypeError, ValueError):
                stamp_prob = 0.0
            if not stamp_class or stamp_prob < _STAMP_MIN_PROB:
                continue
            if stamp_class == "SN":
                ternary = "nonIa_snlike"
            elif stamp_class in _STAMP_OTHER_CLASSES:
                ternary = "other"
            else:
                continue
            truth[oid] = {
                "object_id": oid,
                "final_class_ternary": ternary,
                "follow_proxy": int(ternary == "snia"),
                "label_source": f"alerce_stamp_lsst ({stamp_class})",
                "label_quality": "weak",
                "stamp_class": stamp_class,
                "stamp_prob": stamp_prob,
            }
            stamp_count += 1
        print(f"  LSST ALeRCE stamp (stamp_only Tier 2): {stamp_count} weak rows added")
    else:
        if lsst_stamp_path:
            print(f"  LSST ALeRCE stamp: not found at {lsst_stamp_path} — skipping Tier 2")

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

    # --- Exclusion (NEW Phase 2): MPC asteroid cross-match ---
    mpc_excluded = 0
    if mpc_exclusion_path and Path(mpc_exclusion_path).exists():
        mpc_df = pd.read_parquet(mpc_exclusion_path)
        asteroid_ids = set(mpc_df[mpc_df["is_asteroid"]]["object_id"].astype(str))
        for oid in list(truth.keys()):
            if oid in asteroid_ids:
                # Force asteroid objects to 'other' with explicit exclusion tag
                existing_quality = truth[oid].get("label_quality")
                if existing_quality == "spectroscopic":
                    # Don't override a TNS spec label — but flag the anomaly
                    truth[oid]["mpc_asteroid_conflict"] = True
                    continue
                truth[oid]["final_class_ternary"] = "other"
                truth[oid]["follow_proxy"] = 0
                truth[oid]["label_source"] = "mpc_exclusion"
                truth[oid]["label_quality"] = "context"
                truth[oid]["is_asteroid"] = True
                mpc_excluded += 1
        print(f"  MPC exclusion: {mpc_excluded} objects routed to 'other'")

    # Build output DataFrame
    rows = list(truth.values())
    if not rows:
        print("  WARNING: No truth rows generated")
        rows = [{"object_id": "placeholder"}]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    # Align mixed-type columns (same pattern as merge_lsst_truth.py).
    # Different tiers write different dtypes into the same column
    # (e.g. TNS writes str redshift, ALeRCE stamp tier writes None);
    # pyarrow can't infer a unified schema → cast to str|None universally.
    _str_cols = [
        "redshift", "tns_redshift", "tns_ra", "tns_dec",
        "consensus_experts", "consensus_n_agree", "consensus_n_total",
        "tns_has_spectra",
    ]
    for c in _str_cols:
        if c in df.columns:
            df[c] = df[c].apply(
                lambda v: None if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v)
            )
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
    # Phase 2: additional truth/exclusion sources
    parser.add_argument("--ztf-bts", default="data/truth/ztf_bts.parquet",
                        help="ZTF BTS spectroscopic catalog (Tier 1.5)")
    parser.add_argument("--yse-dr1", default="data/truth/yse_dr1.parquet",
                        help="YSE DR1 low-z SN cross-check (Tier 1.6)")
    parser.add_argument("--mpc-exclusion", default="data/truth/mpc_exclusion.parquet",
                        help="MPC asteroid exclusion (forces target='other')")
    parser.add_argument("--lsst-stamp", default="data/truth_candidates/alerce_stamp_lsst.parquet",
                        help="ALeRCE LSST stamp classifications (v5c stamp_only Tier 2)")
    args = parser.parse_args()

    existing = Path(args.existing_truth) if Path(args.existing_truth).exists() else None
    bts = Path(args.ztf_bts) if Path(args.ztf_bts).exists() else None
    yse = Path(args.yse_dr1) if Path(args.yse_dr1).exists() else None
    mpc = Path(args.mpc_exclusion) if Path(args.mpc_exclusion).exists() else None
    stamp = Path(args.lsst_stamp) if Path(args.lsst_stamp).exists() else None

    build_truth_multisource(
        labels_path=Path(args.labels),
        silver_dir=Path(args.silver_dir),
        existing_truth_path=existing,
        output_path=Path(args.output),
        ztf_bts_path=bts,
        yse_dr1_path=yse,
        mpc_exclusion_path=mpc,
        lsst_stamp_path=stamp,
    )


if __name__ == "__main__":
    main()
