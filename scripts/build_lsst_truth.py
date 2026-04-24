"""Build LSST truth rows from ALeRCE stamp + Fink xm + broker consensus.

Applies the Tier 1-4 rule defined in docs/execplan_lsst_truth_gap.md §2.

Inputs (all on SCC):
  data/silver/broker_events.parquet       (for fink_lsst/crossmatch)
  data/gold/object_epoch_snapshots_safe_v5.parquet  (for latest-epoch consensus)
  data/truth_candidates/alerce_stamp_lsst.parquet   (from Step 1)
  data/labels.csv                          (LSST id list)

Output:
  data/truth_candidates/object_truth_lsst.parquet   (schema matching object_truth.parquet)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from debass_meta.access.identifiers import infer_identifier_kind


_TNS_IA_RE = re.compile(r"^SN\s*Ia", re.I)
_TNS_NONIA_RE = re.compile(r"^SN\s*(II|Ib|Ic|IIn|Ibc|SLSN)", re.I)
_TNS_OTHER_RE = re.compile(r"^(AGN|QSO|TDE|CV)", re.I)

_CONSENSUS_EXPERTS = [
    "proj__fink_lsst__snn__p_snia",
    "proj__fink_lsst__cats__p_snia",
    "proj__pittgoogle__supernnova_lsst__p_snia",
]


def tier1_spec(tns_type: str | None) -> tuple[str, str] | None:
    """Return (ternary, label_source) or None if unclassifiable."""
    if not tns_type or not isinstance(tns_type, str):
        return None
    t = tns_type.strip()
    if _TNS_IA_RE.search(t):
        return ("snia", "fink_lsst_xm_tns_spec")
    if _TNS_NONIA_RE.search(t):
        return ("nonIa_snlike", "fink_lsst_xm_tns_spec")
    if _TNS_OTHER_RE.search(t):
        return ("other", "fink_lsst_xm_tns_spec")
    return None


def tier4_context(simbad: str | None, zphot: float | None, plx: float | None) -> tuple[str, str] | None:
    """Host-context weak labels."""
    # Galactic stellar exclusion
    try:
        plx_f = float(plx) if plx is not None else None
    except (TypeError, ValueError):
        plx_f = None
    if plx_f is not None and np.isfinite(plx_f) and plx_f > 1.0:
        return ("other", "gaia_parallax_exclusion")
    if simbad in {"QSO", "Sy1", "Sy2", "AGN"}:
        return ("other", "fink_lsst_xm_simbad_agn")
    if simbad in {"*", "RR*", "EB*", "CV"}:
        return ("other", "fink_lsst_xm_stellar")
    try:
        zp_f = float(zphot) if zphot is not None else None
    except (TypeError, ValueError):
        zp_f = None
    if simbad in {"G", "rG", "GiG", "GiC", "BiC"} and zp_f is not None and np.isfinite(zp_f):
        return ("nonIa_snlike", "fink_lsst_xm_host_context")
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver", default="data/silver/broker_events.parquet")
    ap.add_argument("--snapshot", default="data/gold/object_epoch_snapshots_safe_v5.parquet")
    ap.add_argument("--stamp", default="data/truth_candidates/alerce_stamp_lsst.parquet")
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--output", default="data/truth_candidates/object_truth_lsst.parquet")
    ap.add_argument("--tier2-mode", choices=["stamp_only", "consensus"], default="stamp_only",
                    help="stamp_only (v5c default, no circular feedback) or consensus (legacy v5b, reads broker p_snia)")
    args = ap.parse_args()
    print(f"tier2_mode = {args.tier2_mode}")

    # --- Load LSST id list ---
    d = pd.read_csv(args.labels)
    d["id_kind"] = d["object_id"].astype(str).apply(infer_identifier_kind)
    lsst_ids = d[d["id_kind"] == "lsst_dia_object_id"]["object_id"].astype(str).tolist()
    print(f"LSST ids: {len(lsst_ids)}")

    # --- Load Fink xm fields from silver ---
    silver = pd.read_parquet(args.silver, columns=["object_id", "expert_key", "raw_label_or_score"])
    silver["object_id"] = silver["object_id"].astype(str)
    xm_rows = silver[silver["expert_key"] == "fink_lsst/crossmatch"]
    xm_map = {}
    for _, r in xm_rows.iterrows():
        try:
            payload = json.loads(r["raw_label_or_score"]) if isinstance(r["raw_label_or_score"], str) else {}
        except Exception:
            payload = {}
        xm_map[r["object_id"]] = {
            "tns_fullname": payload.get("fink_xm_tns_fullname"),
            "tns_type": payload.get("fink_xm_tns_type"),
            "simbad_otype": payload.get("fink_xm_simbad_otype"),
            "legacydr8_zphot": payload.get("fink_xm_legacydr8_zphot"),
            "gaiadr3_Plx": payload.get("fink_xm_gaiadr3_Plx"),
        }
    print(f"Fink xm coverage: {len(xm_map)} / {len(lsst_ids)}")

    # --- Load ALeRCE stamp classifier ---
    stamp = pd.read_parquet(args.stamp)
    stamp["object_id"] = stamp["object_id"].astype(str)
    stamp_map = stamp.set_index("object_id")[["stamp_class", "stamp_prob", "stamp_classifier_version"]].to_dict("index")
    print(f"ALeRCE stamp coverage: {stamp['stamp_class'].notna().sum()} / {len(lsst_ids)}")

    # --- Latest-epoch broker consensus from gold snapshot ---
    snap = pd.read_parquet(args.snapshot, columns=["object_id", "n_det", *_CONSENSUS_EXPERTS])
    snap["object_id"] = snap["object_id"].astype(str)
    snap_lsst = snap[snap["object_id"].isin(lsst_ids)]
    latest = snap_lsst.sort_values("n_det").groupby("object_id").tail(1).set_index("object_id")
    print(f"Snapshot latest-epoch coverage for LSST: {len(latest)}")

    # --- Apply tier rule ---
    truth_rows = []
    tier_counts = {1: 0, 2: 0, 3: 0, 4: 0, "drop": 0}
    for oid in lsst_ids:
        xm = xm_map.get(oid, {})
        st = stamp_map.get(oid, {})

        ternary = None
        label_source = None
        label_quality = None
        tier = None

        # Tier 1 — spectroscopic
        t1 = tier1_spec(xm.get("tns_type"))
        if t1:
            ternary, label_source = t1
            label_quality = "spectroscopic"
            tier = 1

        # Tier 2 — stamp-based weak label.
        # In stamp_only mode (v5c default), we NEVER fabricate snia positives from weak
        # signals. SN-stamp rows get "nonIa_snlike" so the followup model's positive
        # class is always spectroscopic; weak rows only teach "this is not-Ia".
        # In consensus mode (v5b legacy), reads broker p_snia — creates circularity
        # because those same p_snia scores are followup features. Retained for A/B only.
        if ternary is None and st.get("stamp_class") and float(st.get("stamp_prob") or 0) >= 0.70:
            stamp_class = st["stamp_class"]
            if args.tier2_mode == "consensus":
                if float(st.get("stamp_prob") or 0) >= 0.85:
                    p_snia_vals = []
                    if oid in latest.index:
                        for c in _CONSENSUS_EXPERTS:
                            v = latest.at[oid, c]
                            if pd.notna(v):
                                p_snia_vals.append(float(v))
                    if stamp_class == "SN":
                        if p_snia_vals and max(p_snia_vals) >= 0.5:
                            ternary = "snia"
                        elif p_snia_vals and sum(1 for v in p_snia_vals if v < 0.3) >= 2:
                            ternary = "nonIa_snlike"
                        else:
                            ternary = "nonIa_snlike"
                    elif stamp_class in {"AGN", "VS", "asteroid", "star", "bogus"}:
                        ternary = "other"
                    if ternary is not None:
                        label_source = f"alerce_stamp_lsst+consensus ({stamp_class})"
                        label_quality = "weak"
                        tier = 2
            else:  # stamp_only (default)
                if stamp_class == "SN":
                    ternary = "nonIa_snlike"
                elif stamp_class in {"AGN", "VS", "asteroid", "star", "bogus"}:
                    ternary = "other"
                if ternary is not None:
                    label_source = f"alerce_stamp_lsst ({stamp_class})"
                    label_quality = "weak"
                    tier = 2

        # Tier 3 — stamp only in the 0.70-0.85 band (only reached in consensus mode;
        # in stamp_only mode, Tier 2 already covers >=0.70).
        if args.tier2_mode == "consensus" and ternary is None and st.get("stamp_class") and 0.70 <= float(st.get("stamp_prob") or 0) < 0.85:
            stamp_class = st["stamp_class"]
            if stamp_class == "SN":
                ternary = "nonIa_snlike"
            elif stamp_class in {"AGN", "VS", "asteroid", "star", "bogus"}:
                ternary = "other"
            if ternary is not None:
                label_source = f"alerce_stamp_lsst ({stamp_class})"
                label_quality = "weak"
                tier = 3

        # Tier 4 — context
        if ternary is None:
            t4 = tier4_context(xm.get("simbad_otype"), xm.get("legacydr8_zphot"), xm.get("gaiadr3_Plx"))
            if t4:
                ternary, label_source = t4
                label_quality = "context"
                tier = 4

        if ternary is None:
            tier_counts["drop"] += 1
            continue
        tier_counts[tier] += 1

        # Build truth row matching object_truth.parquet schema
        truth_rows.append({
            "object_id": oid,
            "final_class_ternary": ternary,
            "follow_proxy": int(ternary == "snia"),
            "label_source": label_source,
            "label_quality": label_quality,
            "bts_type": None,
            "tns_name": xm.get("tns_fullname"),
            "redshift": xm.get("legacydr8_zphot"),
            "final_class_raw": xm.get("tns_type") or st.get("stamp_class"),
            "truth_timestamp": time.time(),
            "tns_prefix": None,
            "tns_type": xm.get("tns_type"),
            "tns_has_spectra": (1 if xm.get("tns_type") else 0),
            "tns_redshift": None,
            "tns_ra": None,
            "tns_dec": None,
            "tns_discovery_date": None,
            "consensus_experts": None,
            "consensus_n_agree": None,
            "consensus_n_total": None,
        })

    df = pd.DataFrame(truth_rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"\n=== Tier counts (of {len(lsst_ids)}) ===")
    for k, v in tier_counts.items():
        print(f"  tier {k}: {v} ({v / len(lsst_ids) * 100:.1f}%)")
    print(f"  covered: {len(df)} / {len(lsst_ids)} ({len(df) / len(lsst_ids) * 100:.1f}%)")
    print()
    print("=== final_class_ternary distribution ===")
    print(df["final_class_ternary"].value_counts(dropna=False).to_string())
    print()
    print("=== label_quality distribution ===")
    print(df["label_quality"].value_counts(dropna=False).to_string())
    print(f"\nwrote {len(df)} LSST truth rows -> {out_path}")


if __name__ == "__main__":
    main()
