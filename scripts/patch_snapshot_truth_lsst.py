"""Surgical truth patch: copy 4 truth columns from LSST truth into the v5 snapshot
and save as v5b — no full snapshot rebuild.

Columns patched (only where target_class is currently None):
  target_class          <- final_class_ternary
  target_follow_proxy   <- follow_proxy
  label_source          <- label_source
  label_quality         <- label_quality
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from debass_meta.access.identifiers import infer_identifier_kind


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-in", default="data/gold/object_epoch_snapshots_safe_v5.parquet")
    ap.add_argument("--lsst-truth", default="data/truth_candidates/object_truth_lsst.parquet")
    ap.add_argument("--snapshot-out", default="data/gold/object_epoch_snapshots_safe_v5b.parquet")
    args = ap.parse_args()

    snap = pd.read_parquet(args.snapshot_in)
    lsst_truth = pd.read_parquet(args.lsst_truth)
    print(f"snapshot rows: {len(snap)}")
    print(f"LSST truth rows: {len(lsst_truth)}")

    snap["object_id"] = snap["object_id"].astype(str)
    lsst_truth["object_id"] = lsst_truth["object_id"].astype(str)

    # Lookup dict keyed by oid
    lookup = lsst_truth.set_index("object_id")[
        ["final_class_ternary", "follow_proxy", "label_source", "label_quality"]
    ]

    # Target mask: LSST-kind rows where target_class is currently None
    snap["id_kind"] = snap["object_id"].apply(infer_identifier_kind)
    mask = (snap["id_kind"] == "lsst_dia_object_id") & (snap["target_class"].isna() | (snap["target_class"].astype(str) == "None"))
    print(f"rows to patch (LSST + currently-None target_class): {mask.sum()}")
    matched_oids = snap.loc[mask, "object_id"].isin(lookup.index)
    print(f"  of which match LSST truth lookup: {matched_oids.sum()}")

    # Apply patch by mapping
    for dst, src in [("target_class", "final_class_ternary"),
                     ("target_follow_proxy", "follow_proxy"),
                     ("label_source", "label_source"),
                     ("label_quality", "label_quality")]:
        snap.loc[mask, dst] = snap.loc[mask, "object_id"].map(lookup[src])

    # Drop the temporary id_kind column
    snap = snap.drop(columns=["id_kind"])

    # Verify
    final_lsst = snap[snap["object_id"].apply(infer_identifier_kind) == "lsst_dia_object_id"]
    filled = final_lsst["target_class"].notna().sum()
    print(f"\nafter patch: LSST rows with target_class = {filled} / {len(final_lsst)} ({filled / len(final_lsst) * 100:.1f}%)")

    # Save
    out = Path(args.snapshot_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    snap.to_parquet(out, index=False)
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
