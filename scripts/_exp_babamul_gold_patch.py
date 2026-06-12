"""Phase 4c (memory-light) — patch v6e.2 gold with Babamul columns into v7 gold.

Full sharded gold rebuild is the safe option but requires 7-8 min of SGE
array jobs. Since Babamul is `static_safe`, the per-object features apply
to ALL (object, n_det) snapshots — no per-epoch time-of-availability logic
needed. We can read the existing frozen v6e.2 gold once, look up the 13
babamul columns per object_id from silver, broadcast across snapshots,
and write `_v7`.

Outputs:
    data/gold/object_epoch_snapshots_safe_v7.parquet

Reads:
    data/gold/object_epoch_snapshots_safe_v6e2.parquet  (frozen)
    data/silver/broker_events.parquet                   (with babamul rows)

The frozen v6e.2 file is read-only here — we never modify it.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pandas as pd

from debass_meta.projectors import project_expert_events, sanitize_expert_key

V6E2_GOLD = REPO / "data" / "gold" / "object_epoch_snapshots_safe_v6e2.parquet"
SILVER = REPO / "data" / "silver" / "broker_events.parquet"
V7_GOLD = REPO / "data" / "gold" / "object_epoch_snapshots_safe_v7.parquet"


def main() -> int:
    if not V6E2_GOLD.exists():
        print(f"FAIL frozen v6e.2 gold not found: {V6E2_GOLD}")
        return 1
    if not SILVER.exists():
        print(f"FAIL silver not found: {SILVER}")
        return 1

    print(f"reading v6e.2 gold ({V6E2_GOLD.stat().st_size / 1e6:.0f} MB) ...")
    gold = pd.read_parquet(V6E2_GOLD)
    print(f"  rows: {len(gold):,}; cols: {len(gold.columns)}")

    print("reading babamul silver subset ...")
    sdf = pd.read_parquet(
        SILVER,
        columns=[
            "object_id", "broker", "expert_key", "field",
            "raw_label_or_score", "canonical_projection",
            "event_time_jd", "temporal_exactness", "availability",
        ],
    )
    bab = sdf[sdf["expert_key"] == "babamul"]
    print(f"  babamul rows: {len(bab):,}; unique objects: {bab['object_id'].nunique():,}")

    san = sanitize_expert_key("babamul")  # → "babamul"
    proj_cols = [
        f"proj__{san}__babamul_star_flag",
        f"proj__{san}__babamul_near_brightstar_flag",
        f"proj__{san}__babamul_rock_flag",
        f"proj__{san}__babamul_stationary_flag",
        f"proj__{san}__babamul_xmatch_lsst",
        f"proj__{san}__babamul_xmatch_ztf",
    ]
    extra_cols = [
        f"avail__{san}", f"exact__{san}",
        f"event_count__{san}", f"prediction_type__{san}",
        f"reason__{san}", f"source_event_time_jd__{san}",
        f"temporal_exactness__{san}",
    ]

    print("computing per-object babamul projections ...")
    object_to_features: dict[str, dict] = {}
    for oid, group in bab.groupby("object_id"):
        events = group.to_dict("records")
        projected = project_expert_events("babamul", events)
        feats = {f"proj__{san}__{k}": v for k, v in projected.items()
                 if k != "prediction_type" and k != "context_tag" and k != "reason"}
        object_to_features[str(oid)] = {
            **feats,
            f"prediction_type__{san}": projected.get("prediction_type"),
            f"reason__{san}": projected.get("reason"),
        }
    print(f"  projected {len(object_to_features):,} objects")

    print("broadcasting to all snapshot rows ...")
    gold["object_id"] = gold["object_id"].astype(str)
    avail_count = 0
    for col in proj_cols:
        gold[col] = pd.NA
    gold[f"avail__{san}"] = 0.0
    gold[f"exact__{san}"] = 0.0  # static_safe — exact=0 by convention
    gold[f"event_count__{san}"] = pd.NA
    gold[f"prediction_type__{san}"] = None
    gold[f"reason__{san}"] = "expert unavailable at this epoch"
    gold[f"source_event_time_jd__{san}"] = pd.NA
    gold[f"temporal_exactness__{san}"] = None

    # Vectorize the join via a per-object map
    feat_df = pd.DataFrame.from_dict(object_to_features, orient="index").reset_index()
    feat_df = feat_df.rename(columns={"index": "object_id"})
    n_hit_objs = len(feat_df)
    print(f"  feat_df cols: {[c for c in feat_df.columns if c != 'object_id'][:6]}...")

    # Update only the rows whose object_id is in the babamul set
    hit_mask = gold["object_id"].isin(set(feat_df["object_id"]))
    print(f"  snapshot rows to be updated (avail=1): {hit_mask.sum():,} of {len(gold):,}")

    # Merge in the projection values for hit rows
    merged = gold.merge(feat_df, on="object_id", how="left", suffixes=("", "_bab"))
    for col in proj_cols + [f"prediction_type__{san}", f"reason__{san}"]:
        bab_col = f"{col}_bab"
        if bab_col in merged.columns:
            merged[col] = merged[bab_col].combine_first(merged[col])
            merged = merged.drop(columns=[bab_col])

    merged.loc[hit_mask, f"avail__{san}"] = 1.0
    merged.loc[hit_mask, f"reason__{san}"] = None
    merged.loc[hit_mask, f"prediction_type__{san}"] = "context_only"
    merged.loc[hit_mask, f"temporal_exactness__{san}"] = "static_safe"

    avail_rate = merged[f"avail__{san}"].mean()
    print(f"  final avail__babamul rate: {avail_rate:.3f} ({n_hit_objs:,} hit objects)")

    # Sanity check: each new column has expected dtype
    for col in proj_cols:
        n_set = merged[col].notna().sum()
        print(f"    {col}: {n_set:,} non-null")

    print(f"writing v7 gold → {V7_GOLD} ...")
    V7_GOLD.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(V7_GOLD, index=False)
    print(f"  wrote {V7_GOLD.stat().st_size / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
