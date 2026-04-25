#!/usr/bin/env python3
"""Repair silver/local_expert_outputs after the v6e.2 race condition.

What happened
-------------
Concurrent local_infer.py runs (4 array tasks at once) merged against a
stale local_expert_outputs.json that did NOT contain the original
ZTF rows for alerce_lc and lc_features_bv. Each task's _merge_records
call therefore wrote back partition parquets containing only the
records present in the JSON cache + that run's new records.

After three of the four tasks finished:
  - alerce_lc/part-latest.parquet  : ZTF gone (74,916 LSST only)
  - lc_features_bv/part-latest.parquet : ZTF gone (74,916 LSST only)
  - supernnova/part-latest.parquet : intact (ZTF was in JSON cache)
  - salt3_chi2/part-latest.parquet : untouched (task 1 killed in time)

We salvage:
  - alerce_lc:        merge part-allobj.parquet (ZTF, Apr 21 backup) +
                      current part-latest.parquet (LSST) → write back
  - lc_features_bv:   no ZTF backup exists; the partition stays at LSST
                      only and we flag re-run is needed
  - supernnova:       no action (already correct)
  - salt3_chi2:       no action (LSST inference still pending)
  - local_expert_outputs.json: rebuild from all 4 partition parquets so
                      the next local_infer.py invocation sees a complete
                      existing-records snapshot.

Subsequent steps after this repair:
  1. Run local_infer.py --expert lc_features_bv --from-labels data/labels.csv
     (full 12,772 objects, restores ZTF + populates LSST).
  2. Run local_infer.py --expert salt3_chi2 --from-labels data/labels_lsst.csv
     (3,998 LSST objects).
  Both must run SEQUENTIALLY (not as a parallel array) to avoid
  re-triggering the race.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_SILVER = Path("data/silver")


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--silver-dir", type=Path, default=DEFAULT_SILVER)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    base = args.silver_dir / "local_expert_outputs"

    print("=== silver state before repair ===")
    state: dict[str, dict[str, pd.DataFrame]] = {}
    for ek in ("alerce_lc", "lc_features_bv", "salt3_chi2", "supernnova"):
        state[ek] = {}
        for fname in ("part-latest.parquet", "part-allobj.parquet",
                      "part-lsst-latest.parquet"):
            p = base / ek / fname
            if not p.exists():
                continue
            df = _read(p)
            state[ek][fname] = df
            oids = df["object_id"].astype(str) if "object_id" in df.columns else pd.Series([], dtype=str)
            ztf = int(oids.str.startswith("ZTF").sum())
            lsst = int(oids.str.match(r"^[0-9]{10,}$").sum())
            print(f"  {ek}/{fname}: {len(df):,} ({ztf:,} ZTF, {lsst:,} LSST)")

    # --- alerce_lc: merge allobj (ZTF backup) + current latest (LSST) ---
    al_alob = state["alerce_lc"].get("part-allobj.parquet", pd.DataFrame())
    al_late = state["alerce_lc"].get("part-latest.parquet", pd.DataFrame())
    if len(al_alob) > 0 and len(al_late) > 0:
        merged = pd.concat([al_alob, al_late], ignore_index=True)
        merged = merged.drop_duplicates(
            subset=["expert", "object_id", "n_det"], keep="last"
        )
        merged_oids = merged["object_id"].astype(str)
        z = int(merged_oids.str.startswith("ZTF").sum())
        l = int(merged_oids.str.match(r"^[0-9]{10,}$").sum())
        print(f"\nalerce_lc merge: {len(merged):,} ({z:,} ZTF, {l:,} LSST)")
        if not args.dry_run:
            (base / "alerce_lc" / "part-latest.parquet").parent.mkdir(parents=True, exist_ok=True)
            merged.to_parquet(base / "alerce_lc" / "part-latest.parquet", index=False)
            print(f"  → wrote {base/'alerce_lc'/'part-latest.parquet'}")
        state["alerce_lc"]["part-latest.parquet"] = merged

    # --- supernnova: nothing to do, but include lsst-latest if present ---
    sn_late = state["supernnova"].get("part-latest.parquet", pd.DataFrame())
    sn_lsst = state["supernnova"].get("part-lsst-latest.parquet", pd.DataFrame())
    # part-lsst-latest is just 50 rows from a preflight; the full LSST data is
    # already in part-latest from the v6e.2 task 4 write. Leave alone.

    # --- Rebuild local_expert_outputs.json from all 4 partition latest files ---
    print("\n=== rebuild local_expert_outputs.json ===")
    all_records: list[dict] = []
    for ek in ("alerce_lc", "lc_features_bv", "salt3_chi2", "supernnova"):
        df = state[ek].get("part-latest.parquet", pd.DataFrame())
        if len(df) == 0:
            continue
        # local_expert_outputs.json stores class_probabilities as dict, not JSON string
        df = df.copy()
        if "class_probabilities" in df.columns:
            df["class_probabilities"] = df["class_probabilities"].apply(
                lambda v: json.loads(v) if isinstance(v, str) else v
            )
        recs = df.to_dict(orient="records")
        all_records.extend(recs)
        print(f"  {ek}: {len(recs):,} records")
    print(f"  total: {len(all_records):,}")

    json_path = args.silver_dir / "local_expert_outputs.json"
    if not args.dry_run:
        with open(json_path, "w") as fh:
            json.dump(all_records, fh)
        print(f"  → wrote {json_path}")

    print("\n=== final per-expert silver counts ===")
    for ek in ("alerce_lc", "lc_features_bv", "salt3_chi2", "supernnova"):
        df = state[ek].get("part-latest.parquet", pd.DataFrame())
        if len(df) == 0:
            print(f"  {ek}: EMPTY")
            continue
        oids = df["object_id"].astype(str)
        z = int(oids.str.startswith("ZTF").sum())
        l = int(oids.str.match(r"^[0-9]{10,}$").sum())
        print(f"  {ek}: {len(df):,} ({z:,} ZTF, {l:,} LSST)")


if __name__ == "__main__":
    main()
