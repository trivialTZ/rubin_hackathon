#!/usr/bin/env python3
"""Fast incremental normalize: process ONLY new ANTARES bronze files and
append to existing silver/broker_events.parquet without touching the big
fink/alerce files.

Use when bronze/fink_*.parquet and alerce_*.parquet are already represented
in silver and you only need to incorporate newly fetched antares/ampel/... data.

Usage:
    python3 scripts/normalize_antares_only.py --broker antares
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from debass_meta.ingest.silver import bronze_to_silver


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--broker", default="antares",
                    help="Broker prefix to include (e.g. antares, ampel)")
    ap.add_argument("--bronze-dir", default="data/bronze")
    ap.add_argument("--silver-dir", default="data/silver")
    ap.add_argument("--work-dir", default="data/bronze_incremental")
    args = ap.parse_args()

    # Stage only the selected broker's parquets in a temp bronze_dir
    bronze_dir = Path(args.bronze_dir)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    # Clean work dir
    for f in work_dir.glob("*.parquet"):
        f.unlink()

    # Symlink the filtered bronzes in work_dir
    selected = list(bronze_dir.glob(f"{args.broker}_*.parquet"))
    if not selected:
        print(f"No {args.broker}_*.parquet files in {bronze_dir}")
        sys.exit(1)
    for f in selected:
        target = work_dir / f.name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(f.resolve())
    print(f"Staged {len(selected)} {args.broker} bronze files for normalize")

    # Run silver normalise against the staging dir
    silver_path = Path(args.silver_dir) / "broker_events.parquet"
    temp_silver_dir = work_dir / "silver"
    temp_silver_dir.mkdir(parents=True, exist_ok=True)
    new_path = bronze_to_silver(
        bronze_dir=work_dir,
        silver_dir=temp_silver_dir,
    )
    new_df = pd.read_parquet(new_path)
    print(f"New {args.broker} silver events: {len(new_df):,}")

    # Append to main silver (drop duplicate rows on payload_hash+event_time_jd+expert_key)
    if silver_path.exists():
        old_df = pd.read_parquet(silver_path)
        print(f"Existing silver rows: {len(old_df):,}")
        # Reconcile schema: coerce alert_id to string across both frames
        for df_ in (old_df, new_df):
            if "alert_id" in df_.columns:
                df_["alert_id"] = df_["alert_id"].astype("string")
        merged = pd.concat([old_df, new_df], ignore_index=True)
        key_cols = [c for c in ("payload_hash", "event_time_jd", "expert_key", "field") if c in merged.columns]
        merged = merged.drop_duplicates(subset=key_cols)
        merged.to_parquet(silver_path, index=False)
        print(f"Merged silver rows: {len(merged):,} → {silver_path}")
    else:
        new_df.to_parquet(silver_path, index=False)
        print(f"Wrote {len(new_df):,} rows → {silver_path}")


if __name__ == "__main__":
    main()
