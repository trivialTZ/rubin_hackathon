#!/usr/bin/env python3
"""Inspect TNS crossmatch results — type mapping, source groups, quality tiers.

Run after crossmatch_tns.py or with the TNS bulk CSV directly.

Usage:
    # Check existing crossmatch parquet
    python3 scripts/check_tns_results.py

    # Check a TNS bulk CSV directly (shows what the pipeline would produce)
    python3 scripts/check_tns_results.py --bulk-csv data/tns_public_objects.csv

    # Check against specific labels
    python3 scripts/check_tns_results.py --bulk-csv data/tns_public_objects.csv \
        --labels data/labels.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from debass_meta.access.tns import (
    TNS_TERNARY_MAP,
    TNS_AMBIGUOUS_TYPES,
    TNS_CREDIBLE_GROUPS,
    map_tns_type_to_ternary,
    is_ambiguous_type,
)


def check_parquet(path: Path) -> None:
    """Print summary of an existing TNS crossmatch parquet."""
    df = pd.read_parquet(path)
    n = len(df)

    print(f"File: {path}  ({n} rows)")
    print(f"Columns: {list(df.columns)}")
    print()

    # --- TNS match rate ---
    matched = df["tns_name"].notna().sum() if "tns_name" in df.columns else 0
    typed = df["tns_type"].notna().sum() if "tns_type" in df.columns else 0
    spectra = int(df["tns_has_spectra"].sum()) if "tns_has_spectra" in df.columns else 0
    print(f"TNS matched: {matched}/{n}  typed: {typed}  has_spectra: {spectra}")
    print()

    # --- Types ---
    if "tns_type" in df.columns:
        types = df["tns_type"].dropna().value_counts()
        print("TNS types:")
        for t, c in types.items():
            ternary = map_tns_type_to_ternary(t)
            ambig = " [AMBIGUOUS]" if is_ambiguous_type(t) else ""
            mapped = ternary or "UNMAPPED"
            print(f"  {t:30s} -> {mapped:15s} (n={c}){ambig}")

        # Any types that don't map?
        unmapped = [t for t in types.index if map_tns_type_to_ternary(t) is None]
        if unmapped:
            print(f"\n  WARNING: {len(unmapped)} unmapped types: {unmapped}")
        print()

    # --- Quality ---
    if "label_quality" in df.columns:
        print("Quality tiers:")
        for q, c in df["label_quality"].value_counts().items():
            print(f"  {q:20s}: {c}")
        print()

    # --- Source groups ---
    if "source_group" in df.columns:
        groups = df["source_group"].dropna()
        if len(groups) > 0:
            print("Source groups (who classified):")
            for g, c in groups.value_counts().items():
                in_registry = "YES" if g in TNS_CREDIBLE_GROUPS else "NO"
                print(f"  {g:30s}: {c:4d}  in_registry={in_registry}")
            print()

    if "credibility_tier" in df.columns:
        cred = df["credibility_tier"].dropna()
        if len(cred) > 0:
            print("Credibility tiers:")
            for t, c in cred.value_counts().items():
                print(f"  {t:20s}: {c}")
            print()

    if "downgrade_reason" in df.columns:
        downgrades = df["downgrade_reason"].dropna()
        if len(downgrades) > 0:
            print("Downgraded objects:")
            for reason, c in downgrades.value_counts().items():
                print(f"  {reason}: {c}")
            print()

    # --- Class balance ---
    if "final_class_ternary" in df.columns:
        print("Ternary class balance:")
        for c, n in df["final_class_ternary"].value_counts().items():
            print(f"  {c:15s}: {n}")
        print()

    # --- Per-object detail ---
    print("Per-object detail:")
    cols = ["object_id", "tns_name", "tns_type", "label_quality",
            "final_class_ternary", "source_group", "credibility_tier",
            "downgrade_reason"]
    show_cols = [c for c in cols if c in df.columns]
    for _, r in df[show_cols].iterrows():
        parts = []
        for c in show_cols:
            v = r[c]
            if pd.notna(v):
                parts.append(f"{c}={v}")
        print(f"  {' | '.join(parts)}")


def check_bulk_csv(csv_path: Path, labels_path: Path | None = None) -> None:
    """Analyze a TNS bulk CSV — type distribution, source groups, match rate."""
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False, quotechar='"')
    if "internal_names" not in df.columns:
        df = pd.read_csv(csv_path, low_memory=False, quotechar='"', skiprows=1)

    # Normalize column name
    if "name" in df.columns and "objname" not in df.columns:
        df = df.rename(columns={"name": "objname"})

    n = len(df)
    print(f"Total TNS objects: {n:,}")
    print(f"Columns: {list(df.columns)}")
    print()

    # --- Name prefix distribution ---
    if "name_prefix" in df.columns:
        prefix_counts = df["name_prefix"].value_counts()
        print("Name prefix (SN = spectroscopic, AT = untyped):")
        for p, c in prefix_counts.items():
            pct = 100 * c / n
            print(f"  {str(p):5s}: {c:>7,}  ({pct:.1f}%)")
        print()

    # --- Type distribution ---
    if "type" in df.columns:
        all_types = df["type"].dropna()
        type_counts = all_types.value_counts()
        print(f"Typed objects: {len(all_types):,}/{n:,}")
        print()

        # Map each type to ternary
        ternary_counts: Counter = Counter()
        unmapped_types: Counter = Counter()
        ambiguous_count = 0

        print(f"Top 40 TNS types:")
        for ttype, count in type_counts.head(40).items():
            ternary = map_tns_type_to_ternary(ttype)
            ambig = is_ambiguous_type(ttype)
            if ambig:
                ambiguous_count += count

            if ternary is None:
                unmapped_types[ttype] += count
                tag = "UNMAPPED"
            else:
                ternary_counts[ternary] += count
                tag = ternary

            flag = " [AMBIGUOUS]" if ambig else ""
            print(f"  {ttype:35s} -> {tag:15s} (n={count:>6,}){flag}")

        # Remaining types beyond top 40
        remaining = type_counts.iloc[40:] if len(type_counts) > 40 else pd.Series(dtype=int)
        for ttype, count in remaining.items():
            ternary = map_tns_type_to_ternary(ttype)
            if ternary is None:
                unmapped_types[ttype] += count
            else:
                ternary_counts[ternary] += count
            if is_ambiguous_type(ttype):
                ambiguous_count += count

        print()
        print("Ternary mapping summary:")
        for cls in ["snia", "nonIa_snlike", "other"]:
            print(f"  {cls:15s}: {ternary_counts[cls]:>7,}")
        print(f"  {'unmapped':15s}: {sum(unmapped_types.values()):>7,}")
        print(f"  {'ambiguous':15s}: {ambiguous_count:>7,}")

        if unmapped_types:
            print(f"\nUnmapped types ({len(unmapped_types)}):")
            for t, c in unmapped_types.most_common(20):
                print(f"  {t:35s}: {c}")
        print()

    # --- Source group distribution ---
    if "source_group" in df.columns:
        groups = df["source_group"].dropna()
        group_counts = groups.value_counts()
        print(f"Source groups (who classified): {len(group_counts)} distinct")
        in_reg = 0
        not_in_reg = 0
        for g, c in group_counts.items():
            if g in TNS_CREDIBLE_GROUPS:
                in_reg += c
            else:
                not_in_reg += c
        total_grp = in_reg + not_in_reg
        print(f"  In registry: {in_reg:>7,} ({100*in_reg/max(total_grp,1):.1f}%)")
        print(f"  Not in registry: {not_in_reg:>7,} ({100*not_in_reg/max(total_grp,1):.1f}%)")
        print()
        print("Top 30 source groups:")
        for g, c in group_counts.head(30).items():
            in_r = "YES" if g in TNS_CREDIBLE_GROUPS else " NO"
            print(f"  {g:30s}: {c:>6,}  in_registry={in_r}")

        # Show groups NOT in registry
        not_in = [(g, c) for g, c in group_counts.items() if g not in TNS_CREDIBLE_GROUPS]
        if not_in:
            print(f"\nGroups NOT in registry ({len(not_in)}):")
            for g, c in sorted(not_in, key=lambda x: -x[1])[:30]:
                print(f"  {g:30s}: {c:>6,}")
        print()

    # --- Reporting group distribution ---
    if "reporting_group" in df.columns:
        rgroups = df["reporting_group"].dropna().value_counts()
        print(f"Top 20 reporting groups (who discovered):")
        for g, c in rgroups.head(20).items():
            print(f"  {g:30s}: {c:>6,}")
        print()

    # --- Match rate against labels ---
    if labels_path and labels_path.exists():
        labels_df = pd.read_csv(labels_path)
        label_ids = set(labels_df["object_id"].astype(str))
        print(f"Labels: {len(label_ids)} objects")

        # Build internal_names lookup
        df["internal_names"] = df["internal_names"].fillna("")
        exploded = df.assign(
            iname=df["internal_names"].str.split(";")
        ).explode("iname")
        exploded["iname"] = exploded["iname"].str.strip()
        known_ids = set(exploded["iname"])

        matched = label_ids & known_ids
        print(f"  Matched in TNS: {len(matched)}/{len(label_ids)}")
        if matched:
            matched_rows = exploded[exploded["iname"].isin(matched)]
            types = matched_rows["type"].dropna().value_counts()
            print(f"  Types of matched objects:")
            for t, c in types.items():
                ternary = map_tns_type_to_ternary(t) or "UNMAPPED"
                print(f"    {t:25s} -> {ternary:15s} (n={c})")
        print()


def check_cache(cache_dir: Path) -> None:
    """Summarize cached TNS API results."""
    if not cache_dir.exists():
        print(f"No cache at {cache_dir}")
        return

    files = list(cache_dir.glob("*.json"))
    print(f"Cache: {len(files)} files in {cache_dir}")

    matched = 0
    types: Counter = Counter()
    groups: Counter = Counter()
    instruments: Counter = Counter()

    for fp in sorted(files):
        d = json.load(open(fp))
        if not d.get("tns_name"):
            continue
        matched += 1
        ttype = d.get("type_name", "")
        if ttype:
            types[ttype] += 1

        raw = d.get("raw", {})
        spectra = raw.get("spectra", [])
        if isinstance(spectra, list):
            for s in spectra:
                sg = s.get("source_group_name", "")
                if sg:
                    groups[sg] += 1
                inst = s.get("instrument", {})
                if isinstance(inst, dict) and inst.get("name"):
                    instruments[inst["name"]] += 1

    print(f"  Matched: {matched}/{len(files)}")
    print()

    if types:
        print("  Types:")
        for t, c in types.most_common():
            ternary = map_tns_type_to_ternary(t) or "UNMAPPED"
            print(f"    {t:25s} -> {ternary:15s} (n={c})")
        print()

    if groups:
        print("  Classification source groups:")
        for g, c in groups.most_common():
            in_r = "YES" if g in TNS_CREDIBLE_GROUPS else " NO"
            print(f"    {g:25s}: {c}  in_registry={in_r}")
        print()

    if instruments:
        print("  Instruments:")
        for i, c in instruments.most_common():
            print(f"    {i:25s}: {c}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect TNS crossmatch results, bulk CSV, or cache.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--parquet", type=Path, help="Check a crossmatch parquet file")
    parser.add_argument("--bulk-csv", type=Path, help="Analyze a TNS bulk CSV directly")
    parser.add_argument("--labels", type=Path, help="Labels CSV to check match rate")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/tns_cache"),
                        help="TNS cache directory (default: data/tns_cache)")
    parser.add_argument("--all", action="store_true",
                        help="Check all available sources (parquet + cache + csv)")
    args = parser.parse_args()

    ran_something = False

    # Check parquet
    parquet_paths = [
        args.parquet,
        Path("data/truth/tns_crossmatch.parquet"),
        Path("data/truth/object_truth.parquet"),
    ]
    for p in parquet_paths:
        if p and p.exists() and (args.all or p == args.parquet or (not args.parquet and not args.bulk_csv)):
            print("=" * 70)
            check_parquet(p)
            print()
            ran_something = True

    # Check bulk CSV
    if args.bulk_csv and args.bulk_csv.exists():
        print("=" * 70)
        check_bulk_csv(args.bulk_csv, args.labels)
        ran_something = True

    # Check cache
    if args.cache_dir.exists() and (args.all or (not args.parquet and not args.bulk_csv)):
        print("=" * 70)
        check_cache(args.cache_dir)
        ran_something = True

    if not ran_something:
        print("No TNS data found. Run one of:")
        print("  python3 scripts/download_tns_bulk.py")
        print("  python3 scripts/crossmatch_tns.py --labels data/labels.csv")


if __name__ == "__main__":
    main()
