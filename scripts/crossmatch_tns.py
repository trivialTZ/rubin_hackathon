"""Crossmatch objects against TNS and build a three-tier truth table.

Usage:
  # API mode (slow, 2 calls/object)
  python3.11 scripts/crossmatch_tns.py --labels data/labels.csv

  # Bulk CSV mode (fast, local crossmatch)
  python3.11 scripts/crossmatch_tns.py --labels data/labels.csv \
      --bulk-csv data/tns_public_objects.csv

  # With broker consensus (needs silver events)
  python3.11 scripts/crossmatch_tns.py --labels data/labels.csv \
      --bulk-csv data/tns_public_objects.csv --silver-dir data/silver --consensus

Bulk CSV mode:
  Download TNS bulk CSV on SCC: python3.11 scripts/download_tns_bulk.py
  Then crossmatch locally (no API calls needed for basic crossmatch)

API mode requires env vars: TNS_API_KEY, TNS_TNS_ID, TNS_MARKER_NAME
  (loaded from .env in repo root if python-dotenv is installed)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Load .env from repo root if present (silent if python-dotenv not installed)
_repo_root = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(_repo_root / ".env")
except ImportError:
    pass

import pandas as pd

from debass_meta.access.tns import (
    TNSClient,
    TNSResult,
    load_tns_credentials,
    map_tns_type_to_ternary,
)
from debass_meta.truth.quality import (
    QualityAssignment,
    assign_quality_from_tns,
    collect_broker_votes,
)


# ------------------------------------------------------------------ #
# Cache layer — resume support for large runs                         #
# ------------------------------------------------------------------ #


def _cache_path(cache_dir: Path, object_id: str) -> Path:
    return cache_dir / f"{object_id}.json"


def _load_cached(cache_dir: Path, object_id: str) -> TNSResult | None:
    p = _cache_path(cache_dir, object_id)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return TNSResult(
            object_id=data["object_id"],
            tns_name=data.get("tns_name"),
            tns_prefix=data.get("tns_prefix"),
            type_name=data.get("type_name"),
            has_spectra=bool(data.get("has_spectra", False)),
            redshift=data.get("redshift"),
            ra=data.get("ra"),
            dec=data.get("dec"),
            discovery_date=data.get("discovery_date"),
            raw=data.get("raw", {}),
        )
    except Exception:
        return None


def _save_cached(cache_dir: Path, result: TNSResult) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = _cache_path(cache_dir, result.object_id)
    data = {
        "object_id": result.object_id,
        "tns_name": result.tns_name,
        "tns_prefix": result.tns_prefix,
        "type_name": result.type_name,
        "has_spectra": result.has_spectra,
        "redshift": result.redshift,
        "ra": result.ra,
        "dec": result.dec,
        "discovery_date": result.discovery_date,
        "raw": result.raw,
    }
    p.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


# ------------------------------------------------------------------ #
# Main crossmatch pipeline                                            #
# ------------------------------------------------------------------ #


def load_object_ids(labels_path: Path) -> list[str]:
    """Load object IDs from labels CSV."""
    ids: list[str] = []
    with open(labels_path) as fh:
        for row in csv.DictReader(fh):
            oid = str(row.get("object_id") or "").strip()
            if oid:
                ids.append(oid)
    return ids


def crossmatch_bulk_csv(
    *,
    object_ids: list[str],
    bulk_csv_path: Path,
) -> list[TNSResult]:
    """Crossmatch against TNS bulk CSV (fast, no API calls).

    The TNS bulk CSV has columns: objname, name_prefix, internal_names, type, ...
    We match on internal_names (contains ZTF IDs as semicolon-separated list).

    Proxy for has_spectra: name_prefix="SN" + non-empty type
    (verified 100% accurate on our 14 matched objects)
    """
    print(f"Loading TNS bulk CSV from {bulk_csv_path}")
    tns_df = pd.read_csv(bulk_csv_path, low_memory=False)
    print(f"Loaded {len(tns_df)} TNS objects")

    # Explode internal_names (semicolon-separated) into one row per ZTF ID
    tns_df["internal_names"] = tns_df["internal_names"].fillna("")
    tns_exploded = tns_df.assign(
        internal_name=tns_df["internal_names"].str.split(";")
    ).explode("internal_name")
    tns_exploded["internal_name"] = tns_exploded["internal_name"].str.strip()

    # Build lookup dict: ZTF ID -> TNS row
    lookup = {}
    for _, row in tns_exploded.iterrows():
        ztf_id = row["internal_name"]
        if ztf_id and ztf_id.startswith("ZTF"):
            lookup[ztf_id] = row

    print(f"Built lookup for {len(lookup)} ZTF IDs")

    # Crossmatch
    results = []
    for oid in object_ids:
        if oid not in lookup:
            results.append(TNSResult(
                object_id=oid,
                tns_name=None,
                tns_prefix=None,
                type_name=None,
                has_spectra=False,
                redshift=None,
                ra=None,
                dec=None,
                discovery_date=None,
                raw={"source": "bulk_csv", "status": "no_match"},
            ))
            continue

        row = lookup[oid]
        tns_name = str(row.get("objname") or "").strip() or None
        tns_prefix = str(row.get("name_prefix") or "").strip() or None
        type_name = str(row.get("type") or "").strip() or None

        # Proxy: prefix=SN + non-empty type → has_spectra=True
        has_spectra = bool(tns_prefix == "SN" and type_name)

        redshift = row.get("redshift")
        if pd.notna(redshift):
            redshift = float(redshift)
        else:
            redshift = None

        ra = row.get("ra")
        if pd.notna(ra):
            ra = float(ra)
        else:
            ra = None

        dec = row.get("declination")
        if pd.notna(dec):
            dec = float(dec)
        else:
            dec = None

        discovery_date = str(row.get("discoverydate") or "").strip() or None

        results.append(TNSResult(
            object_id=oid,
            tns_name=tns_name,
            tns_prefix=tns_prefix,
            type_name=type_name,
            has_spectra=has_spectra,
            redshift=redshift,
            ra=ra,
            dec=dec,
            discovery_date=discovery_date,
            raw={"source": "bulk_csv"},
        ))

    return results


def crossmatch_tns(
    *,
    object_ids: list[str],
    cache_dir: Path | None = None,
    offline: bool = False,
    sandbox: bool = False,
    max_workers: int = 4,
    progress: bool = True,
) -> list[TNSResult]:
    """Crossmatch a list of object IDs against TNS.

    Supports:
    - Caching for resume on large runs
    - Offline mode (cache-only, no API calls)
    - Concurrent fetching with rate-limit backoff
    """
    results: list[TNSResult] = []
    to_fetch: list[str] = []

    # Check cache first
    for oid in object_ids:
        if cache_dir is not None:
            cached = _load_cached(cache_dir, oid)
            if cached is not None:
                results.append(cached)
                continue
        to_fetch.append(oid)

    cached_count = len(results)
    if cached_count > 0:
        print(f"Loaded {cached_count} objects from cache, {len(to_fetch)} to fetch")

    if offline:
        # Fill missing with empty results
        for oid in to_fetch:
            results.append(TNSResult(
                object_id=oid,
                tns_name=None,
                tns_prefix=None,
                type_name=None,
                has_spectra=False,
                redshift=None,
                ra=None,
                dec=None,
                discovery_date=None,
                raw={"status": "offline_no_cache"},
            ))
        return results

    if not to_fetch:
        return results

    # Fetch from TNS
    creds = load_tns_credentials(sandbox=sandbox)
    client = TNSClient(creds, timeout_s=60.0, max_retries=3)

    fetched = client.crossmatch_bulk(
        to_fetch,
        max_workers=max_workers,
        progress=progress,
    )

    # Cache successful results only (don't cache failures so retries work)
    for r in fetched:
        if cache_dir is not None and r.raw.get("error") != "crossmatch_failed":
            _save_cached(cache_dir, r)
        results.append(r)

    return results


def build_truth_from_crossmatch(
    *,
    tns_results: list[TNSResult],
    events_df: Any = None,
    use_consensus: bool = True,
    existing_labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build the truth table from TNS crossmatch results + optional broker consensus.

    Parameters
    ----------
    tns_results : list of TNSResult
    events_df : DataFrame of silver broker events, optional
        If provided and use_consensus=True, late-time broker votes
        are collected for consensus tier assignment.
    use_consensus : bool
        Whether to compute broker consensus votes.
    existing_labels : dict mapping object_id -> ternary label, optional
        Fallback labels (e.g. from labels.csv) for objects without TNS match.
    """
    existing_labels = existing_labels or {}
    rows: list[dict[str, Any]] = []

    for result in tns_results:
        tns_ternary = map_tns_type_to_ternary(result.type_name)

        # Collect broker consensus votes if silver events available
        broker_votes: dict[str, str] = {}
        if use_consensus and events_df is not None:
            broker_votes = collect_broker_votes(events_df, result.object_id)

        qa = assign_quality_from_tns(
            tns_type=result.type_name,
            tns_ternary=tns_ternary,
            has_spectra=result.has_spectra,
            broker_votes=broker_votes,
        )

        final_class = qa.final_class_ternary
        label_source = qa.label_source
        quality = qa.quality

        # Fallback to existing labels if no TNS match and no consensus
        if final_class is None and result.object_id in existing_labels:
            final_class = existing_labels[result.object_id]
            label_source = "alerce_self_label"
            quality = "weak"

        if final_class is None:
            continue

        rows.append({
            "object_id": result.object_id,
            "final_class_raw": result.type_name or final_class,
            "final_class_ternary": final_class,
            "follow_proxy": int(final_class == "snia"),
            "label_source": label_source,
            "label_quality": quality,
            "tns_name": result.tns_name,
            "tns_prefix": result.tns_prefix,
            "tns_type": result.type_name,
            "tns_has_spectra": result.has_spectra,
            "tns_redshift": result.redshift,
            "tns_ra": result.ra,
            "tns_dec": result.dec,
            "tns_discovery_date": result.discovery_date,
            "consensus_experts": (
                ",".join(qa.consensus_experts) if qa.consensus_experts else None
            ),
            "consensus_n_agree": qa.consensus_n_agree,
            "consensus_n_total": qa.consensus_n_total,
        })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Crossmatch objects against TNS")
    parser.add_argument("--labels", required=True, help="Labels CSV with object_id column")
    parser.add_argument("--output", default="data/truth/tns_crossmatch.parquet")
    parser.add_argument("--bulk-csv", help="TNS bulk CSV for fast local crossmatch (no API calls)")
    parser.add_argument("--cache-dir", default="data/tns_cache")
    parser.add_argument("--silver-dir", default=None, help="Silver dir for broker consensus")
    parser.add_argument("--consensus", action="store_true", help="Enable broker consensus tier")
    parser.add_argument("--offline", action="store_true", help="Cache-only, no TNS API calls")
    parser.add_argument("--resume", action="store_true", help="Skip objects already in cache")
    parser.add_argument("--sandbox", action="store_true", help="Use TNS sandbox")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    if not labels_path.exists():
        raise SystemExit(f"Labels file not found: {labels_path}")

    # Load object IDs
    object_ids = load_object_ids(labels_path)
    if not object_ids:
        raise SystemExit("No object IDs found in labels file")
    print(f"Loaded {len(object_ids)} object IDs from {labels_path}")

    # Load existing labels as fallback
    existing_labels: dict[str, str] = {}
    with open(labels_path) as fh:
        for row in csv.DictReader(fh):
            oid = str(row.get("object_id") or "").strip()
            label = str(row.get("label") or "").strip()
            if oid and label:
                existing_labels[oid] = label

    # TNS crossmatch — bulk CSV or API mode
    if args.bulk_csv:
        bulk_csv_path = Path(args.bulk_csv)
        if not bulk_csv_path.exists():
            raise SystemExit(f"Bulk CSV not found: {bulk_csv_path}")
        tns_results = crossmatch_bulk_csv(
            object_ids=object_ids,
            bulk_csv_path=bulk_csv_path,
        )
    else:
        # API mode
        cache_dir = Path(args.cache_dir) if (args.cache_dir or args.resume) else None
        tns_results = crossmatch_tns(
            object_ids=object_ids,
            cache_dir=cache_dir,
            offline=args.offline,
            sandbox=args.sandbox,
            max_workers=args.max_workers,
            progress=not args.no_progress,
        )

    # Load silver events for consensus
    events_df = None
    if args.consensus and args.silver_dir:
        silver_path = Path(args.silver_dir) / "broker_events.parquet"
        if silver_path.exists():
            events_df = pd.read_parquet(silver_path)
            print(f"Loaded {len(events_df)} silver event rows for consensus")
        else:
            print(f"WARNING: silver events not found at {silver_path}, skipping consensus")

    # Build truth table
    truth_df = build_truth_from_crossmatch(
        tns_results=tns_results,
        events_df=events_df,
        use_consensus=args.consensus,
        existing_labels=existing_labels,
    )

    # Summary
    quality_counts = truth_df["label_quality"].value_counts().to_dict()
    source_counts = truth_df["label_source"].value_counts().to_dict()
    print(f"\nTruth table: {len(truth_df)} objects")
    print(f"  Quality:  {quality_counts}")
    print(f"  Sources:  {source_counts}")

    tns_matched = truth_df["tns_name"].notna().sum()
    tns_typed = truth_df["tns_type"].notna().sum()
    tns_spectra = truth_df["tns_has_spectra"].sum()
    print(f"  TNS matched: {tns_matched}, typed: {tns_typed}, with spectra: {tns_spectra}")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    truth_df.to_parquet(output_path, index=False)
    print(f"\nWrote TNS crossmatch truth table -> {output_path}")


if __name__ == "__main__":
    main()
