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
    is_ambiguous_type,
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
            source_group=data.get("source_group"),
            classification_instrument=data.get("classification_instrument"),
            credibility_tier=data.get("credibility_tier"),
            credibility_reason=data.get("credibility_reason"),
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
        "source_group": result.source_group,
        "classification_instrument": result.classification_instrument,
        "credibility_tier": result.credibility_tier,
        "credibility_reason": result.credibility_reason,
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


def _load_tns_bulk_csv(bulk_csv_path: Path) -> pd.DataFrame:
    """Load the TNS bulk CSV, handling the optional timestamp header row.

    The real TNS bulk export has ~22 columns:
      objid, name_prefix, name, ra, declination, redshift, typeid, type,
      reporting_groupid, reporting_group, source_groupid, source_group,
      discoverydate, discoverymag, discmagfilter, filter, reporters,
      time_received, internal_names, Discovery_ADS_bibcode,
      Class_ADS_bibcodes, creationdate, lastmodified

    Key columns for credibility: source_group (who classified),
    reporting_group (who discovered). Older/trimmed CSVs may lack these.
    """
    tns_df = pd.read_csv(bulk_csv_path, low_memory=False, quotechar='"')
    if "internal_names" not in tns_df.columns:
        tns_df = pd.read_csv(bulk_csv_path, low_memory=False, quotechar='"', skiprows=1)

    # The official TNS bulk CSV uses "name" not "objname" — normalize
    if "name" in tns_df.columns and "objname" not in tns_df.columns:
        tns_df = tns_df.rename(columns={"name": "objname"})

    available = set(tns_df.columns)
    expected_key = {"objname", "name_prefix", "internal_names", "type"}
    found = expected_key & available
    if len(found) < 3:
        raise ValueError(
            f"TNS CSV missing expected columns. Found: {sorted(available)}. "
            f"Expected at least: {sorted(expected_key)}"
        )

    has_source = "source_group" in available
    has_reporting = "reporting_group" in available
    print(f"  TNS CSV columns: {len(available)} "
          f"(source_group={'yes' if has_source else 'NO'}, "
          f"reporting_group={'yes' if has_reporting else 'NO'})")
    if not has_source:
        print("  WARNING: source_group column missing — "
              "credibility filtering will rely on API data or be skipped")

    return tns_df


def _safe_str(val: Any) -> str | None:
    """Convert a pandas value to string, handling NA/NaN/None."""
    if val is None:
        return None
    if pd.isna(val):
        return None
    s = str(val).strip()
    return s if s else None


def _tns_row_to_result(
    object_id: str, row: Any, *, source: str = "bulk_csv"
) -> TNSResult:
    """Convert a TNS bulk CSV row to a TNSResult.

    The full TNS bulk export has columns including:
      objid, name_prefix, name, ra, declination, redshift, typeid, type,
      reporting_groupid, reporting_group, source_groupid, source_group,
      discoverydate, discoverymag, discmagfilter, filter, reporters,
      time_received, internal_names, Discovery_ADS_bibcode,
      Class_ADS_bibcodes, creationdate, lastmodified

    We extract source_group for credibility verification when available.
    Older or trimmed CSV exports may lack these columns.
    """
    tns_name = _safe_str(row.get("objname") or row.get("name"))
    tns_prefix = _safe_str(row.get("name_prefix"))
    type_name = _safe_str(row.get("type"))

    # has_spectra proxy: name_prefix="SN" + non-empty type means TNS
    # moderators accepted a spectroscopic classification report.
    # Prefix "AT" = astronomical transient (no confirmed spectrum).
    has_spectra = bool(tns_prefix == "SN" and type_name)

    redshift = row.get("redshift")
    redshift = float(redshift) if pd.notna(redshift) else None

    ra = row.get("ra")
    ra = float(ra) if pd.notna(ra) else None

    dec = row.get("declination")
    dec = float(dec) if pd.notna(dec) else None

    discovery_date = _safe_str(row.get("discoverydate"))

    # Extract source/reporting group when available in the CSV.
    # The full TNS bulk export has "source_group" (who classified)
    # and "reporting_group" (who discovered).
    source_group = _safe_str(row.get("source_group"))
    reporting_group = _safe_str(row.get("reporting_group"))

    # Determine credibility from the CSV source_group column.
    # If the column exists and has a value, check against our registry.
    credibility_tier = None
    credibility_reason = None
    if source_group is not None:
        from debass_meta.access.tns import TNS_CREDIBLE_GROUPS
        if source_group in TNS_CREDIBLE_GROUPS:
            credibility_tier = "professional"
            credibility_reason = f"credible_group:{source_group}"
        else:
            credibility_tier = "unverified"
            credibility_reason = f"group_not_in_registry:{source_group}"

    return TNSResult(
        object_id=object_id,
        tns_name=tns_name,
        tns_prefix=tns_prefix,
        type_name=type_name,
        has_spectra=has_spectra,
        redshift=redshift,
        ra=ra,
        dec=dec,
        discovery_date=discovery_date,
        raw={"source": source, "reporting_group": reporting_group},
        source_group=source_group,
        credibility_tier=credibility_tier,
        credibility_reason=credibility_reason,
    )


def _no_match_result(object_id: str, *, source: str = "bulk_csv") -> TNSResult:
    return TNSResult(
        object_id=object_id,
        tns_name=None,
        tns_prefix=None,
        type_name=None,
        has_spectra=False,
        redshift=None,
        ra=None,
        dec=None,
        discovery_date=None,
        raw={"source": source, "status": "no_match"},
    )


def crossmatch_bulk_csv_positional(
    *,
    object_ids: list[str],
    object_coords: dict[str, tuple[float, float]],
    tns_df: pd.DataFrame,
    match_radius_arcsec: float = 3.0,
) -> list[TNSResult]:
    """Positional crossmatch against TNS bulk CSV using astropy SkyCoord.

    For each object with coordinates, finds the nearest TNS object within
    ``match_radius_arcsec``.  This handles Rubin-only transients that have
    no ZTF counterpart and thus no internal_name match.

    Parameters
    ----------
    object_ids : list[str]
        Object IDs to crossmatch (only those present in object_coords are matched).
    object_coords : dict
        Mapping of object_id → (ra_deg, dec_deg).
    tns_df : DataFrame
        Pre-loaded TNS bulk CSV (must have ``ra`` and ``declination`` columns).
    match_radius_arcsec : float
        Maximum separation for a valid match (default 3 arcsec).
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    # Filter TNS rows with valid coordinates
    tns_pos = tns_df.dropna(subset=["ra", "declination"]).copy()
    if len(tns_pos) == 0:
        return [_no_match_result(oid, source="bulk_csv_positional") for oid in object_ids]

    tns_catalog = SkyCoord(
        ra=tns_pos["ra"].values, dec=tns_pos["declination"].values, unit="deg"
    )

    results = []
    for oid in object_ids:
        if oid not in object_coords:
            results.append(_no_match_result(oid, source="bulk_csv_positional"))
            continue

        ra, dec = object_coords[oid]
        obj_coord = SkyCoord(ra=ra, dec=dec, unit="deg")
        sep = obj_coord.separation(tns_catalog)
        min_idx = sep.argmin()
        min_sep_arcsec = sep[min_idx].to(u.arcsec).value

        if min_sep_arcsec > match_radius_arcsec:
            results.append(_no_match_result(oid, source="bulk_csv_positional"))
            continue

        tns_row = tns_pos.iloc[min_idx]
        result = _tns_row_to_result(oid, tns_row, source="bulk_csv_positional")
        # Inject separation metadata into raw dict
        result.raw["match_sep_arcsec"] = round(min_sep_arcsec, 4)
        result.raw["match_method"] = "positional"
        results.append(result)

    return results


def crossmatch_bulk_csv(
    *,
    object_ids: list[str],
    bulk_csv_path: Path,
    object_coords: dict[str, tuple[float, float]] | None = None,
    positional_radius_arcsec: float = 3.0,
) -> list[TNSResult]:
    """Crossmatch against TNS bulk CSV (fast, no API calls).

    The TNS bulk CSV has columns: objname, name_prefix, internal_names, type, ...
    We first match on internal_names (contains ZTF IDs as semicolon-separated list).

    If ``object_coords`` is provided, objects that fail name-matching are
    retried with a positional cone search against the TNS catalog.
    This handles Rubin-only transients with no ZTF counterpart.

    Proxy for has_spectra: name_prefix="SN" + non-empty type
    (verified 100% accurate on our 14 matched objects)
    """
    print(f"Loading TNS bulk CSV from {bulk_csv_path}")
    tns_df = _load_tns_bulk_csv(bulk_csv_path)
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

    # Phase 1: Name-based crossmatch
    results = {}
    unmatched = []
    for oid in object_ids:
        if oid in lookup:
            results[oid] = _tns_row_to_result(oid, lookup[oid])
        else:
            unmatched.append(oid)

    name_matched = len(object_ids) - len(unmatched)
    print(f"Name-matched: {name_matched}/{len(object_ids)}")

    # Phase 2: Positional fallback for unmatched objects with coordinates
    if unmatched and object_coords:
        coords_available = {oid: object_coords[oid] for oid in unmatched if oid in object_coords}
        if coords_available:
            print(f"Positional fallback: {len(coords_available)} objects with coordinates")
            positional_results = crossmatch_bulk_csv_positional(
                object_ids=list(coords_available.keys()),
                object_coords=coords_available,
                tns_df=tns_df,
                match_radius_arcsec=positional_radius_arcsec,
            )
            pos_matched = sum(1 for r in positional_results if r.tns_name is not None)
            print(f"Positional matches: {pos_matched}/{len(coords_available)}")
            for r in positional_results:
                results[r.object_id] = r

    # Fill remaining unmatched
    for oid in object_ids:
        if oid not in results:
            results[oid] = _no_match_result(oid)

    return [results[oid] for oid in object_ids]


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
            credibility_tier=result.credibility_tier,
            source_group=result.source_group,
            type_is_ambiguous=is_ambiguous_type(result.type_name),
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
            # Credibility metadata
            "source_group": result.source_group,
            "classification_instrument": result.classification_instrument,
            "credibility_tier": result.credibility_tier,
            "credibility_reason": result.credibility_reason,
            "downgrade_reason": qa.downgrade_reason,
        })

    return pd.DataFrame(rows)


def _load_object_coords(labels_path: Path) -> dict[str, tuple[float, float]]:
    """Load RA/Dec from a CSV with object_id, ra, dec columns.

    Returns dict of object_id → (ra_deg, dec_deg) for rows with valid coords.
    """
    coords: dict[str, tuple[float, float]] = {}
    with open(labels_path) as fh:
        for row in csv.DictReader(fh):
            oid = str(row.get("object_id") or "").strip()
            if not oid:
                continue
            try:
                ra = float(row["ra"])
                dec = float(row["dec"])
                if ra == ra and dec == dec:  # NaN check
                    coords[oid] = (ra, dec)
            except (KeyError, ValueError, TypeError):
                continue
    return coords


def main() -> None:
    parser = argparse.ArgumentParser(description="Crossmatch objects against TNS")
    parser.add_argument("--labels", required=True, help="Labels CSV with object_id column")
    parser.add_argument("--output", default="data/truth/tns_crossmatch.parquet")
    parser.add_argument("--bulk-csv", help="TNS bulk CSV for fast local crossmatch (no API calls)")
    parser.add_argument("--positional-fallback", action="store_true",
                        help="Enable positional RA/Dec fallback for objects not matched by name")
    parser.add_argument("--positional-radius", type=float, default=3.0,
                        help="Cone search radius in arcsec for positional fallback (default: 3.0)")
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

    # Load coordinates for positional fallback
    object_coords: dict[str, tuple[float, float]] | None = None
    if args.positional_fallback:
        object_coords = _load_object_coords(labels_path)
        print(f"Loaded coordinates for {len(object_coords)}/{len(object_ids)} objects")

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
            object_coords=object_coords,
            positional_radius_arcsec=args.positional_radius,
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

    # Credibility summary
    if "credibility_tier" in truth_df.columns:
        cred_counts = truth_df["credibility_tier"].value_counts().to_dict()
        if cred_counts:
            print(f"  Credibility: {cred_counts}")
    if "downgrade_reason" in truth_df.columns:
        downgrades = truth_df["downgrade_reason"].dropna()
        if len(downgrades) > 0:
            print(f"  Downgraded: {len(downgrades)} objects")
            for reason in downgrades.unique():
                n = (downgrades == reason).sum()
                print(f"    {reason}: {n}")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    truth_df.to_parquet(output_path, index=False)
    print(f"\nWrote TNS crossmatch truth table -> {output_path}")


if __name__ == "__main__":
    main()
