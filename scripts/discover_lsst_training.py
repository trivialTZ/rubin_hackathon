"""Discover LSST transient candidates for mixed-survey training.

Samples a REPRESENTATIVE population across all ALeRCE stamp classes
(SN, bogus, AGN, VS, asteroid) so the trust model learns both when
to trust and when to distrust broker classifiers.

Primary source: ALeRCE LSST API (public, no token, no rate limit).
Optionally enriches with Lasair Sherlock classifications (needs LASAIR_TOKEN).

Usage:
    python3 scripts/discover_lsst_training.py --max-candidates 400
    python3 scripts/discover_lsst_training.py --max-candidates 400 --with-lasair
    python3 scripts/discover_lsst_training.py --sn 200 --bogus 100 --agn 50 --vs 50 --asteroid 20
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Default class budget — mirrors what the model sees in production.
# SN is oversampled (science target); contaminants ensure the model
# learns to reject false positives.
DEFAULT_CLASS_BUDGET = {
    "SN": 200,
    "bogus": 100,
    "AGN": 50,
    "VS": 50,
    "asteroid": 20,
}


def _discover_class_from_alerce(
    *,
    class_name: str,
    max_count: int,
    min_detections: int,
) -> list[dict]:
    """Query ALeRCE LSST for one stamp class. No token needed."""
    from alerce.core import Alerce

    a = Alerce()
    candidates = []
    page = 1

    while len(candidates) < max_count:
        try:
            result = a.query_objects(
                survey="lsst",
                format="pandas",
                page_size=100,
                page=page,
                classifier="stamp_classifier_rubin_beta",
                class_name=class_name,
                order_by="n_det",
                order_mode="DESC",
            )
        except Exception as e:
            print(f"    {class_name} page {page} error: {e}")
            break

        if len(result) == 0:
            break

        for _, row in result.iterrows():
            n_det = int(row.get("n_det", 0))
            if n_det < min_detections:
                continue
            candidates.append({
                "object_id": str(row["oid"]),
                "ra": float(row.get("meanra", 0)),
                "dec": float(row.get("meandec", 0)),
                "n_det": n_det,
                "first_mjd": float(row.get("firstmjd", 0)),
                "last_mjd": float(row.get("lastmjd", 0)),
                "alerce_stamp_class": class_name,
                "alerce_stamp_prob": float(row.get("probability", 0)),
                "discovery_broker": "alerce_lsst",
                "survey": "LSST",
            })

        page += 1
        if len(candidates) >= max_count:
            break

    return candidates[:max_count]


def _enrich_with_alerce_probs(candidates: list[dict]) -> None:
    """Fetch full stamp probabilities for each candidate (parallel)."""
    from alerce.core import Alerce

    a = Alerce()
    lookup = {c["object_id"]: c for c in candidates}

    def _fetch_one(oid):
        try:
            probs = a.query_probabilities(oid=oid, survey="lsst", format="pandas")
            if len(probs) == 0:
                return oid, {}
            scores = {}
            for _, row in probs.iterrows():
                cn = row.get("classifier_name", "")
                cls = row.get("class_name", "")
                scores[f"{cn}__{cls}"] = float(row.get("probability", 0))
            return oid, scores
        except Exception:
            return oid, {}

    print(f"  Fetching full ALeRCE probabilities for {len(candidates)} objects (8 threads)...")
    n_enriched = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch_one, c["object_id"]): c["object_id"] for c in candidates}
        for future in as_completed(futures):
            oid, scores = future.result()
            if scores and oid in lookup:
                c = lookup[oid]
                c["alerce_stamp_p_sn"] = scores.get("stamp_classifier_rubin_beta__SN", "")
                c["alerce_stamp_p_bogus"] = scores.get("stamp_classifier_rubin_beta__bogus", "")
                c["alerce_stamp_p_agn"] = scores.get("stamp_classifier_rubin_beta__AGN", "")
                c["alerce_stamp_p_vs"] = scores.get("stamp_classifier_rubin_beta__VS", "")
                n_enriched += 1
    print(f"  Enriched {n_enriched}/{len(candidates)} with full probabilities")


def _enrich_with_lasair(candidates: list[dict]) -> None:
    """Fetch Sherlock classifications from Lasair LSST (needs token)."""
    import requests

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    token = os.environ.get("LASAIR_LSST_TOKEN") or os.environ.get("LASAIR_TOKEN", "")
    endpoint = os.environ.get("LASAIR_LSST_ENDPOINT",
               os.environ.get("LASAIR_ENDPOINT", "https://lasair.lsst.ac.uk"))

    if not token:
        print("  Skipping Lasair enrichment (no LASAIR_TOKEN)")
        return

    api_base = endpoint.rstrip("/")
    if not api_base.endswith("/api"):
        api_base += "/api"

    lookup = {c["object_id"]: c for c in candidates}
    oids = list(lookup.keys())

    print(f"  Querying Lasair LSST Sherlock for {len(oids)} objects...")
    n_matched = 0
    chunk_size = 50
    for i in range(0, len(oids), chunk_size):
        chunk = oids[i:i + chunk_size]
        oid_list = ", ".join(str(oid) for oid in chunk)
        try:
            resp = requests.post(
                f"{api_base}/query/",
                headers={"Authorization": f"Token {token}", "Accept": "application/json"},
                data={
                    "selected": "objects.diaObjectId, sherlock_classifications.classification",
                    "tables": "objects, sherlock_classifications",
                    "conditions": f"objects.diaObjectId IN ({oid_list}) "
                                  "AND objects.diaObjectId = sherlock_classifications.diaObjectId",
                    "limit": str(len(chunk)),
                },
                timeout=30,
            )
            resp.raise_for_status()
            rows = resp.json()
            if isinstance(rows, list):
                for row in rows:
                    oid = str(row.get("diaObjectId", ""))
                    cls = row.get("classification", "")
                    if oid in lookup and cls:
                        lookup[oid]["sherlock_class"] = cls
                        n_matched += 1
        except Exception as e:
            print(f"    Lasair chunk error: {e}")

    print(f"  Lasair: {n_matched}/{len(oids)} have Sherlock classifications")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover LSST candidates for mixed-survey training. "
                    "Samples across all stamp classes for representative training.",
    )
    parser.add_argument("--max-candidates", type=int, default=None,
                        help="Total candidates (overrides per-class budgets, split proportionally)")
    parser.add_argument("--sn", type=int, default=None, help="SN class budget")
    parser.add_argument("--bogus", type=int, default=None, help="Bogus class budget")
    parser.add_argument("--agn", type=int, default=None, help="AGN class budget")
    parser.add_argument("--vs", type=int, default=None, help="VS class budget")
    parser.add_argument("--asteroid", type=int, default=None, help="Asteroid class budget")
    parser.add_argument("--min-detections", type=int, default=2)
    parser.add_argument("--output", default="data/lsst_candidates.csv")
    parser.add_argument("--with-lasair", action="store_true",
                        help="Also fetch Sherlock from Lasair LSST")
    parser.add_argument("--skip-probs", action="store_true",
                        help="Skip per-object probability enrichment")
    args = parser.parse_args()

    # Build class budget
    budget = dict(DEFAULT_CLASS_BUDGET)
    if args.sn is not None:
        budget["SN"] = args.sn
    if args.bogus is not None:
        budget["bogus"] = args.bogus
    if args.agn is not None:
        budget["AGN"] = args.agn
    if args.vs is not None:
        budget["VS"] = args.vs
    if args.asteroid is not None:
        budget["asteroid"] = args.asteroid

    # If --max-candidates given, scale proportionally
    if args.max_candidates is not None:
        total_default = sum(budget.values())
        for cls in budget:
            budget[cls] = max(1, int(budget[cls] * args.max_candidates / total_default))

    total = sum(budget.values())
    print(f"=== LSST Training Set Discovery ===")
    print(f"Target: {total} objects across {len(budget)} classes")
    for cls, n in budget.items():
        print(f"  {cls:10s}: {n}")
    print()

    # Step 1: Discover from ALeRCE per class
    all_candidates = []
    for class_name, target_count in budget.items():
        print(f"[{class_name}] Discovering up to {target_count} objects...")
        found = _discover_class_from_alerce(
            class_name=class_name,
            max_count=target_count,
            min_detections=args.min_detections,
        )
        print(f"  → {len(found)} objects (n_det: "
              f"{min(c['n_det'] for c in found) if found else 0}"
              f"—{max(c['n_det'] for c in found) if found else 0})")
        all_candidates.extend(found)

    if not all_candidates:
        print("No candidates found.")
        sys.exit(0)

    print(f"\nTotal: {len(all_candidates)} LSST candidates")

    # Step 2: Enrich with full ALeRCE probabilities
    if not args.skip_probs:
        _enrich_with_alerce_probs(all_candidates)

    # Step 3: Optionally enrich with Lasair Sherlock
    if args.with_lasair:
        _enrich_with_lasair(all_candidates)

    # Deduplicate within this run
    seen = set()
    deduped = []
    for c in all_candidates:
        if c["object_id"] not in seen:
            seen.add(c["object_id"])
            deduped.append(c)
    if len(deduped) < len(all_candidates):
        print(f"\nDeduplicated: {len(all_candidates)} → {len(deduped)} (removed {len(all_candidates) - len(deduped)} duplicates)")
    all_candidates = deduped

    # Merge with existing output file (append new, update existing)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "object_id", "ra", "dec", "n_det", "first_mjd", "last_mjd",
        "alerce_stamp_class", "alerce_stamp_prob",
        "alerce_stamp_p_sn", "alerce_stamp_p_bogus", "alerce_stamp_p_agn", "alerce_stamp_p_vs",
        "sherlock_class", "discovery_broker", "survey",
    ]

    existing: dict[str, dict] = {}
    if out_path.exists():
        with open(out_path) as fh:
            for row in csv.DictReader(fh):
                existing[row["object_id"]] = row

    n_new = 0
    n_updated = 0
    for c in all_candidates:
        oid = c["object_id"]
        if oid in existing:
            # Update: keep new data (fresher n_det, probs)
            existing[oid].update({k: v for k, v in c.items() if v not in (None, "", 0, 0.0)})
            n_updated += 1
        else:
            existing[oid] = c
            n_new += 1

    # Write merged output
    merged = list(existing.values())
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)

    # Summary
    from collections import Counter
    counts = Counter(c.get("alerce_stamp_class", "?") for c in merged)
    print(f"\n{'='*50}")
    print(f"Output: {out_path}")
    print(f"  Total:   {len(merged)} objects ({n_new} new, {n_updated} updated)")
    print(f"  Classes:")
    for cls, n in counts.most_common():
        print(f"    {cls:10s}: {n}")


if __name__ == "__main__":
    main()
