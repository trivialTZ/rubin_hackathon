"""scripts/download_alerce_training.py

Download labelled ZTF objects + lightcurves from ALeRCE for training.

Strategy:
  1. Query ALeRCE lc_classifier_transient for each SN class
     (SNIa, SNIbc, SNII, SLSN) with high probability.
  2. Query stamp_classifier for non-transient 'other' class objects.
  3. For each object fetch the full lightcurve.
  4. Write data/labels.csv and data/lightcurves/<oid>.json.

This script is designed to run on SCC (long-running, no GPU needed).
For a full training set use --limit 2000.
For a local smoke test use --limit 20.

Usage:
    python scripts/download_alerce_training.py --limit 20          # smoke test
    python scripts/download_alerce_training.py --limit 2000        # full run
    python scripts/download_alerce_training.py --limit 2000 --resume  # skip cached
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ------------------------------------------------------------------ #
# Class quota: fraction of total limit per class                       #
# ------------------------------------------------------------------ #
_CLASS_QUOTA = [
    # (classifier,              class_name,  ternary_label,     fraction, min_prob)
    ("lc_classifier_transient", "SNIa",      "snia",            0.35,    0.7),
    ("lc_classifier_transient", "SNII",      "nonIa_snlike",    0.20,    0.6),
    ("lc_classifier_transient", "SNIbc",     "nonIa_snlike",    0.15,    0.6),
    ("lc_classifier_transient", "SLSN",      "nonIa_snlike",    0.05,    0.5),
    ("stamp_classifier",        "AGN",       "other",           0.10,    0.7),
    ("stamp_classifier",        "asteroid",  "other",           0.08,    0.7),
    ("stamp_classifier",        "bogus",     "other",           0.05,    0.7),
    ("lc_classifier",           "VS",        "other",           0.02,    0.5),
]


def _query_class(
    client,
    classifier: str,
    class_name: str,
    n: int,
    min_prob: float,
    seen: set[str],
) -> list[dict]:
    """Query ALeRCE for up to n objects of a given class, skipping seen OIDs."""
    results = []
    page = 1
    while len(results) < n:
        needed = n - len(results)
        try:
            df = client.query_objects(
                format="pandas",
                classifier=classifier,
                class_name=class_name,
                probability=min_prob,
                page_size=min(needed * 3, 100),
                page=page,
            )
        except Exception as exc:
            print(f"    query error (page {page}): {exc}")
            break
        if df is None or len(df) == 0:
            break
        oids = df["oid"].tolist() if "oid" in df.columns else df.index.tolist()
        for oid in oids:
            oid = str(oid)
            if oid not in seen and len(results) < n:
                seen.add(oid)
                results.append(oid)
        if len(oids) < min(needed * 3, 100):
            break  # no more pages
        page += 1
        time.sleep(0.1)  # polite rate limiting
    return results


def _fetch_lightcurve(client, oid: str, lc_dir: Path) -> bool:
    """Fetch and cache lightcurve. Returns True on success."""
    cache = lc_dir / f"{oid}.json"
    if cache.exists():
        return True
    try:
        df = client.query_lightcurve(oid=oid, format="pandas")
        if "detections" in df.columns:
            records = df["detections"].iloc[0]
            if not isinstance(records, list):
                return False
        else:
            records = df.to_dict(orient="records")
        if not records:
            return False
        # Serialise — convert numpy types
        clean = []
        for r in records:
            clean.append({k: (float(v) if hasattr(v, 'item') else v)
                          for k, v in r.items()})
        with open(cache, "w") as fh:
            json.dump(clean, fh)
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ALeRCE training data (objects + lightcurves)"
    )
    parser.add_argument("--limit",    type=int, default=100,
                        help="Total number of labelled objects to download")
    parser.add_argument("--lc-dir",   default="data/lightcurves")
    parser.add_argument("--labels-out", default="data/labels.csv")
    parser.add_argument("--resume",   action="store_true",
                        help="Skip objects whose lightcurves are already cached")
    parser.add_argument("--rate-limit", type=float, default=0.15,
                        help="Seconds to wait between LC fetches (be polite)")
    args = parser.parse_args()

    lc_dir = Path(args.lc_dir)
    lc_dir.mkdir(parents=True, exist_ok=True)
    labels_out = Path(args.labels_out)

    try:
        from alerce.core import Alerce
        client = Alerce()
    except ImportError:
        print("alerce package not installed. Run: pip install alerce")
        sys.exit(1)

    # --- Step 1: collect object IDs per class ---
    print(f"=== Collecting {args.limit} labelled objects from ALeRCE ===")
    print(f"PROGRESS download phase=collect current=0 total={args.limit} ok=0 failed=0")
    seen_oids: set[str] = set()
    collected: list[tuple[str, str]] = []  # (oid, ternary_label)

    for classifier, class_name, ternary_label, frac, min_prob in _CLASS_QUOTA:
        n = max(1, int(args.limit * frac))
        print(f"  {classifier}/{class_name} → {ternary_label}: requesting {n} objects...")
        oids = _query_class(client, classifier, class_name, n, min_prob, seen_oids)
        for oid in oids:
            collected.append((oid, ternary_label))
        print(f"    got {len(oids)} objects (total so far: {len(collected)})")
        print(
            f"PROGRESS download phase=collect current={len(collected)} "
            f"total={args.limit} ok=0 failed=0"
        )

    if not collected:
        print("No objects collected. Check ALeRCE connectivity.")
        sys.exit(1)

    from collections import Counter
    dist = Counter(lbl for _, lbl in collected)
    print(f"\nLabel distribution: {dict(dist)}")
    print(f"Total objects: {len(collected)}")

    # --- Step 2: fetch lightcurves ---
    print(f"\n=== Fetching {len(collected)} lightcurves → {lc_dir} ===")
    print(f"PROGRESS download phase=fetch current=0 total={len(collected)} ok=0 failed=0")
    ok, failed = 0, 0
    for i, (oid, label) in enumerate(collected):
        cache = lc_dir / f"{oid}.json"
        if args.resume and cache.exists():
            ok += 1
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(collected)}] {ok} cached, {failed} failed")
                print(
                    f"PROGRESS download phase=fetch current={i+1} "
                    f"total={len(collected)} ok={ok} failed={failed}"
                )
            continue
        success = _fetch_lightcurve(client, oid, lc_dir)
        if success:
            ok += 1
        else:
            failed += 1
            print(f"  [{i+1}] {oid}: FAILED")
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(collected)}] {ok} ok, {failed} failed")
            print(
                f"PROGRESS download phase=fetch current={i+1} "
                f"total={len(collected)} ok={ok} failed={failed}"
            )
        time.sleep(args.rate_limit)

    print(f"\nLightcurves: {ok} ok, {failed} failed")

    # --- Step 3: write labels.csv (only for objects with lightcurves) ---
    labels_out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(labels_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["object_id", "label"])
        writer.writeheader()
        for oid, label in collected:
            if (lc_dir / f"{oid}.json").exists():
                writer.writerow({"object_id": oid, "label": label})
                written += 1

    print(f"Wrote {written} labels → {labels_out}")
    print(
        f"PROGRESS download phase=complete current={len(collected)} "
        f"total={len(collected)} ok={ok} failed={failed} written={written}"
    )
    print(f"\nNext: python scripts/build_epoch_table_from_lc.py")


if __name__ == "__main__":
    main()
