"""scripts/fetch_training_data.py

Fetch labelled training data from ALeRCE (TNS-classified SNe) and
optionally Fink. Writes bronze → silver → gold and a labels.csv.

Workflow:
  1. Query ALeRCE for objects with known LC classifier labels
     (SNIa, SNIbc, SNII, SLSN → mapped to ternary target)
  2. Fetch probabilities + lightcurves for each object
  3. Write bronze Parquet
  4. Run bronze→silver→gold transforms
  5. Write data/labels.csv with object_id,label columns

Usage:
    python scripts/fetch_training_data.py --limit 10      # smoke test
    python scripts/fetch_training_data.py --limit 500
    python scripts/fetch_training_data.py --limit 500 --brokers alerce fink
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from debass_meta.access.alerce import AlerceAdapter
from debass_meta.access.fink import FinkAdapter
from debass_meta.ingest.bronze import write_bronze
from debass_meta.ingest.silver import bronze_to_silver
from debass_meta.ingest.gold import silver_to_gold

# ------------------------------------------------------------------ #
# Ternary label mapping from ALeRCE LC classifier classes             #
# ------------------------------------------------------------------ #
_ALERCE_TO_TERNARY = {
    "SNIa":     "snia",
    "SNIbc":    "nonIa_snlike",
    "SNII":     "nonIa_snlike",
    "SLSN":     "nonIa_snlike",
    "AGN":      "other",
    "VS":       "other",
    "asteroid": "other",
    "bogus":    "other",
}

# Per-class quota fractions and which ALeRCE classifier to use.
# lc_classifier_transient is the dedicated SN sub-classifier and has
# the best coverage for SNIa/SNIbc/SNII/SLSN.
# stamp_classifier covers AGN/bogus/asteroid for the 'other' class.
_CLASS_QUOTA = {
    # (classifier, class_name, ternary_label, fraction)
    ("lc_classifier_transient", "SNIa",     "snia"):         0.40,
    ("lc_classifier_transient", "SNII",     "nonIa_snlike"): 0.25,
    ("lc_classifier_transient", "SNIbc",    "nonIa_snlike"): 0.15,
    ("lc_classifier_transient", "SLSN",     "nonIa_snlike"): 0.05,
    ("stamp_classifier",        "AGN",      "other"):         0.08,
    ("stamp_classifier",        "asteroid", "other"):         0.04,
    ("stamp_classifier",        "bogus",    "other"):         0.03,
}


def _query_alerce_objects(adapter: AlerceAdapter, limit: int) -> list[dict]:
    """Query ALeRCE for classified objects. Returns list of {oid, ternary_label}."""
    if adapter._client is None:
        print("  [alerce] client not available — cannot query objects")
        return []

    seen_oids: set[str] = set()
    objects = []
    for (classifier, cls_name, ternary_label), frac in _CLASS_QUOTA.items():
        n = max(1, int(limit * frac))
        try:
            df = adapter._client.query_objects(
                format="pandas",
                classifier=classifier,
                class_name=cls_name,
                probability=0.5,   # lowered threshold for better coverage
                page_size=n * 2,   # fetch extra to account for duplicates
                page=1,
            )
            if df is not None and len(df) > 0:
                oids = df["oid"].tolist() if "oid" in df.columns else df.index.tolist()
                added = 0
                for oid in oids:
                    oid = str(oid)
                    if oid not in seen_oids and added < n:
                        seen_oids.add(oid)
                        objects.append({"oid": oid, "ternary_label": ternary_label})
                        added += 1
                print(f"  [alerce] {classifier}/{cls_name} → {ternary_label}: got {added} objects")
            else:
                print(f"  [alerce] {classifier}/{cls_name}: no results")
        except Exception as exc:
            print(f"  [alerce] {classifier}/{cls_name}: query error — {exc}")
    return objects


def _fetch_and_write_bronze(
    adapter,
    oids: list[str],
    bronze_dir: Path,
) -> list[str]:
    """Fetch broker outputs for all oids, write bronze. Returns successful oids."""
    outputs = []
    successful = []
    for oid in oids:
        try:
            out = adapter.fetch_object(oid)
            outputs.append(out)
            successful.append(oid)
            tag = "(fixture)" if out.fixture_used else "(live)"
            print(f"    {oid}: {len(out.fields)} fields {tag}")
        except Exception as exc:
            print(f"    {oid}: SKIP — {exc}")
    if outputs:
        path = write_bronze(outputs, bronze_dir=bronze_dir)
        print(f"  [{adapter.name}] wrote {len(outputs)} records → {path}")
    return successful


def _write_labels(label_map: dict[str, str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["object_id", "label"])
        writer.writeheader()
        for oid, lbl in sorted(label_map.items()):
            writer.writerow({"object_id": oid, "label": lbl})
    print(f"  labels.csv: {len(label_map)} objects → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch labelled training data")
    parser.add_argument("--limit", type=int, default=10,
                        help="Total number of labelled objects to fetch")
    parser.add_argument("--brokers", nargs="+", default=["alerce"],
                        choices=["alerce", "fink"],
                        help="Which brokers to fetch from (fink adds score features)")
    parser.add_argument("--bronze-dir", default="data/bronze")
    parser.add_argument("--silver-dir", default="data/silver")
    parser.add_argument("--gold-dir",   default="data/gold")
    parser.add_argument("--labels-out", default="data/labels.csv")
    args = parser.parse_args()

    bronze_dir = Path(args.bronze_dir)
    silver_dir = Path(args.silver_dir)
    gold_dir   = Path(args.gold_dir)
    labels_out = Path(args.labels_out)

    print(f"=== Fetching {args.limit} labelled objects ===")

    # ---- Step 1: query ALeRCE for classified objects ---- #
    alerce = AlerceAdapter()
    print("\n[1/5] Querying ALeRCE for classified objects...")
    classified = _query_alerce_objects(alerce, args.limit)

    if not classified:
        print("  No objects returned. Check ALeRCE connectivity (run scripts/probe.py).")
        sys.exit(1)

    oids = [o["oid"] for o in classified]
    label_map = {o["oid"]: o["ternary_label"] for o in classified}
    print(f"  Total objects: {len(oids)}")
    from collections import Counter
    print(f"  Label distribution: {dict(Counter(label_map.values()))}")

    # ---- Step 2: fetch ALeRCE broker outputs ---- #
    print("\n[2/5] Fetching ALeRCE broker outputs...")
    _fetch_and_write_bronze(alerce, oids, bronze_dir)

    # ---- Step 3: optionally fetch Fink scores ---- #
    if "fink" in args.brokers:
        print("\n[3/5] Fetching Fink broker outputs...")
        fink = FinkAdapter()
        _fetch_and_write_bronze(fink, oids, bronze_dir)
    else:
        print("\n[3/5] Skipping Fink (not in --brokers)")

    # ---- Step 4: bronze → silver → gold ---- #
    print("\n[4/5] Normalising bronze → silver → gold...")
    try:
        silver_path = bronze_to_silver(bronze_dir=bronze_dir, silver_dir=silver_dir)
        print(f"  silver: {silver_path}")
        gold_path = silver_to_gold(
            silver_dir=silver_dir,
            gold_dir=gold_dir,
            label_map=label_map,
            label_source="alerce_lc_classifier",
        )
        print(f"  gold:   {gold_path}")
    except Exception as exc:
        print(f"  Normalise error: {exc}")
        print("  (This is expected if no data was fetched live — check bronze dir.)")

    # ---- Step 5: write labels.csv ---- #
    print("\n[5/5] Writing labels.csv...")
    _write_labels(label_map, labels_out)

    print("\n=== Done ===")
    print(f"  Next: python scripts/train.py --labels {labels_out}")


if __name__ == "__main__":
    main()
