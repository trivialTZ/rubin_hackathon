"""scripts/fetch_lightcurves.py — Fetch and cache raw lightcurves for all training objects.

Lightcurves are stored in data/lightcurves/{oid}.json and are used by:
  - build_epoch_table.py  (to key broker scores by n_det)
  - local_infer_gpu.sh    (SCC: truncate to N detections, re-run SNN/ParSNIP)

Usage:
    python scripts/fetch_lightcurves.py
    python scripts/fetch_lightcurves.py --objects ZTF21abywdxt ZTF22aalpfln
    python scripts/fetch_lightcurves.py --from-labels data/labels.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from debass.access.alerce_history import AlerceHistoryFetcher


def _load_oids_from_labels(path: Path) -> list[str]:
    oids = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            oids.append(row["object_id"])
    return oids


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and cache lightcurves")
    parser.add_argument("--objects", nargs="+", default=None)
    parser.add_argument("--from-labels", default="data/labels.csv",
                        help="CSV with object_id column")
    parser.add_argument("--lc-dir", default="data/lightcurves")
    parser.add_argument("--max-epochs", type=int, default=20)
    args = parser.parse_args()

    lc_dir = Path(args.lc_dir)

    if args.objects:
        oids = args.objects
    elif Path(args.from_labels).exists():
        oids = _load_oids_from_labels(Path(args.from_labels))
    else:
        print(f"No objects specified and {args.from_labels} not found.")
        sys.exit(1)

    fetcher = AlerceHistoryFetcher(lc_dir=lc_dir)
    print(f"Fetching lightcurves for {len(oids)} objects → {lc_dir}")

    ok, skipped = 0, 0
    for oid in oids:
        cache = lc_dir / f"{oid}.json"
        if cache.exists():
            print(f"  {oid}: cached")
            ok += 1
            continue
        epochs = fetcher.fetch_epoch_scores(oid, max_epochs=args.max_epochs)
        if epochs:
            print(f"  {oid}: {len(epochs)} detection epochs")
            ok += 1
        else:
            print(f"  {oid}: no data")
            skipped += 1

    print(f"\nDone: {ok} fetched, {skipped} skipped")


if __name__ == "__main__":
    main()
