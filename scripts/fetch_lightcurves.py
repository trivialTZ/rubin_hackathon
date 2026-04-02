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
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from debass_meta.access.alerce_history import AlerceHistoryFetcher
from debass_meta.access.associations import (
    load_lsst_ztf_associations,
    resolve_object_reference,
)


def _load_oids_from_labels(path: Path) -> list[str]:
    oids = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            oids.append(row["object_id"])
    return oids


def fetch_lightcurves_for_objects(
    *,
    object_ids: list[str],
    fetcher: AlerceHistoryFetcher,
    lc_dir: Path,
    max_epochs: int = 20,
    associations: dict[str, dict[str, object]] | None = None,
) -> tuple[int, int]:
    ok = 0
    skipped = 0

    for oid in object_ids:
        cache = lc_dir / f"{oid}.json"
        if cache.exists():
            print(f"  {oid}: cached")
            ok += 1
            continue

        reference = resolve_object_reference(
            oid,
            broker="alerce_history",
            associations=associations,
        )
        if not reference.can_query:
            print(f"  {oid}: {reference.resolution_reason or 'no lightcurve source'}")
            skipped += 1
            continue

        epochs = fetcher.fetch_epoch_scores(reference.requested_object_id, max_epochs=max_epochs)
        if not epochs:
            print(f"  {oid}: no data")
            skipped += 1
            continue

        if reference.requested_object_id != oid:
            source_cache = lc_dir / f"{reference.requested_object_id}.json"
            if source_cache.exists():
                shutil.copyfile(source_cache, cache)
                print(
                    f"  {oid}: {len(epochs)} detection epochs "
                    f"(via {reference.requested_object_id})"
                )
            else:
                print(f"  {oid}: no cached lightcurve for associated object {reference.requested_object_id}")
                skipped += 1
                continue
        else:
            print(f"  {oid}: {len(epochs)} detection epochs")
        ok += 1

    return ok, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and cache lightcurves")
    parser.add_argument("--objects", nargs="+", default=None)
    parser.add_argument("--from-labels", default="data/labels.csv",
                        help="CSV with object_id column")
    parser.add_argument("--association-csv", default=None,
                        help="Optional LSST→ZTF association CSV for LSST primary objects")
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
    associations = load_lsst_ztf_associations(Path(args.association_csv)) if args.association_csv else {}
    print(f"Fetching lightcurves for {len(oids)} objects → {lc_dir}")

    ok, skipped = fetch_lightcurves_for_objects(
        object_ids=oids,
        fetcher=fetcher,
        lc_dir=lc_dir,
        max_epochs=args.max_epochs,
        associations=associations,
    )

    print(f"\nDone: {ok} fetched, {skipped} skipped")


if __name__ == "__main__":
    main()
