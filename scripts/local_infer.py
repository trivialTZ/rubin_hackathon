"""scripts/local_infer.py — Run local experts on per-epoch truncated lightcurves.

For each (object_id, n_det) pair, truncates the cached lightcurve to exactly
n_det detections (ordered by mjd) and runs SuperNNova / ParSNIP.

Output: data/silver/local_expert_outputs.json
  List of records: {expert, object_id, n_det, alert_mjd, class_probabilities, ...}
  Consumed by build_epoch_table_from_lc.py to fill snn_prob_ia / parsnip_prob_ia.

Usage (local smoke test — weights not required, returns stub probs):
    python scripts/local_infer.py --expert supernnova --limit 5 --max-n-det 5

Usage (SCC GPU, called by local_infer_gpu.sh):
    python scripts/local_infer.py --expert supernnova --lc-dir data/lightcurves \\
        --from-labels data/labels.csv --max-n-det 20
    python scripts/local_infer.py --expert parsnip --lc-dir data/lightcurves \\
        --from-labels data/labels.csv --max-n-det 20
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from debass.experts.local import ALL_LOCAL_EXPERTS

_SILVER_DIR = Path("data/silver")
_LC_DIR = Path("data/lightcurves")
_DEFAULT_LABELS = Path("data/labels.csv")


def _load_oids_from_labels(path: Path) -> list[str]:
    oids = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            oids.append(row["object_id"])
    return oids


def _load_oids_from_silver(silver_dir: Path, limit: int) -> list[str]:
    path = silver_dir / "broker_outputs.parquet"
    if not path.exists():
        return []
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        return list(df["object_id"].unique())[:limit]
    except Exception:
        return []


def _load_lightcurve(lc_dir: Path, oid: str) -> list[dict]:
    """Load cached lightcurve JSON for an object. Returns [] if missing."""
    path = lc_dir / f"{oid}.json"
    if not path.exists():
        return []
    try:
        with open(path) as fh:
            dets = json.load(fh)
        # Filter to positive detections only, sort by mjd
        pos = [
            d for d in dets
            if isinstance(d, dict) and d.get("isdiffpos") in ("t", "1", 1, True, "true")
        ]
        if not pos:
            pos = [d for d in dets if isinstance(d, dict)]
        return sorted(pos, key=lambda d: d.get("mjd") or d.get("jd") or 0.0)
    except Exception:
        return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local expert inference per (object_id, n_det) epoch"
    )
    parser.add_argument("--expert", default="all",
                        help="Expert name ('supernnova', 'parsnip') or 'all'")
    parser.add_argument("--objects", nargs="+", default=None,
                        help="Explicit list of object IDs")
    parser.add_argument("--from-labels", default=None,
                        help="CSV with object_id column (preferred for large runs)")
    parser.add_argument("--lc-dir", default=str(_LC_DIR),
                        help="Directory containing per-object lightcurve JSON files")
    parser.add_argument("--silver-dir", default=str(_SILVER_DIR))
    parser.add_argument("--max-n-det", type=int, default=20,
                        help="Maximum detection epoch to score")
    parser.add_argument("--limit", type=int, default=100000,
                        help="Max objects (for smoke tests)")
    args = parser.parse_args()

    lc_dir = Path(args.lc_dir)
    silver_dir = Path(args.silver_dir)

    # Resolve object list
    if args.objects:
        object_ids = args.objects[:args.limit]
    elif args.from_labels and Path(args.from_labels).exists():
        object_ids = _load_oids_from_labels(Path(args.from_labels))[:args.limit]
    else:
        object_ids = _load_oids_from_silver(silver_dir, args.limit)

    if not object_ids:
        print("No objects found. Run download_alerce_training.py first.")
        sys.exit(1)

    # Instantiate requested experts
    experts = [
        cls() for cls in ALL_LOCAL_EXPERTS
        if args.expert == "all" or cls.name == args.expert
    ]
    if not experts:
        print(f"No expert found for '{args.expert}'")
        sys.exit(1)

    for expert in experts:
        meta = expert.metadata()
        print(f"[{expert.name}] available={meta['available']}")

    # Per-(object, n_det) inference
    results = []
    for oid in object_ids:
        dets = _load_lightcurve(lc_dir, oid)
        if not dets:
            print(f"  {oid}: no lightcurve — skipping")
            continue

        n_epochs = min(len(dets), args.max_n_det)
        for n_det in range(1, n_epochs + 1):
            truncated = dets[:n_det]
            # MJD of the n_det-th detection (last in truncated slice)
            alert_mjd = truncated[-1].get("mjd") or truncated[-1].get("jd") or 0.0
            # Convert to JD for experts that use it internally
            alert_jd = float(alert_mjd) + 2400000.5

            for expert in experts:
                try:
                    out = expert.predict_epoch(oid, truncated, alert_jd)
                    results.append({
                        "expert": out.expert,
                        "object_id": oid,
                        "n_det": n_det,
                        "alert_mjd": float(alert_mjd),
                        "class_probabilities": out.class_probabilities,
                        "model_version": out.model_version,
                        "available": out.available,
                    })
                except Exception as exc:
                    print(f"  {oid} n_det={n_det} [{expert.name}]: ERROR — {exc}")

        if (object_ids.index(oid) + 1) % 100 == 0:
            print(f"  Processed {object_ids.index(oid)+1}/{len(object_ids)} objects...")

    out_path = silver_dir / "local_expert_outputs.json"
    silver_dir.mkdir(parents=True, exist_ok=True)

    # Merge with any existing records (e.g. SNN already written, now adding ParSNIP)
    existing: list[dict] = []
    if out_path.exists():
        try:
            with open(out_path) as fh:
                existing = json.load(fh)
        except Exception:
            existing = []

    # Build set of existing (expert, object_id, n_det) to avoid duplicates
    seen = {
        (r.get("expert"), r.get("object_id"), r.get("n_det"))
        for r in existing
    }
    new_records = [r for r in results
                   if (r.get("expert"), r.get("object_id"), r.get("n_det")) not in seen]
    combined = existing + new_records

    with open(out_path, "w") as fh:
        json.dump(combined, fh)
    print(f"\nWrote {len(new_records)} new records ({len(combined)} total) → {out_path}")
    print("Next: python scripts/build_epoch_table_from_lc.py")


if __name__ == "__main__":
    main()
