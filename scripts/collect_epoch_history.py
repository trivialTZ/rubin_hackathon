"""Collect per-detection classification history for training objects.

For each object in labels.csv, runs local classifiers at every epoch
(n_det=1,2,3,...) on the cached lightcurve, producing genuine per-epoch
scores with temporal_exactness="rerun_exact".

This solves the key limitation: broker APIs give static object-level
snapshots, but the trust model needs to see how classifications evolve
with detection count to learn when experts become reliable.

Output: data/silver/local_expert_outputs/{expert}/part-latest.parquet

Usage:
    python3 scripts/collect_epoch_history.py --from-labels data/labels.csv
    python3 scripts/collect_epoch_history.py --from-labels data/labels.csv --experts alerce_lc supernnova
    python3 scripts/collect_epoch_history.py --from-labels data/labels.csv --max-n-det 20
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from debass_meta.features.detection import normalize_lightcurve
from debass_meta.features.lightcurve import extract_features


def _load_object_ids(path: Path) -> list[str]:
    with open(path) as fh:
        return [row["object_id"] for row in csv.DictReader(fh)]


def _load_lightcurve(lc_dir: Path, object_id: str) -> list[dict] | None:
    path = lc_dir / f"{object_id}.json"
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def _run_alerce_lc_expert(
    detections_truncated: list[dict],
    n_det: int,
) -> dict | None:
    """Run AlerceLCExpert on truncated detections."""
    try:
        from debass_meta.experts.local.alerce_lc import AlerceLCExpert
        expert = AlerceLCExpert()
        if not expert.is_available():
            return None
        result = expert.predict_epoch(detections_truncated, n_det=n_det)
        if result is None:
            return None
        return {
            "expert": "alerce_lc",
            "class_probabilities": result.class_probabilities,
            "available": True,
        }
    except Exception:
        return None


def _run_supernnova_expert(
    detections_truncated: list[dict],
    n_det: int,
) -> dict | None:
    """Run SuperNNova on truncated detections."""
    try:
        from debass_meta.experts.local.supernnova import SuperNNovaExpert
        expert = SuperNNovaExpert()
        if not expert.is_available():
            return None
        result = expert.predict_epoch(detections_truncated, n_det=n_det)
        if result is None:
            return None
        return {
            "expert": "supernnova",
            "class_probabilities": result.class_probabilities,
            "available": True,
        }
    except Exception:
        return None


# Map expert name → runner function
_EXPERT_RUNNERS = {
    "alerce_lc": _run_alerce_lc_expert,
    "supernnova": _run_supernnova_expert,
}


def collect_epoch_history(
    *,
    object_ids: list[str],
    lc_dir: Path,
    experts: list[str],
    max_n_det: int = 20,
    output_dir: Path,
) -> dict[str, int]:
    """Run local experts at each epoch for all objects.

    Returns dict of {expert: n_rows_produced}.
    """
    stats: dict[str, int] = {e: 0 for e in experts}
    all_rows: dict[str, list[dict]] = {e: [] for e in experts}

    for i, oid in enumerate(object_ids):
        if i % 100 == 0:
            print(f"  [{i}/{len(object_ids)}] Processing {oid}...", flush=True)

        raw_dets = _load_lightcurve(lc_dir, oid)
        if not raw_dets:
            continue

        # Normalize detections
        ndets = normalize_lightcurve(raw_dets)

        # Sort by MJD, filter positives
        ndets.sort(key=lambda d: d.get("mjd", 0))
        pos_dets = [d for d in ndets if d.get("is_positive", True)]
        if not pos_dets:
            pos_dets = ndets

        n_total = min(len(pos_dets), max_n_det)

        for n_det in range(1, n_total + 1):
            truncated = pos_dets[:n_det]
            alert_mjd = truncated[-1].get("mjd", 0)

            for expert_name in experts:
                runner = _EXPERT_RUNNERS.get(expert_name)
                if runner is None:
                    continue

                result = runner(truncated, n_det)
                if result is None:
                    continue

                row = {
                    "object_id": oid,
                    "expert": expert_name,
                    "model_name": expert_name,
                    "n_det": n_det,
                    "alert_mjd": alert_mjd,
                    "alert_jd": alert_mjd + 2400000.5 if alert_mjd < 2400000.5 else alert_mjd,
                    "class_probabilities": json.dumps(result["class_probabilities"]),
                    "available": result["available"],
                    "temporal_exactness": "rerun_exact",
                    "survey": "LSST" if str(oid).isdigit() else "ZTF",
                }
                all_rows[expert_name].append(row)
                stats[expert_name] += 1

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    for expert_name, rows in all_rows.items():
        if not rows:
            continue
        expert_dir = output_dir / expert_name
        expert_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        out_path = expert_dir / "part-latest.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  {expert_name}: {len(rows)} epoch rows → {out_path}")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect per-epoch classification history from local experts"
    )
    parser.add_argument("--from-labels", default="data/labels.csv")
    parser.add_argument("--lc-dir", default="data/lightcurves")
    parser.add_argument("--experts", nargs="+", default=list(_EXPERT_RUNNERS.keys()),
                        help=f"Experts to run (default: {list(_EXPERT_RUNNERS.keys())})")
    parser.add_argument("--max-n-det", type=int, default=20)
    parser.add_argument("--output-dir", default="data/silver/local_expert_outputs")
    args = parser.parse_args()

    object_ids = _load_object_ids(Path(args.from_labels))
    print(f"=== Collecting per-epoch history ===")
    print(f"Objects: {len(object_ids)}")
    print(f"Experts: {args.experts}")
    print(f"Max n_det: {args.max_n_det}")
    print()

    # Check which experts are available
    available = []
    for expert_name in args.experts:
        if expert_name == "alerce_lc":
            try:
                from debass_meta.experts.local.alerce_lc import AlerceLCExpert
                if AlerceLCExpert().is_available():
                    available.append(expert_name)
                    print(f"  {expert_name}: available")
                else:
                    print(f"  {expert_name}: model not trained (run train_alerce_lc.py first)")
            except Exception as e:
                print(f"  {expert_name}: unavailable ({e})")
        elif expert_name == "supernnova":
            try:
                from debass_meta.experts.local.supernnova import SuperNNovaExpert
                if SuperNNovaExpert().is_available():
                    available.append(expert_name)
                    print(f"  {expert_name}: available")
                else:
                    print(f"  {expert_name}: weights not found")
            except Exception as e:
                print(f"  {expert_name}: unavailable ({e})")

    if not available:
        print("\nNo local experts available. Train alerce_lc first:")
        print("  python3 scripts/train_alerce_lc.py --from-labels data/labels.csv")
        sys.exit(1)

    print()
    stats = collect_epoch_history(
        object_ids=object_ids,
        lc_dir=Path(args.lc_dir),
        experts=available,
        max_n_det=args.max_n_det,
        output_dir=Path(args.output_dir),
    )

    print(f"\n=== Done ===")
    for expert_name, count in stats.items():
        print(f"  {expert_name}: {count} epoch rows")
    total = sum(stats.values())
    print(f"  Total: {total} epoch rows across {len(available)} experts")


if __name__ == "__main__":
    main()
