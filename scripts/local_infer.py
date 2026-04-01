"""scripts/local_infer.py — Run local experts on per-epoch truncated lightcurves.

For each (object_id, n_det) pair, truncates the cached lightcurve to exactly
n_det detections (ordered by mjd) and runs SuperNNova / ParSNIP.

Outputs:
  - `data/silver/local_expert_outputs/<expert>/part-latest.parquet`
  - `data/silver/local_expert_outputs.json` (legacy compatibility for the baseline path)

Usage (local smoke test — weights not required, returns stub probs):
    python scripts/local_infer.py --expert supernnova --limit 5 --max-n-det 5

Usage (SCC GPU, called by local_infer_gpu.sh):
    python scripts/local_infer.py --expert supernnova --lc-dir data/lightcurves \\
        --from-labels data/labels.csv --max-n-det 20 --require-live-model
    python scripts/local_infer.py --expert parsnip --lc-dir data/lightcurves \\
        --from-labels data/labels.csv --max-n-det 20 --require-live-model
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from debass_meta.experts.local import ALL_LOCAL_EXPERTS, CPU_LOCAL_EXPERTS, GPU_LOCAL_EXPERTS

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
    path = silver_dir / "broker_events.parquet"
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


def _record_key(record: dict) -> tuple[str | None, str | None, int | None]:
    n_det = record.get("n_det")
    return (
        record.get("expert"),
        record.get("object_id"),
        int(n_det) if n_det is not None else None,
    )


def _merge_records(existing: list[dict], new_records: list[dict]) -> tuple[list[dict], int, int]:
    """Merge local expert outputs, replacing stale duplicates with the new record."""
    combined: list[dict] = []
    index: dict[tuple[str | None, str | None, int | None], int] = {}

    for record in existing:
        key = _record_key(record)
        index[key] = len(combined)
        combined.append(record)

    refreshed = 0
    appended = 0
    for record in new_records:
        key = _record_key(record)
        if key in index:
            combined[index[key]] = record
            refreshed += 1
        else:
            index[key] = len(combined)
            combined.append(record)
            appended += 1

    return combined, appended, refreshed


def _write_parquet_partitions(records: list[dict], silver_dir: Path) -> list[Path]:
    import pandas as pd

    out_paths: list[Path] = []
    base_dir = silver_dir / "local_expert_outputs"
    for expert in sorted({str(record.get("expert")) for record in records if record.get("expert")}):
        expert_records = [record for record in records if record.get("expert") == expert]
        if not expert_records:
            continue
        df = pd.DataFrame(expert_records).copy()
        if "class_probabilities" in df.columns:
            df["class_probabilities"] = df["class_probabilities"].apply(
                lambda value: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
            )
        out_dir = base_dir / expert
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "part-latest.parquet"
        df.to_parquet(out_path, index=False)
        out_paths.append(out_path)
    return out_paths


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
    parser.add_argument("--require-live-model", action="store_true",
                        help="Fail if the requested expert is only available in stub mode")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Only run experts that do not require GPU (e.g. alerce_lc)")
    parser.add_argument("--gpu-only", action="store_true",
                        help="Only run experts that require GPU (e.g. supernnova, parsnip)")
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

    # Select expert pool based on CPU/GPU flags
    if args.cpu_only:
        pool = CPU_LOCAL_EXPERTS
    elif args.gpu_only:
        pool = GPU_LOCAL_EXPERTS
    else:
        pool = ALL_LOCAL_EXPERTS

    # Instantiate requested experts
    experts = [
        cls() for cls in pool
        if args.expert == "all" or cls.name == args.expert
    ]
    if not experts:
        print(f"No expert found for '{args.expert}'")
        sys.exit(1)

    for expert in experts:
        meta = expert.metadata()
        model_loaded = bool(meta.get("model_loaded"))
        inference_implemented = bool(meta.get("inference_implemented", True))
        print(f"[{expert.name}] available={meta['available']} model_loaded={model_loaded}")
        if args.require_live_model and not inference_implemented:
            print(
                f"[{expert.name}] live inference is not implemented in this repository yet. "
                "Do not run the GPU training stage until the wrapper is completed."
            )
            sys.exit(2)
        if args.require_live_model and (not meta.get("available") or not model_loaded):
            classifier_path = meta.get("classifier_path")
            model_path = meta.get("model_path") or meta.get("model_dir")
            print(
                f"[{expert.name}] live model unavailable. "
                f"Install the package, place model weights at {model_path}"
                + (f", classifier at {classifier_path}" if classifier_path else "")
                + ", "
                "and re-run."
            )
            sys.exit(2)

    # Per-(object, n_det) inference
    results = []
    for idx, oid in enumerate(object_ids, start=1):
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

        if idx % 100 == 0:
            print(f"  Processed {idx}/{len(object_ids)} objects...")

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

    combined, appended, refreshed = _merge_records(existing, results)

    with open(out_path, "w") as fh:
        json.dump(combined, fh)
    parquet_paths = _write_parquet_partitions(combined, silver_dir)
    print(
        f"\nWrote {appended} new records and refreshed {refreshed} existing records "
        f"({len(combined)} total) → {out_path}"
    )
    for parquet_path in parquet_paths:
        print(f"Parquet partition updated → {parquet_path}")
    print("Next: python scripts/build_epoch_table_from_lc.py")


if __name__ == "__main__":
    main()
