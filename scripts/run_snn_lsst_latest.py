"""Run local SuperNNova on every cached LSST LC at its latest epoch.

Produces `data/silver/local_expert_outputs/supernnova/part-lsst-latest.parquet`
so the downstream cross-check script can compare local vs fink_lsst/snn and
pittgoogle/supernnova_lsst on the same objects.

This script emits ONE row per LSST object (latest epoch only). It's a
methodology-validation artifact, NOT a full per-epoch silver feed — use
scripts/collect_epoch_history.py for that once this preflight lands.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from debass_meta.experts.local.supernnova import SuperNNovaExpert


def is_lsst_filename(p: Path) -> bool:
    # ALeRCE LSST diaObjectIds are long numeric strings; ZTF is "ZTF…"
    return p.stem.isdigit() and len(p.stem) >= 15


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lc-dir", default="data/lightcurves")
    ap.add_argument(
        "--output",
        default="data/silver/local_expert_outputs/supernnova/part-lsst-latest.parquet",
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    lc_files = sorted(f for f in Path(args.lc_dir).glob("*.json") if is_lsst_filename(f))
    if args.limit:
        lc_files = lc_files[: args.limit]
    print(f"LSST LC files to score: {len(lc_files)}")

    expert = SuperNNovaExpert()
    print(
        f"  SNN available: {expert._snn_available}, "
        f"model file: {expert._model_file}, device: {expert._device}"
    )
    if not expert._snn_available or expert._model_file is None:
        print("ABORT: SNN not installed or weights missing.")
        sys.exit(2)

    # Build items list
    items: list[tuple[str, list[dict], float, int]] = []
    for f in lc_files:
        try:
            dets = json.loads(f.read_text())
        except Exception:
            continue
        if not isinstance(dets, list) or not dets:
            continue
        last_mjd = max(float(d.get("mjd", 0)) for d in dets)
        epoch_jd = last_mjd + 2400000.5
        items.append((f.stem, dets, epoch_jd, len(dets)))
    print(f"  parsed {len(items)} LCs with >=1 detection")

    # Batch inference
    t0 = time.time()
    results = []
    bs = args.batch_size
    for i in range(0, len(items), bs):
        chunk = items[i : i + bs]
        outs = expert.predict_epoch_batch(chunk)
        for (oid, dets, epoch_jd, n_det), out in zip(chunk, outs):
            probs = out.class_probabilities or {}
            results.append({
                "object_id": oid,
                "expert": "supernnova",
                "n_det": n_det,
                "alert_mjd": epoch_jd - 2400000.5,
                "alert_jd": epoch_jd,
                "p_snia": probs.get("SN Ia"),
                "p_non_snia": probs.get("non-Ia"),
                "class_probabilities": json.dumps(probs),
                "model_version": out.model_version,
                "available": bool(out.model_version == "loaded"),
            })
        if (i // bs) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + len(chunk)) / max(elapsed, 1e-6)
            print(f"  scored {i + len(chunk):,}/{len(items):,} "
                  f"({rate:.1f} obj/s, {elapsed:.0f}s elapsed)")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_parquet(out, index=False)
    print(f"\nWrote {len(df)} rows → {out}")
    print(f"  non-stub outputs: {int(df['available'].sum())} / {len(df)}")
    print(f"  p_snia summary:")
    p = df["p_snia"].dropna()
    print(f"    mean={p.mean():.3f}  median={p.median():.3f}  "
          f"min={p.min():.3f}  max={p.max():.3f}  std={p.std():.3f}")
    print(f"  fraction with p_snia > 0.5: {(p > 0.5).mean():.1%}")
    print(f"  fraction with p_snia > 0.7: {(p > 0.7).mean():.1%}")
    print(f"  fraction with p_snia < 0.3: {(p < 0.3).mean():.1%}")


if __name__ == "__main__":
    main()
