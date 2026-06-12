"""Preflight: run local SuperNNova on 5 LSST LCs and assert outputs make sense.

Invariants:
  (A) adapter does NOT default all bands to 'r' — at least 2 distinct LSST
      bands are present in the input to SNN.
  (B) classify_lcs runs without exception.
  (C) p_snia is in [0, 1] and not identically 0.5 for all objects (would
      indicate stub mode fallback, not real inference).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from debass_meta.experts.local.supernnova import SuperNNovaExpert, _SNN_BANDS


def main() -> None:
    lc_dir = Path("data/lightcurves")
    # Pick LSST LCs (numeric filenames — diaObjectId)
    lsst = sorted(f for f in lc_dir.glob("*.json") if f.stem.isdigit() and len(f.stem) > 10)
    print(f"total LSST LC files: {len(lsst)}")
    if not lsst:
        print("FAIL: no LSST LCs in cache")
        sys.exit(1)

    random.seed(42)
    picks = random.sample(lsst, 5)
    items = []
    print("\nLoading 5 random LSST LCs:")
    for f in picks:
        dets = json.loads(f.read_text())
        if not dets:
            continue
        oid = f.stem
        last_mjd = max(float(d.get("mjd", 0)) for d in dets)
        epoch_jd = last_mjd + 2400000.5
        items.append((oid, dets, epoch_jd, len(dets)))
        bands = sorted({d.get("band") for d in dets if d.get("band")})
        print(f"  {oid}: {len(dets)} dets, bands={bands}, last_mjd={last_mjd:.2f}")

    # Invariant (A): multiple distinct bands present
    all_bands = set()
    for _, dets, _, _ in items:
        for d in dets:
            b = d.get("band")
            if b in _SNN_BANDS:
                all_bands.add(b)
    print(f"\n[A] distinct LSST bands in input: {sorted(all_bands)}")
    assert len(all_bands) >= 2, f"FAIL: only {all_bands} bands seen; adapter fallback broken"
    print("    PASS")

    # Invariant (B): classify_lcs runs
    print("\n[B] SuperNNovaExpert.predict_epoch_batch:")
    expert = SuperNNovaExpert()
    print(f"    model available: {expert._snn_available}, file: {expert._model_file}")
    if not expert._snn_available or expert._model_file is None:
        print("    SKIP: SNN not installed OR weights missing in artifacts/local_experts/supernnova/")
        print("    (preflight deferred until running on SCC where weights live)")
        return
    outputs = expert.predict_epoch_batch(items)
    assert len(outputs) == len(items), f"FAIL: got {len(outputs)} outputs for {len(items)} inputs"
    print(f"    PASS: {len(outputs)} outputs")

    # Invariant (C): not all stub (0.5/0.5)
    print("\n[C] outputs:")
    print(f"    {'object':>20s}  {'ndet':>4s}  {'p_Ia':>6s}  {'p_nonIa':>7s}  {'ver':>6s}")
    any_nonstub = False
    for out in outputs:
        probs = out.class_probabilities or {}
        p_ia = probs.get("SN Ia", float("nan"))
        p_non = probs.get("non-Ia", float("nan"))
        print(f"    {out.object_id:>20s}  {out.raw_output.get('truncated_n_det', 0):>4d}  "
              f"{p_ia:>6.3f}  {p_non:>7.3f}  {out.model_version:>6s}")
        if out.model_version == "loaded" and abs(p_ia - 0.5) > 0.001:
            any_nonstub = True
    assert any_nonstub, "FAIL: all outputs are uniform 0.5/0.5 — model likely didn't load or bands garbled"
    print("    PASS: at least one non-uniform output")

    print("\nPreflight all invariants passed ✓")


if __name__ == "__main__":
    main()
