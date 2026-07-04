#!/usr/bin/env python
"""Export frozen v9 sequence features and join them onto a fusion snapshot.

For every (object_id, n_det) row of the given snapshot, runs the pretrained
sequence encoder (models/seq_encoder_v9) over the lightcurve prefix truncated
to exactly n_det detections (the gold builder's truncation, re-asserted via
alert_jd agreement) and emits:

    seq_emb_00..seq_emb_15   16-dim frozen GRU prefix embedding
    seq_surprisal            NLL of detection n under the forecast at n-1 (NaN at n=1)
    seq_nll_mean             mean surprisal over the prefix (NaN at n=1)

NaN everywhere a lightcurve is missing (e.g. most DP1 objects locally) — the
Stage-B LightGBM treats absence natively, and the component gate decides
whether the columns earn their place.

Usage:
    python scripts/build_seq_features.py \
        --snapshots data/gold/object_epoch_snapshots_fusion_v8_trust.parquet \
        --out data/gold/seq_features_v9.parquet \
        --join-onto data/gold/object_epoch_snapshots_fusion_v8_trust.parquet \
        --joined-out data/gold/object_epoch_snapshots_fusion_v9_trust.parquet
    python scripts/build_seq_features.py --dp1 \
        --snapshots data/gold/dp1_snapshots_fusion_v8.parquet \
        --joined-out data/gold/dp1_snapshots_fusion_v9.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from debass_meta.features.detection import normalize_lightcurve  # noqa: E402
from debass_meta.features.sequence_dataset import (  # noqa: E402
    sequence_arrays,
    truncated_positive_detections,
)
from debass_meta.ingest.gold import (  # noqa: E402
    _load_lightcurve,
    _resolve_lightcurve_path,
    _to_jd,
)
from debass_meta.models.seq_encoder import (  # noqa: E402
    SEQ_EMB_DIM,
    encode_prefixes,
    load_encoder,
    resolve_device,
)

SEQ_FEATURE_NAMES = [f"seq_emb_{i:02d}" for i in range(SEQ_EMB_DIM)] + [
    "seq_surprisal",
    "seq_nll_mean",
]


def _object_detections(
    object_id: str,
    *,
    lc_dir: Path,
    dp1: bool,
) -> list[dict] | None:
    if dp1:
        p = lc_dir / f"{object_id}.parquet"
        if not p.exists():
            return None
        lc = pd.read_parquet(p)
        if "midpointMjdTai" in lc.columns:
            lc = lc.sort_values("midpointMjdTai")
        records = lc.to_dict("records")
        return normalize_lightcurve(records, survey="LSST") if records else None
    lc_path, _src = _resolve_lightcurve_path(lc_dir, object_id=object_id, associations=None)
    if lc_path is None:
        return None
    return _load_lightcurve(lc_path) or None


def build_features(
    snapshot: pd.DataFrame,
    *,
    lc_dir: Path,
    encoder_dir: Path,
    dp1: bool,
    device_req: str,
    max_n_det: int,
) -> pd.DataFrame:
    device = resolve_device(device_req)
    encoder, stats = load_encoder(encoder_dir, device=device)
    keys = snapshot[["object_id", "n_det", "alert_jd"]].copy()
    keys["object_id"] = keys["object_id"].astype(str)
    keys["n_det"] = keys["n_det"].astype(int)

    out = {name: np.full(len(keys), np.nan, dtype=np.float32) for name in SEQ_FEATURE_NAMES}
    grouped = keys.groupby("object_id", sort=False).indices
    n_objects = len(grouped)
    n_with_lc = 0
    n_rows_filled = 0
    n_mismatch = 0
    t0 = time.time()

    for j, (oid, row_idx) in enumerate(grouped.items()):
        detections = _object_detections(oid, lc_dir=lc_dir, dp1=dp1)
        if not detections:
            continue
        truncated_full = truncated_positive_detections(detections, max_len=max_n_det)
        if len(truncated_full) == 0:
            continue
        n_with_lc += 1
        cont, bands = sequence_arrays(truncated_full, pre_truncated=True)
        enc = encode_prefixes(encoder, stats.apply(cont), bands, device=device)
        for ri in row_idx:
            n_det = int(keys["n_det"].iloc[ri])
            if n_det < 1 or n_det > len(truncated_full):
                continue
            expected_jd = _to_jd(float(truncated_full[n_det - 1].get("mjd") or 0))
            row_jd = float(keys["alert_jd"].iloc[ri])
            if np.isfinite(row_jd) and abs(expected_jd - row_jd) > 1e-6:
                n_mismatch += 1
                continue
            k = n_det - 1
            for i in range(SEQ_EMB_DIM):
                out[f"seq_emb_{i:02d}"][ri] = enc["emb"][k, i]
            out["seq_surprisal"][ri] = enc["surprisal"][k]
            out["seq_nll_mean"][ri] = enc["nll_mean"][k]
            n_rows_filled += 1
        if (j + 1) % 1000 == 0:
            print(f"  {j + 1:,}/{n_objects:,} objects ({n_rows_filled:,} rows filled, "
                  f"{time.time() - t0:.0f}s)", flush=True)

    print(f"  done: {n_with_lc:,}/{n_objects:,} objects with lightcurves, "
          f"{n_rows_filled:,}/{len(keys):,} rows filled, "
          f"{n_mismatch:,} alert_jd mismatches left NaN", flush=True)
    feats = pd.DataFrame(out)
    feats.insert(0, "n_det", keys["n_det"].to_numpy())
    feats.insert(0, "object_id", keys["object_id"].to_numpy())
    return feats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshots", required=True, help="Snapshot parquet providing (object_id, n_det, alert_jd) keys")
    ap.add_argument("--encoder", default="models/seq_encoder_v9")
    ap.add_argument("--lc-dir", default=None, help="Default: data/lightcurves (or data/lightcurves/dp1 with --dp1)")
    ap.add_argument("--dp1", action="store_true", help="DP1 mode: parquet lightcurves, LSST normalization")
    ap.add_argument("--out", default=None, help="Write the bare (object_id, n_det, seq_*) feature table here")
    ap.add_argument("--join-onto", default=None, help="Snapshot to left-join the features onto (default: --snapshots)")
    ap.add_argument("--joined-out", default=None, help="Write the joined snapshot here")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    lc_dir = Path(args.lc_dir) if args.lc_dir else (Path("data/lightcurves/dp1") if args.dp1 else Path("data/lightcurves"))
    snapshot = pd.read_parquet(args.snapshots)
    print(f"Loaded {len(snapshot):,} snapshot rows from {args.snapshots}", flush=True)
    max_n_det = int(snapshot["n_det"].max()) if len(snapshot) else 20

    feats = build_features(
        snapshot,
        lc_dir=lc_dir,
        encoder_dir=Path(args.encoder),
        dp1=args.dp1,
        device_req=args.device,
        max_n_det=max_n_det,
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(args.out, index=False)
        print(f"Wrote seq features → {args.out} ({len(feats):,} rows)", flush=True)

    if args.joined_out:
        target_path = args.join_onto or args.snapshots
        target = pd.read_parquet(target_path)
        n_in = len(target)
        drop = [c for c in SEQ_FEATURE_NAMES if c in target.columns]
        if drop:
            target = target.drop(columns=drop)
        # Positional attach: feats rows were built 1:1 from --snapshots rows.
        if target_path == args.snapshots and len(target) == len(feats):
            for name in SEQ_FEATURE_NAMES:
                target[name] = feats[name].to_numpy()
        else:
            key_cols = ["object_id", "n_det"]
            f = feats.copy()
            f["object_id"] = f["object_id"].astype(str)
            t_keys = target["object_id"].astype(str)
            target = target.assign(_oid=t_keys, _nd=target["n_det"].astype(int))
            f = f.rename(columns={"object_id": "_oid", "n_det": "_nd"})
            assert not f.duplicated(["_oid", "_nd"]).any(), "duplicate (object_id, n_det) in seq features"
            target = target.merge(f, on=["_oid", "_nd"], how="left", validate="many_to_one")
            target = target.drop(columns=["_oid", "_nd"])
        assert len(target) == n_in, "join changed row count"
        Path(args.joined_out).parent.mkdir(parents=True, exist_ok=True)
        target.to_parquet(args.joined_out, index=False)
        cov = float(np.isfinite(target["seq_emb_00"]).mean())
        print(f"Wrote joined snapshot → {args.joined_out} ({len(target):,} rows, "
              f"seq coverage {cov:.1%})", flush=True)


if __name__ == "__main__":
    main()
