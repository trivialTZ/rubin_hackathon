#!/usr/bin/env python
"""Self-supervised pretraining of the v9/v10 sequence encoder.

Corpus = every cached lightcurve EXCEPT objects in the locked cal/test splits
(read from the fusion split manifest) — the transduction-hygiene rule from the
design review: the encoder must never have seen a test/cal object's photometry,
so headline metrics measure deployment behavior, not memorization.  v10 also
excludes ASSOCIATION COUNTERPARTS of cal/test objects (LSST diaObjectId ↔ ZTF
oid share photometry — the duplicate-photometry leak) and can fold DP1 parquet
lightcurves into the corpus (``--dp1-lc-dir``) so the encoder sees real LSST
cadence unsupervised.

Objective: heteroscedastic next-detection Δmag forecasting (Gaussian NLL),
masked over real steps.  Early stopping on a held-out 10%-of-corpus object
split.  ``--surveys {ztf,lsst,both}`` filters the corpus for staged
ZTF-pretrain → LSST-transfer recipes; ``--per-survey-norm`` fits survey-keyed
z-score statistics (pooled fallback below 200 sequences per survey).
Artifact: <out>/{encoder.pt, norm_stats.json, config.json, train_log.json}.

Local (MPS/CPU) ~minutes; SCC CUDA via jobs/run_fusion_v10_pretrain.sh.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from debass_meta.access.associations import load_lsst_ztf_associations  # noqa: E402
from debass_meta.features.sequence_dataset import (  # noqa: E402
    NormStats,
    load_object_sequence,
    sequence_arrays,
    sequence_survey,
)
from debass_meta.models.seq_encoder import (  # noqa: E402
    SeqEncoder,
    SeqEncoderConfig,
    resolve_device,
    save_encoder,
    ssl_next_dmag_loss,
)


def load_split_exclusions(
    split_path: Path,
    association_csv: Path | None = None,
) -> set[str]:
    """cal ∪ test ∪ quarantined object IDs from the fusion split manifest
    (or v6e2 trust metadata — both carry cal_ids/test_ids), plus (v10) their
    association counterparts: an LSST diaObjectId in cal/test excludes its
    ZTF oid from the corpus and vice versa — the same physical photometry
    cached under two IDs must never straddle the exclusion boundary.
    ``quarantined_ids`` (v10 manifests) are locked-test counterparts."""
    payload = json.loads(split_path.read_text())
    if "split" in payload and isinstance(payload["split"], dict):
        payload = payload["split"]
    excl: set[str] = set()
    for key in ("cal_ids", "test_ids", "cal", "test", "quarantined_ids"):
        excl.update(str(o) for o in payload.get(key, []) or [])
    if not excl:
        raise SystemExit(f"No cal/test IDs found in {split_path} — refusing to pretrain without exclusions")
    if association_csv is not None and Path(association_csv).exists():
        associations = load_lsst_ztf_associations(Path(association_csv))
        counterparts: set[str] = set()
        for lsst_id, entry in associations.items():
            ztf_id = str(entry.get("ztf_object_id") or "")
            if not ztf_id:
                continue
            if lsst_id in excl and ztf_id not in excl:
                counterparts.add(ztf_id)
            if ztf_id in excl and lsst_id not in excl:
                counterparts.add(lsst_id)
        if counterparts:
            print(f"  excluding {len(counterparts):,} association counterparts "
                  f"of cal/test objects (duplicate-photometry defense)", flush=True)
        excl |= counterparts
    elif association_csv is not None:
        print(f"  note: association CSV {association_csv} not found — "
              f"no counterpart exclusions applied", flush=True)
    return excl


def _load_dp1_sequence(
    path: Path,
    *,
    max_len: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """One DP1 parquet lightcurve → sequence arrays (LSST normalization),
    replicating scripts/build_seq_features.py --dp1."""
    import pandas as pd

    from debass_meta.features.detection import normalize_lightcurve

    lc = pd.read_parquet(path)
    if "midpointMjdTai" in lc.columns:
        lc = lc.sort_values("midpointMjdTai")
    records = lc.to_dict("records")
    if not records:
        return None
    detections = normalize_lightcurve(records, survey="LSST")
    cont, bands = sequence_arrays(detections, max_len=max_len)
    if len(cont) == 0:
        return None
    return cont, bands


def build_corpus(
    lc_dir: Path,
    exclude: set[str],
    *,
    max_len: int,
    limit: int | None = None,
    surveys: str = "both",
    dp1_dir: Path | None = None,
) -> tuple[list[str], list[tuple[np.ndarray, np.ndarray]]]:
    stems = sorted(p.stem for p in lc_dir.glob("*.json"))
    kept_ids: list[str] = []
    seqs: list[tuple[np.ndarray, np.ndarray]] = []
    n_excluded = 0
    n_survey_filtered = 0
    for i, stem in enumerate(stems):
        if limit is not None and len(kept_ids) >= limit:
            break
        if stem in exclude:
            n_excluded += 1
            continue
        loaded = load_object_sequence(lc_dir, stem, max_len=max_len)
        if loaded is None or len(loaded[0]) < 2:  # need >=2 dets for a forecast target
            continue
        if surveys != "both" and sequence_survey(loaded[0]) != surveys:
            n_survey_filtered += 1
            continue
        kept_ids.append(stem)
        seqs.append(loaded)
        if (i + 1) % 2000 == 0:
            print(f"  corpus: scanned {i + 1:,}/{len(stems):,} stems, kept {len(kept_ids):,}", flush=True)
    n_dp1 = 0
    if dp1_dir is not None and surveys != "ztf" and Path(dp1_dir).exists():
        known = set(kept_ids)
        for p in sorted(Path(dp1_dir).glob("*.parquet")):
            if limit is not None and len(kept_ids) >= limit:
                break
            stem = p.stem
            if stem in exclude:
                n_excluded += 1
                continue
            if stem in known:
                continue
            loaded = _load_dp1_sequence(p, max_len=max_len)
            if loaded is None or len(loaded[0]) < 2:
                continue
            kept_ids.append(stem)
            seqs.append(loaded)
            n_dp1 += 1
        if n_dp1:
            print(f"  corpus: +{n_dp1:,} DP1 parquet lightcurves from {dp1_dir}", flush=True)
    print(f"  corpus: {len(kept_ids):,} objects kept ({n_dp1:,} DP1), "
          f"{n_excluded:,} excluded (cal/test+counterparts), "
          f"{n_survey_filtered:,} filtered (--surveys {surveys})", flush=True)
    return kept_ids, seqs


def pad_batch(
    seqs: list[tuple[np.ndarray, np.ndarray]],
    idxs: np.ndarray,
    stats: NormStats,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    chunk = [seqs[i] for i in idxs]
    lengths = [len(c) for c, _ in chunk]
    L = max(lengths)
    cont = np.zeros((len(chunk), L, chunk[0][0].shape[1]), dtype=np.float32)
    bands = np.zeros((len(chunk), L), dtype=np.int64)
    for j, (c, b) in enumerate(chunk):
        cont[j, : len(c)] = stats.apply(c)
        bands[j, : len(b)] = b
    return (
        torch.from_numpy(cont).to(device),
        torch.from_numpy(bands).to(device),
        torch.tensor(lengths, dtype=torch.long, device=device),
    )


def evaluate(encoder, seqs, idxs, stats, device, batch: int) -> float:
    encoder.eval()
    tot, n_tot = 0.0, 0
    with torch.no_grad():
        for s in range(0, len(idxs), batch):
            cont, bands, lengths = pad_batch(seqs, idxs[s : s + batch], stats, device)
            loss, n = ssl_next_dmag_loss(encoder, cont, bands, lengths)
            tot += float(loss) * int(n)
            n_tot += int(n)
    return tot / max(n_tot, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lc-dir", default="data/lightcurves")
    ap.add_argument("--split", default="data/gold/split_fusion_v8.json",
                    help="Split manifest whose cal+test objects are EXCLUDED from the SSL corpus")
    ap.add_argument("--association-csv", default="data/crossmatch/lsst_to_ztf.csv",
                    help="LSST↔ZTF association CSV: counterparts of cal/test objects are "
                         "ALSO excluded (duplicate photometry). Missing file → no-op.")
    ap.add_argument("--surveys", choices=("ztf", "lsst", "both"), default="both",
                    help="Restrict the SSL corpus to one survey (staged ZTF→LSST recipes)")
    ap.add_argument("--per-survey-norm", action="store_true",
                    help="Fit survey-keyed NormStats (pooled fallback below "
                         "200 sequences per survey)")
    ap.add_argument("--dp1-lc-dir", default=None,
                    help="Optional DP1 parquet lightcurve dir (e.g. data/lightcurves/dp1) "
                         "folded into the SSL corpus with cal/test exclusions applied")
    ap.add_argument("--out", default="models/seq_encoder_v9")
    ap.add_argument("--max-len", type=int, default=60, help="SSL uses longer prefixes than gold's 20")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="Corpus cap (smoke)")
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    print(f"seq_encoder SSL pretraining — device={device}", flush=True)

    exclude = load_split_exclusions(
        Path(args.split),
        association_csv=Path(args.association_csv) if args.association_csv else None,
    )
    print(f"  excluding {len(exclude):,} cal/test(+counterpart) objects from the corpus", flush=True)
    ids, seqs = build_corpus(
        Path(args.lc_dir), exclude, max_len=args.max_len, limit=args.limit,
        surveys=args.surveys,
        dp1_dir=Path(args.dp1_lc_dir) if args.dp1_lc_dir else None,
    )
    if len(ids) < 50:
        raise SystemExit(f"Corpus too small ({len(ids)}) — wrong --lc-dir?")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(ids))
    n_val = max(1, int(len(ids) * args.val_frac))
    val_idx, train_idx = order[:n_val], order[n_val:]
    stats = NormStats.fit([seqs[i][0] for i in train_idx], per_survey=args.per_survey_norm)
    print(f"  train {len(train_idx):,} / val {len(val_idx):,} objects; "
          f"norm mean={['%.3f' % m for m in stats.mean[:5]]}; "
          f"per-survey stats: {sorted(stats.per_survey) or 'pooled only'}", flush=True)

    encoder = SeqEncoder(SeqEncoderConfig()).to(device)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  encoder params: {n_params:,}", flush=True)
    opt = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)

    best_val, best_state, bad, log = float("inf"), None, 0, []
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        ep_order = rng.permutation(train_idx)
        tot, n_tot = 0.0, 0
        for s in range(0, len(ep_order), args.batch):
            cont, bands, lengths = pad_batch(seqs, ep_order[s : s + args.batch], stats, device)
            opt.zero_grad()
            loss, n = ssl_next_dmag_loss(encoder, cont, bands, lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            opt.step()
            tot += float(loss) * int(n)
            n_tot += int(n)
        train_nll = tot / max(n_tot, 1)
        val_nll = evaluate(encoder, seqs, val_idx, stats, device, args.batch)
        sched.step(val_nll)
        log.append({"epoch": epoch, "train_nll": train_nll, "val_nll": val_nll,
                    "lr": opt.param_groups[0]["lr"]})
        marker = ""
        if val_nll < best_val - 1e-4:
            best_val, bad = val_nll, 0
            best_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
            marker = "  *best*"
        else:
            bad += 1
        print(f"  epoch {epoch:3d}  train {train_nll:.4f}  val {val_nll:.4f}{marker}", flush=True)
        if bad >= args.patience:
            print(f"  early stop at epoch {epoch} (patience {args.patience})", flush=True)
            break

    if best_state is not None:
        encoder.load_state_dict(best_state)
    encoder.to("cpu")
    out_dir = Path(args.out)
    save_encoder(encoder, stats, out_dir, extra_meta={
        "n_params": n_params,
        "corpus_objects": len(ids),
        "n_excluded_cal_test": len(exclude),
        "split_manifest": str(args.split),
        "association_csv": str(args.association_csv) if args.association_csv else None,
        "surveys": args.surveys,
        "per_survey_norm": bool(args.per_survey_norm),
        "per_survey_stats_fitted": sorted(stats.per_survey),
        "dp1_lc_dir": str(args.dp1_lc_dir) if args.dp1_lc_dir else None,
        "best_val_nll": best_val,
        "max_len": args.max_len,
        "seed": args.seed,
        "device_trained": str(device),
    })
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=1))
    print(f"Saved encoder → {out_dir} (best val NLL {best_val:.4f}, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
