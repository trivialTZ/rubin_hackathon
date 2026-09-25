#!/usr/bin/env python3
"""Self-supervised NEXT-DETECTION pretraining of the v9/v11 sequence encoder.

Objective (causal, natural for a unidirectional GRU): from the hidden state at
token ``k`` predict token ``k+1`` — its delta-t and normalized brightness/flux
change (Huber regression) AND its photometric band (cross-entropy).  A
unidirectional GRU's state at step k depends only on detections 1..k, so the
forecast target k+1 is a genuine hold-out at every step and the no-leakage
contract that the classifier inherits is preserved.

This complements scripts/train_seq_encoder.py (which pretrains the same
:class:`~debass_meta.models.seq_encoder.SeqEncoder` backbone with a
heteroscedastic Δmag Gaussian-NLL objective on the LABELED corpus).  Here the
corpus is the UNLABELED LSST harvest (scripts/harvest_unlabeled_lsst.py) — the
encoder finally sees LSST cadence/depth at scale.

Artifact compatibility is the deliverable: we write EXACTLY the
{encoder.pt, norm_stats.json, config.json} that
:func:`debass_meta.models.seq_encoder.load_encoder` reads, so
``scripts/train_seq_classifier.py --encoder <this>`` warm-starts from it with
no changes (fit_classifier() calls load_encoder() when encoder.pt exists).  The
SSL prediction heads live OUTSIDE the SeqEncoder (saved separately as
ssl_heads.pt) so encoder.pt stays a strict SeqEncoder state_dict.  main()
asserts the artifact round-trips through load_encoder + SeqClassifier before
declaring success.

CLI: --lc-dir, --out models/seq_encoder_ssl_v1, --epochs, --batch-size,
--device (cpu|mps|cuda auto).  Local (MPS/CPU) minutes; SCC CUDA via
jobs/run_seq_ssl_pretrain.sh.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from debass_meta.features.sequence_dataset import (  # noqa: E402
    N_BANDS,
    _SIGNED_FLUX_DIM,
    NormStats,
    _cont_dim_for_schema,
    load_object_sequence,
    sequence_survey,
)
from debass_meta.models.seq_encoder import (  # noqa: E402
    SeqEncoder,
    SeqEncoderConfig,
    load_encoder,
    resolve_device,
    save_encoder,
)

# Continuous next-detection regression targets, as indices into the (normalized)
# continuous feature tensor: log_dt_prev (delta-t) and dmag_prev (the z-scored
# normalized brightness/flux change — the "flux" the encoding carries in v9).
# v11 additionally regresses the signed diff-flux channel.
_DELTA_T_DIM = 0
_DMAG_DIM = 1


def continuous_target_dims(schema: str) -> tuple[int, ...]:
    if schema == "v11":
        return (_DELTA_T_DIM, _DMAG_DIM, _SIGNED_FLUX_DIM)
    return (_DELTA_T_DIM, _DMAG_DIM)


class NextDetHeads(nn.Module):
    """Two SSL prediction heads over the GRU hidden state.

    Kept separate from :class:`SeqEncoder` on purpose: only the encoder is
    persisted (encoder.pt), so the downstream classifier's strict
    ``load_state_dict`` never sees these extra parameters.
    """

    def __init__(self, hidden: int, n_cont: int, n_bands: int = N_BANDS) -> None:
        super().__init__()
        self.reg = nn.Linear(hidden, n_cont)      # (delta-t, flux[, signed_flux])
        self.band = nn.Linear(hidden, n_bands)    # next-band logits

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.reg(hidden), self.band(hidden)


def ssl_next_detection_loss(
    encoder: SeqEncoder,
    heads: NextDetHeads,
    cont: torch.Tensor,
    bands: torch.Tensor,
    lengths: torch.Tensor,
    target_dims: tuple[int, ...],
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    """Masked next-detection loss (Huber on continuous targets + CE on band).

    At step k the heads predict token k+1: the regression target is the
    NORMALIZED continuous channels of k+1 (already z-scored by NormStats.apply
    upstream) and the band target is band[k+1].  Steps k with k+1 >= length are
    masked out.  Returns (total_loss, {"reg", "band"} per-head floats, n_terms).
    """
    hidden = encoder(cont, bands)                     # (B, L, H)
    reg_pred, band_logits = heads(hidden)
    reg_pred = reg_pred[:, :-1]                        # (B, L-1, n_cont)
    band_logits = band_logits[:, :-1]                  # (B, L-1, n_bands)
    reg_target = cont[:, 1:][..., list(target_dims)]   # (B, L-1, n_cont)
    band_target = bands[:, 1:]                          # (B, L-1)

    steps = torch.arange(reg_pred.shape[1], device=cont.device).unsqueeze(0)
    mask = (steps + 1) < lengths.unsqueeze(1)          # (B, L-1) valid targets
    n = mask.sum().clamp(min=1)

    huber = F.huber_loss(reg_pred, reg_target, reduction="none").mean(dim=-1)  # (B, L-1)
    reg_loss = (huber * mask).sum() / n
    ce = F.cross_entropy(
        band_logits.reshape(-1, band_logits.shape[-1]),
        band_target.reshape(-1),
        reduction="none",
    ).reshape(band_target.shape)
    band_loss = (ce * mask).sum() / n

    total = reg_loss + band_loss
    return total, {"reg": float(reg_loss), "band": float(band_loss)}, mask.sum()


def build_corpus(
    lc_dir: Path,
    *,
    exclude: set[str],
    max_len: int,
    schema: str,
    surveys: str = "both",
    limit: int | None = None,
) -> tuple[list[str], list[tuple[np.ndarray, np.ndarray]]]:
    stems = sorted(p.stem for p in lc_dir.glob("*.json"))
    ids: list[str] = []
    seqs: list[tuple[np.ndarray, np.ndarray]] = []
    n_excluded = n_survey = 0
    for i, stem in enumerate(stems):
        if limit is not None and len(ids) >= limit:
            break
        if stem in exclude:
            n_excluded += 1
            continue
        loaded = load_object_sequence(lc_dir, stem, max_len=max_len, schema=schema)
        if loaded is None or len(loaded[0]) < 2:      # need >=2 dets for a forecast target
            continue
        if surveys != "both" and sequence_survey(loaded[0]) != surveys:
            n_survey += 1
            continue
        ids.append(stem)
        seqs.append(loaded)
        if (i + 1) % 2000 == 0:
            print(f"  corpus: scanned {i + 1:,}/{len(stems):,}, kept {len(ids):,}", flush=True)
    print(f"  corpus: {len(ids):,} kept, {n_excluded:,} benchmark-excluded, "
          f"{n_survey:,} survey-filtered", flush=True)
    return ids, seqs


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


def evaluate(encoder, heads, seqs, idxs, stats, device, batch, target_dims):
    encoder.eval()
    heads.eval()
    tot, reg_s, band_s, n_tot = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for s in range(0, len(idxs), batch):
            cont, bands, lengths = pad_batch(seqs, idxs[s : s + batch], stats, device)
            loss, parts, n = ssl_next_detection_loss(
                encoder, heads, cont, bands, lengths, target_dims)
            k = int(n)
            tot += float(loss) * k
            reg_s += parts["reg"] * k
            band_s += parts["band"] * k
            n_tot += k
    d = max(n_tot, 1)
    return tot / d, reg_s / d, band_s / d


def assert_encoder_roundtrip(encoder: SeqEncoder, stats: NormStats, out_dir: Path) -> None:
    """Load the just-saved artifact back through the SAME path the classifier
    trainer uses and assert byte-for-byte encoder equality — compatibility is
    the deliverable, so we verify it, not assume it."""
    from debass_meta.models.seq_classifier import SeqClassifier

    enc2, _stats2 = load_encoder(out_dir)
    assert enc2.config.cont_dim == encoder.config.cont_dim, "cont_dim mismatch on reload"
    ref = encoder.state_dict()
    got = enc2.state_dict()
    assert set(ref) == set(got), "encoder state_dict keys changed on reload"
    for k in ref:
        assert torch.equal(ref[k].cpu(), got[k].cpu()), f"encoder param {k} changed on reload"
    # The real consumer: the classifier wraps this encoder.
    SeqClassifier(enc2)
    print("  round-trip OK: load_encoder + SeqClassifier accept the artifact", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lc-dir", default="data/pretrain_lsst/lightcurves",
                    help="Unlabeled lightcurve dir (scripts/harvest_unlabeled_lsst.py output)")
    ap.add_argument("--out", default="models/seq_encoder_ssl_v1")
    ap.add_argument("--seq-schema", choices=("v9", "v11"), default="v11",
                    help="Tokenization schema; sizes the encoder cont_dim and is "
                         "stamped into the artifact meta (default v11 — LSST negatives).")
    ap.add_argument("--surveys", choices=("ztf", "lsst", "both"), default="both")
    ap.add_argument("--benchmark-manifest", default="data/gold/lsst_live_locked_test.json",
                    help="Safety net: any test_ids present in --lc-dir are dropped from "
                         "the corpus (the harvester already excludes them at fetch time).")
    ap.add_argument("--per-survey-norm", action="store_true", default=True,
                    help="Fit survey-keyed NormStats (pooled fallback below 200 seqs/survey)")
    ap.add_argument("--pooled-norm", dest="per_survey_norm", action="store_false",
                    help="Force pooled NormStats")
    ap.add_argument("--max-len", type=int, default=60, help="SSL uses longer prefixes than gold's 20")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="Corpus cap (smoke)")
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    print(f"seq_encoder SSL next-detection pretraining — device={device}, "
          f"schema={args.seq_schema}", flush=True)

    from scripts.harvest_unlabeled_lsst import load_benchmark_exclusions

    exclude = load_benchmark_exclusions(args.benchmark_manifest)
    print(f"  benchmark exclusions: {len(exclude):,} ids", flush=True)

    ids, seqs = build_corpus(
        Path(args.lc_dir), exclude=exclude, max_len=args.max_len,
        schema=args.seq_schema, surveys=args.surveys, limit=args.limit)
    if len(ids) < 20:
        raise SystemExit(f"Corpus too small ({len(ids)}) — wrong --lc-dir? "
                         f"(run scripts/harvest_unlabeled_lsst.py first)")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(ids))
    n_val = max(1, int(len(ids) * args.val_frac))
    val_idx, train_idx = order[:n_val], order[n_val:]
    stats = NormStats.fit([seqs[i][0] for i in train_idx], per_survey=args.per_survey_norm)
    print(f"  train {len(train_idx):,} / val {len(val_idx):,}; "
          f"per-survey stats: {sorted(stats.per_survey) or 'pooled only'}", flush=True)

    target_dims = continuous_target_dims(args.seq_schema)
    cont_dim = _cont_dim_for_schema(args.seq_schema)
    encoder = SeqEncoder(SeqEncoderConfig(cont_dim=cont_dim)).to(device)
    heads = NextDetHeads(encoder.config.hidden, len(target_dims)).to(device)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  encoder params: {n_params:,}; regression targets: {target_dims}", flush=True)
    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(heads.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)

    init_val = evaluate(encoder, heads, seqs, val_idx, stats, device, args.batch_size, target_dims)
    print(f"  initial val: total {init_val[0]:.4f}  reg {init_val[1]:.4f}  band {init_val[2]:.4f}",
          flush=True)

    best_val, best_state, bad, log = float("inf"), None, 0, []
    log.append({"epoch": 0, "val_total": init_val[0], "val_reg": init_val[1],
                "val_band": init_val[2], "phase": "init"})
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        heads.train()
        ep_order = rng.permutation(train_idx)
        tot, reg_s, band_s, n_tot = 0.0, 0.0, 0.0, 0
        for s in range(0, len(ep_order), args.batch_size):
            cont, bands, lengths = pad_batch(seqs, ep_order[s : s + args.batch_size], stats, device)
            opt.zero_grad()
            loss, parts, n = ssl_next_detection_loss(
                encoder, heads, cont, bands, lengths, target_dims)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(heads.parameters()), 5.0)
            opt.step()
            k = int(n)
            tot += float(loss) * k
            reg_s += parts["reg"] * k
            band_s += parts["band"] * k
            n_tot += k
        d = max(n_tot, 1)
        tr = (tot / d, reg_s / d, band_s / d)
        val = evaluate(encoder, heads, seqs, val_idx, stats, device, args.batch_size, target_dims)
        sched.step(val[0])
        log.append({"epoch": epoch, "train_total": tr[0], "train_reg": tr[1], "train_band": tr[2],
                    "val_total": val[0], "val_reg": val[1], "val_band": val[2],
                    "lr": opt.param_groups[0]["lr"]})
        marker = ""
        if val[0] < best_val - 1e-4:
            best_val, bad = val[0], 0
            best_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
            marker = "  *best*"
        else:
            bad += 1
        print(f"  epoch {epoch:3d}  train {tr[0]:.4f} (reg {tr[1]:.4f} band {tr[2]:.4f})  "
              f"val {val[0]:.4f} (reg {val[1]:.4f} band {val[2]:.4f}){marker}", flush=True)
        if bad >= args.patience:
            print(f"  early stop at epoch {epoch} (patience {args.patience})", flush=True)
            break

    if best_state is not None:
        encoder.load_state_dict(best_state)
    encoder.to("cpu")
    heads.to("cpu")
    out_dir = Path(args.out)
    save_encoder(encoder, stats, out_dir, extra_meta={
        "pretrain_objective": "next_detection (huber dt+flux, ce band)",
        "n_params": n_params,
        "corpus_objects": len(ids),
        "surveys": args.surveys,
        "seq_schema": args.seq_schema,
        "cont_dim": cont_dim,
        "regression_target_dims": list(target_dims),
        "per_survey_norm": bool(args.per_survey_norm),
        "per_survey_stats_fitted": sorted(stats.per_survey),
        "init_val_total": init_val[0],
        "best_val_total": best_val,
        "max_len": args.max_len,
        "seed": args.seed,
        "device_trained": str(device),
        "lc_dir": str(args.lc_dir),
    })
    # SSL heads saved beside the encoder for reproducibility / continued
    # pretraining — NOT part of encoder.pt (which stays a strict SeqEncoder dict).
    torch.save(heads.state_dict(), out_dir / "ssl_heads.pt")
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=1))

    assert_encoder_roundtrip(encoder, stats, out_dir)
    print(f"Saved SSL encoder -> {out_dir} (init val {init_val[0]:.4f} -> best {best_val:.4f}, "
          f"{time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
