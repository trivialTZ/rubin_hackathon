"""v9 sequence encoder: a small causal GRU over per-detection features.

Self-supervised objective: heteroscedastic next-detection forecasting — at
step k the model predicts a Gaussian (mean, log-variance) over the NEXT
detection's normalized Δmag.  Two frozen exports feed Stage B of the fusion
stack (as gate-tested features, never as a decision layer):

  * ``seq_emb_00..15`` — 16-dim projection of the GRU hidden state at the
    epoch's last detection (prefix state ⇒ leak-free by construction:
    a unidirectional GRU's state at step k depends on detections 1..k only);
  * ``seq_surprisal`` — NLL of detection k under the forecast made at k−1
    (NaN at n_det=1), plus ``seq_nll_mean`` over the prefix.  Lightcurves that
    evolve atypically for the alert-stream population (e.g. oscillating
    periodic variables under a transient-dominated corpus) score high.

The SSL corpus must EXCLUDE the locked cal/test objects (enforced by
scripts/train_seq_encoder.py and unit-tested) so the headline protocol stays
transduction-free.

~50k parameters; trains in minutes on MPS/CPU and scales on SCC CUDA.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from debass_meta.features.sequence_dataset import (
    N_BANDS,
    SEQ_CONTINUOUS_DIM,
    NormStats,
    load_norm_stats,
    save_norm_stats,
)

_LOG2PI = math.log(2.0 * math.pi)
SEQ_EMB_DIM = 16


@dataclass
class SeqEncoderConfig:
    cont_dim: int = SEQ_CONTINUOUS_DIM
    n_bands: int = N_BANDS
    band_dim: int = 8
    proj_dim: int = 32
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.15
    emb_dim: int = SEQ_EMB_DIM

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, payload: dict) -> "SeqEncoderConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


class SeqEncoder(nn.Module):
    def __init__(self, config: SeqEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or SeqEncoderConfig()
        c = self.config
        self.band_emb = nn.Embedding(c.n_bands, c.band_dim)
        self.in_proj = nn.Sequential(
            nn.Linear(c.cont_dim + c.band_dim, c.proj_dim),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            c.proj_dim,
            c.hidden,
            num_layers=c.layers,
            batch_first=True,
            dropout=c.dropout if c.layers > 1 else 0.0,
        )
        self.emb_head = nn.Linear(c.hidden, c.emb_dim)
        self.forecast_head = nn.Linear(c.hidden, 2)  # (mean, logvar) of next Δmag

    def forward(self, cont: torch.Tensor, bands: torch.Tensor) -> torch.Tensor:
        """cont (B, L, cont_dim) float32, bands (B, L) int64 → hidden (B, L, H).

        Padding rows are fine: callers mask downstream by length; the GRU's
        state at valid step k never sees padding (it only looks backwards).
        """
        x = torch.cat([cont, self.band_emb(bands)], dim=-1)
        out, _ = self.gru(self.in_proj(x))
        return out

    def embeddings(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.emb_head(hidden)

    def forecast(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.forecast_head(hidden)
        mean = params[..., 0]
        logvar = params[..., 1].clamp(-6.0, 4.0)
        return mean, logvar


def gaussian_nll(target: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Element-wise heteroscedastic Gaussian NLL (natural units)."""
    return 0.5 * ((target - mean) ** 2 * torch.exp(-logvar) + logvar + _LOG2PI)


def ssl_next_dmag_loss(
    encoder: SeqEncoder,
    cont: torch.Tensor,
    bands: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked mean NLL of next-detection Δmag.

    The target for step k is the NORMALIZED dmag_prev feature of step k+1
    (cont[..., k+1, 1] == z-scored mag_{k+1} − mag_k), so no extra target
    tensor is needed.  Steps k with k+1 ≥ length are masked out.
    Returns (loss, n_terms).
    """
    hidden = encoder(cont, bands)
    mean, logvar = encoder.forecast(hidden)
    target = cont[:, 1:, 1]
    nll = gaussian_nll(target, mean[:, :-1], logvar[:, :-1])
    steps = torch.arange(nll.shape[1], device=nll.device).unsqueeze(0)
    mask = (steps + 1) < lengths.unsqueeze(1)
    n = mask.sum()
    loss = (nll * mask).sum() / n.clamp(min=1)
    return loss, n


@torch.no_grad()
def encode_prefixes(
    encoder: SeqEncoder,
    cont_norm: np.ndarray,
    bands: np.ndarray,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, np.ndarray]:
    """Frozen per-prefix exports for ONE object (already normalized).

    Returns arrays of length L where index k (0-based) describes the prefix
    with n_det = k+1: ``emb`` (L, emb_dim), ``surprisal`` (L,) — NLL of
    detection k+1 under the forecast at k (NaN at k=0) — and ``nll_mean``
    (L,) — running mean surprisal over the prefix (NaN at k=0).
    """
    encoder.eval()
    c = torch.from_numpy(cont_norm[None, ...]).to(device=device, dtype=torch.float32)
    b = torch.from_numpy(bands[None, ...]).to(device=device, dtype=torch.long)
    hidden = encoder(c, b)
    emb = encoder.embeddings(hidden)[0].cpu().numpy().astype(np.float32)
    mean, logvar = encoder.forecast(hidden)
    target = c[0, 1:, 1]
    nll = gaussian_nll(target, mean[0, :-1], logvar[0, :-1]).cpu().numpy()
    L = cont_norm.shape[0]
    surprisal = np.full(L, np.nan, dtype=np.float32)
    if L > 1:
        surprisal[1:] = nll
    nll_mean = np.full(L, np.nan, dtype=np.float32)
    if L > 1:
        nll_mean[1:] = np.cumsum(nll) / (np.arange(L - 1) + 1)
    return {"emb": emb, "surprisal": surprisal, "nll_mean": nll_mean}


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        # Modern torch wheels support sm_75+ only; ancient cluster GPUs
        # (e.g. Tesla P100, sm_60) crash at kernel launch — fall back to CPU.
        cap = torch.cuda.get_device_capability(0)
        if cap >= (7, 5):
            return torch.device("cuda")
        print(f"  [resolve_device] GPU capability {cap} unsupported by this "
              f"torch build — falling back to CPU", flush=True)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_encoder(
    encoder: SeqEncoder,
    norm_stats: NormStats,
    out_dir: Path,
    *,
    extra_meta: dict | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), out_dir / "encoder.pt")
    save_norm_stats(norm_stats, out_dir / "norm_stats.json")
    meta = {"config": encoder.config.to_json()}
    if extra_meta:
        meta.update(extra_meta)
    (out_dir / "config.json").write_text(json.dumps(meta, indent=1, default=str))


def load_encoder(out_dir: Path, *, device: torch.device | str = "cpu") -> tuple[SeqEncoder, NormStats]:
    meta = json.loads((out_dir / "config.json").read_text())
    encoder = SeqEncoder(SeqEncoderConfig.from_json(meta["config"]))
    state = torch.load(out_dir / "encoder.pt", map_location="cpu", weights_only=True)
    encoder.load_state_dict(state)
    encoder.to(device)
    encoder.eval()
    return encoder, load_norm_stats(out_dir / "norm_stats.json")
