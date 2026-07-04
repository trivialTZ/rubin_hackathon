"""v10 sequence classifier — the SSL encoder fine-tuned into a 4-way expert.

Takes the pretrained :class:`~debass_meta.models.seq_encoder.SeqEncoder`
backbone and adds a per-step classification head: at step k the model emits
P(snia / snii / other_sn / non_sn | detections 1..k).  One forward pass
therefore yields the prediction for EVERY epoch prefix (deep supervision over
n_det), matching the (object_id, n_det) snapshot structure and the no-leakage
contract (unidirectional GRU ⇒ step k sees only detections 1..k).

The pipeline contract stays TERNARY: (snii + other_sn) fold to nonIa_snlike
and non_sn folds to other (:func:`fold_probs_to_ternary`); the projector
(projectors/local_seq_v9.py) emits exactly (p_snia, p_nonIa_snlike, p_other).
Legacy 3-class (v9c) artifacts still load — the artifact records its own
class tuple and the head is sized from it.

Supervision is mask-based (:func:`classification_loss` with ``class_masks``):
each row is penalised only on the GROUPED probability mass of the classes its
label can actually distinguish —

  * full fine label (spec SN IIn, …)        → one-hot mass
  * spectroscopic non-Ia without subtype    → P(snii) + P(other_sn)
  * weak stamp-SN (LSST candidates etc.)    → P(snia) + P(snii) + P(other_sn)
    vs P(non_sn) ONLY — never which SN subclass, so weak labels can no longer
    teach "survey ⇒ not Ia" (the v9c structural anti-Ia bias on LSST).

Training (scripts/train_seq_classifier.py): TRAIN split only, with an inner
10% object-grouped validation carve-out for early stopping; cal is used ONLY
for temperature; headline diagnostics come from the locked TEST split.
``--oof-folds K`` additionally trains K fold models (each excluding one fold
of train objects) saved under ``fold_{k}/`` plus ``fold_map.json`` so train
rows can be scored out-of-fold (kills the self-trained-expert stacking trap).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from debass_meta.features.sequence_dataset import NormStats, load_norm_stats, save_norm_stats
from debass_meta.models.seq_encoder import SeqEncoder, SeqEncoderConfig

CLASSES = ("snia", "snii", "other_sn", "non_sn")
TERNARY_CLASSES = ("snia", "nonIa_snlike", "other")
SN_GROUP = ("snia", "snii", "other_sn")

# 4-way (and passthrough ternary) class → canonical ternary bucket.
CLASS_TO_TERNARY = {
    "snia": "snia",
    "snii": "nonIa_snlike",
    "other_sn": "nonIa_snlike",
    "non_sn": "other",
    "nonIa_snlike": "nonIa_snlike",
    "other": "other",
}
# Ternary label → the 4-way classes its probability mass folds from.
TERNARY_TO_CLASSES = {
    "snia": ("snia",),
    "nonIa_snlike": ("snii", "other_sn"),
    "other": ("non_sn",),
}

MODEL_VERSION = "seq_v10-1.0"

_NON_SN_TYPES = frozenset({
    "agn", "cv", "nova", "varstar", "galaxy", "qso", "lbv", "ilrt", "lrn",
    "kilonova", "frb", "impostor-sn", "other", "m dwarf", "star", "asteroid",
    "bogus", "vs", "afterglow",
})


def fine_class_from_raw(raw: Any) -> str | None:
    """Map a raw truth-table type string (bts_type / tns_type /
    final_class_raw) to a 4-way class, or None when ambiguous/unknown.

    Generic "SN" (subtype undetermined) is None on purpose — such rows keep
    grouped supervision instead of a fabricated subclass.
    """
    if raw is None:
        return None
    s = " ".join(str(raw).split()).strip().lower()
    if not s or s in {"-", "none", "nan", "unknown", "nonia_snlike", "sn", "at", "slsn"}:
        return None
    if s == "snia" or s.startswith("sn ia"):          # Ia, Iax, 91T/91bg, CSM, SC, pec
        return "snia"
    if s.startswith("sn ii") or s.startswith("slsn-ii") or s.startswith("slsn ii"):
        return "snii"                                  # II, IIn, IIb, IIP, IIL, II-pec, SLSN-II
    if s.startswith(("sn ib", "sn ic")) or s == "sn i" \
            or s.startswith("slsn-i") or s.startswith("slsn-r"):
        return "other_sn"                              # stripped-envelope, SLSN-I/R
    if s in _NON_SN_TYPES or s.startswith("tde"):
        return "non_sn"
    return None


def supervision_mask(
    ternary: str,
    fine: str | None = None,
    grouped_sn: bool = False,
) -> np.ndarray:
    """Bool mask over :data:`CLASSES` defining the supervised probability mass.

    ``grouped_sn=True`` (weak stamp-SN rows) → P(sn-like) vs P(non_sn) only;
    a ``fine`` 4-way label → one-hot; otherwise the ternary label's fold group
    ({snii, other_sn} for nonIa_snlike).
    """
    mask = np.zeros(len(CLASSES), dtype=bool)
    if grouped_sn:
        for c in SN_GROUP:
            mask[CLASSES.index(c)] = True
        return mask
    if fine is not None:
        if fine not in CLASSES:
            raise ValueError(f"unknown fine class {fine!r}")
        mask[CLASSES.index(fine)] = True
        return mask
    group = TERNARY_TO_CLASSES.get(ternary)
    if not group:
        raise ValueError(f"unknown ternary class {ternary!r}")
    for c in group:
        mask[CLASSES.index(c)] = True
    return mask


def fold_probs_to_ternary(proba: np.ndarray, classes: Sequence[str] = CLASSES) -> np.ndarray:
    """(…, C) probabilities over ``classes`` → (…, 3) canonical ternary."""
    proba = np.asarray(proba, dtype=np.float64)
    classes = tuple(classes)
    if classes == TERNARY_CLASSES:
        return proba
    out = np.zeros(proba.shape[:-1] + (len(TERNARY_CLASSES),), dtype=np.float64)
    for i, name in enumerate(classes):
        tern = CLASS_TO_TERNARY.get(name)
        if tern is None:
            raise ValueError(f"class {name!r} has no ternary bucket")
        out[..., TERNARY_CLASSES.index(tern)] += proba[..., i]
    return out


class SeqClassifier(nn.Module):
    def __init__(self, encoder: SeqEncoder | None = None,
                 classes: Sequence[str] = CLASSES) -> None:
        super().__init__()
        self.encoder = encoder or SeqEncoder(SeqEncoderConfig())
        self.classes = tuple(classes)
        hidden = self.encoder.config.hidden
        self.cls_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, len(self.classes)),
        )

    def forward(self, cont: torch.Tensor, bands: torch.Tensor) -> torch.Tensor:
        """(B, L, cont_dim), (B, L) → per-step logits (B, L, n_classes)."""
        return self.cls_head(self.encoder(cont, bands))


def classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor | None,
    lengths: torch.Tensor,
    object_weights: torch.Tensor,
    class_masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted grouped-CE over all real prefix steps.

    labels (B,) int64 — one class label per object, applied to every prefix
    (used only when ``class_masks`` is None → one-hot supervision, identical
    to plain CE).  class_masks (B, C) bool — the supervised probability mass
    per object: CE = −log Σ_{c∈mask} p_c.  A weak stamp-SN row masked over
    SN_GROUP therefore carries ZERO gradient on the snia-vs-snii distinction
    (the loss depends only on the grouped logsumexp).
    object_weights (B,) — quality × class-balance weight; internally divided
    by the object's number of real steps so each object contributes its
    weight once, not once per epoch (the Stage-B 1/n_rows principle).
    Returns (loss, n_terms).
    """
    B, L, C = logits.shape
    steps = torch.arange(L, device=logits.device).unsqueeze(0)
    mask = steps < lengths.unsqueeze(1)                       # (B, L)
    if class_masks is None:
        if labels is None:
            raise ValueError("labels required when class_masks is None")
        class_masks = nn.functional.one_hot(labels, C).bool()
    if not bool(class_masks.any(dim=-1).all()):
        raise ValueError("every row needs at least one supervised class")
    lse_all = torch.logsumexp(logits, dim=-1)                 # (B, L)
    grouped = logits.masked_fill(~class_masks.unsqueeze(1), float("-inf"))
    ce = lse_all - torch.logsumexp(grouped, dim=-1)           # −log Σ_mask p_c
    per_step_w = (object_weights / lengths.clamp(min=1).to(logits.dtype)).unsqueeze(1)
    loss = (ce * mask * per_step_w).sum() / object_weights.sum().clamp(min=1e-8)
    return loss, mask.sum()


@torch.no_grad()
def predict_prefix_proba(
    classifier: SeqClassifier,
    cont_norm: np.ndarray,
    bands: np.ndarray,
    *,
    temperature: float = 1.0,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """(L, n_classes) calibrated class probabilities for every prefix of ONE object."""
    classifier.eval()
    c = torch.from_numpy(cont_norm[None, ...]).to(device=device, dtype=torch.float32)
    b = torch.from_numpy(bands[None, ...]).to(device=device, dtype=torch.long)
    logits = classifier(c, b)[0] / max(temperature, 1e-3)
    return torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray | None,
    weights: np.ndarray | None = None,
    class_masks: np.ndarray | None = None,
) -> float:
    """1-parameter temperature minimizing weighted NLL (grid + golden refine).

    With ``class_masks`` (B, C) bool the NLL is the grouped-mass NLL
    (−log Σ_mask p_c), so rows with grouped supervision calibrate the mass
    they were trained on; ``labels`` is then ignored.
    """
    z = torch.from_numpy(logits.astype(np.float64))
    n = len(z)
    w = torch.from_numpy((weights if weights is not None else np.ones(n)).astype(np.float64))
    if class_masks is not None:
        m = torch.from_numpy(class_masks.astype(bool))

        def nll(t: float) -> float:
            logp = torch.log_softmax(z / t, dim=-1)
            grouped = torch.logsumexp(logp.masked_fill(~m, float("-inf")), dim=-1)
            return float(-(w * grouped).sum() / w.sum())
    else:
        if labels is None:
            raise ValueError("labels required when class_masks is None")
        y = torch.from_numpy(labels.astype(np.int64))

        def nll(t: float) -> float:
            logp = torch.log_softmax(z / t, dim=-1)
            return float(-(w * logp[torch.arange(n), y]).sum() / w.sum())

    grid = np.geomspace(0.25, 8.0, 33)
    best_t = min(grid, key=nll)
    lo, hi = best_t / 1.4, best_t * 1.4
    for _ in range(24):
        m1, m2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        if nll(m1) <= nll(m2):
            hi = m2
        else:
            lo = m1
    refined = 0.5 * (lo + hi)
    return float(refined if nll(refined) <= nll(1.0) else 1.0)


@dataclass
class SeqClassifierArtifact:
    classifier: SeqClassifier
    norm_stats: NormStats
    temperature: float
    meta: dict

    @property
    def classes(self) -> tuple[str, ...]:
        return self.classifier.classes

    @classmethod
    def load(cls, art_dir: Path | str, *, device: torch.device | str = "cpu") -> "SeqClassifierArtifact":
        art_dir = Path(art_dir)
        meta = json.loads((art_dir / "config.json").read_text())
        encoder = SeqEncoder(SeqEncoderConfig.from_json(meta["config"]))
        classes = tuple(meta.get("classes") or CLASSES)
        clf = SeqClassifier(encoder, classes=classes)
        state = torch.load(art_dir / "classifier.pt", map_location="cpu", weights_only=True)
        clf.load_state_dict(state)
        clf.to(device)
        clf.eval()
        stats = load_norm_stats(art_dir / "norm_stats.json")
        return cls(classifier=clf, norm_stats=stats,
                   temperature=float(meta.get("temperature", 1.0)), meta=meta)

    def save(self, art_dir: Path | str) -> None:
        art_dir = Path(art_dir)
        art_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.classifier.state_dict(), art_dir / "classifier.pt")
        save_norm_stats(self.norm_stats, art_dir / "norm_stats.json")
        meta = dict(self.meta)
        meta["config"] = self.classifier.encoder.config.to_json()
        meta["temperature"] = float(self.temperature)
        meta["classes"] = list(self.classifier.classes)
        meta["model_version"] = MODEL_VERSION
        (art_dir / "config.json").write_text(json.dumps(meta, indent=1, default=str))

    def predict_proba_prefixes(
        self, cont_raw: np.ndarray, bands: np.ndarray, *, device: torch.device | str = "cpu"
    ) -> np.ndarray:
        return predict_prefix_proba(
            self.classifier, self.norm_stats.apply(cont_raw), bands,
            temperature=self.temperature, device=device,
        )
