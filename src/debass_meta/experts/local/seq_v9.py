"""seq_v9 local expert — the fine-tuned sequence classifier (v10 artifacts).

A causal GRU over per-detection photometry (mixed ZTF+LSST band vocabulary)
emitting class probabilities per epoch prefix.  Trained by
``scripts/train_seq_classifier.py`` (SSL warm start + supervised fine-tune,
train split only); artifact at ``models/seq_classifier_v10`` (falls back to
``models/seq_classifier_v9``; override via ``DEBASS_SEQ_V9_MODEL``).

v10 K-fold OOF routing: when the artifact dir carries ``fold_map.json``
(object_id → fold index k, written by --oof-folds), an object listed there is
scored by ``fold_{k}/`` — the model whose training EXCLUDED fold k — so train
rows reach silver out-of-fold instead of in-sample (the v9c stacking trap).
Objects not in the map (cal/test/new) use the full-data root artifact.
Without a fold_map.json the behaviour is exactly the pre-v10 single-artifact
path.

The emitted ``class_probabilities`` keep the canonical TERNARY contract
(snia / nonIa_snlike / other, folded from the 4-way head).  When the artifact
is 4-way, the raw 4-way probabilities are ALSO included under ``p4_<class>``
keys — they flow into the silver payload as extra class events (recoverable
downstream) and are ignored by the ternary projector.

Availability follows the repo convention: if the artifact is missing the
expert reports ``available=False`` and the meta-classifier skips it.
CPU inference (~1 ms/epoch) — ``requires_gpu`` stays False so the expert runs
in the CPU pipeline stage and on RSP/DP1 nodes.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from .base import ExpertOutput, LocalExpert

_MODEL_DIR_CANDIDATES = ("models/seq_classifier_v10", "models/seq_classifier_v9")


def _default_model_dir() -> Path:
    env = os.environ.get("DEBASS_SEQ_V9_MODEL")
    if env:
        return Path(env)
    for candidate in _MODEL_DIR_CANDIDATES:
        if (Path(candidate) / "classifier.pt").exists():
            return Path(candidate)
    return Path(_MODEL_DIR_CANDIDATES[0])


class SeqV9Expert(LocalExpert):
    name = "seq_v9"
    semantic_type = "probability"
    requires_gpu = False

    def __init__(self, model_dir: Path | str | None = None) -> None:
        self._model_dir = Path(model_dir) if model_dir else _default_model_dir()
        self._artifact = None
        self._fold_map: dict[str, int] = {}
        self._fold_artifacts: dict[int, Any] = {}
        self._load_failed: str | None = None

    # -- lazy loading ------------------------------------------------------
    def _ensure_loaded(self) -> bool:
        if self._artifact is not None:
            return True
        if self._load_failed is not None:
            return False
        try:
            from debass_meta.models.seq_classifier import SeqClassifierArtifact

            self._artifact = SeqClassifierArtifact.load(self._model_dir)
        except Exception as exc:  # missing artifact / torch absent
            self._load_failed = f"{type(exc).__name__}: {exc}"
            return False
        # OUTSIDE the availability guard: a fold_map.json that EXISTS but
        # cannot be parsed must fail LOUDLY — silently routing every train
        # object to the full-data root model would re-create the v9c
        # stacking trap (in-sample train-row projections) with no error
        # anywhere in the chain.
        self._fold_map = self._load_fold_map()
        return True

    def _load_fold_map(self) -> dict[str, int]:
        """object_id → fold index (empty when no fold_map.json — pre-v10).

        Raises RuntimeError when the file exists but cannot be parsed
        (e.g. a truncated/interrupted write): OOF routing is a correctness
        contract, never a best-effort optimisation.
        """
        path = self._model_dir / "fold_map.json"
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text())
            return {str(k): int(v) for k, v in (payload.get("assignments") or {}).items()}
        except Exception as exc:
            raise RuntimeError(
                f"corrupt fold_map.json at {path} ({type(exc).__name__}: {exc}) — "
                f"refusing to score without OOF routing (train rows would be "
                f"in-sample). Rebuild the artifact or retrain with --oof-folds."
            ) from exc

    def _artifact_for(self, object_id: str) -> tuple[Any, Any]:
        """(artifact, fold_route) — the model that never saw this object.

        Objects in the fold map route to fold_{k} (trained WITHOUT fold k);
        everything else uses the full-data root artifact (fold_route None).
        Falls back to the root artifact if a fold dir is missing/broken.
        """
        k = self._fold_map.get(str(object_id))
        if k is None:
            return self._artifact, None
        if k not in self._fold_artifacts:
            from debass_meta.models.seq_classifier import SeqClassifierArtifact

            try:
                self._fold_artifacts[k] = SeqClassifierArtifact.load(self._model_dir / f"fold_{k}")
            except Exception:
                self._fold_artifacts[k] = None
        fold_art = self._fold_artifacts[k]
        if fold_art is None:
            return self._artifact, f"missing_fold_{k}_fallback_full"
        return fold_art, k

    # -- LocalExpert interface ----------------------------------------------
    def fit(self, lightcurves: Any, labels: Any) -> None:
        raise NotImplementedError(
            "seq_v9 is trained via scripts/train_seq_classifier.py "
            "(SSL warm start + split-disciplined fine-tune), not via fit()."
        )

    def predict_epoch(self, object_id: str, lightcurve: Any, epoch_jd: float) -> ExpertOutput:
        if not self._ensure_loaded():
            return ExpertOutput(
                expert=self.name, object_id=str(object_id), epoch_jd=float(epoch_jd),
                available=False, model_version="unavailable",
                raw_output={"reason": self._load_failed or "artifact missing"},
            )
        from debass_meta.features.sequence_dataset import sequence_arrays
        from debass_meta.models.seq_classifier import TERNARY_CLASSES, fold_probs_to_ternary

        detections = list(lightcurve or [])
        # The caller hands the truncated epoch prefix; sequence_arrays
        # re-normalizes/sorts/positive-filters idempotently (same recipe as
        # the gold builder) so raw or normalized input both work.
        cont, bands = sequence_arrays(detections)
        if len(cont) == 0:
            return ExpertOutput(
                expert=self.name, object_id=str(object_id), epoch_jd=float(epoch_jd),
                available=False, model_version=self._version(),
                raw_output={"reason": "no positive detections"},
            )
        artifact, fold_route = self._artifact_for(object_id)
        proba = artifact.predict_proba_prefixes(cont, bands)[-1]
        classes = artifact.classes
        ternary = fold_probs_to_ternary(proba, classes)
        probs = {name: float(p) for name, p in zip(TERNARY_CLASSES, ternary)}
        top = max(probs.values())
        entropy = -sum(p * math.log(p + 1e-12) for p in probs.values())
        raw: dict[str, Any] = {
            "n_det_used": int(len(cont)),
            "top1_prob": top,
            "entropy": entropy,
            "temperature": float(artifact.temperature),
            "fold_route": fold_route,
        }
        if classes != TERNARY_CLASSES:
            # Persist the fine-grained head in the silver payload: extra
            # p4_<class> keys become class events downstream (the ternary
            # projector ignores them; they stay recoverable).
            fine = {f"p4_{name}": float(p) for name, p in zip(classes, proba)}
            probs.update(fine)
            raw["class_probabilities_4way"] = {name: float(p) for name, p in zip(classes, proba)}
        return ExpertOutput(
            expert=self.name, object_id=str(object_id), epoch_jd=float(epoch_jd),
            class_probabilities=probs,
            raw_output=raw,
            model_version=self._version(),
            available=True,
        )

    def metadata(self) -> dict[str, Any]:
        loaded = self._ensure_loaded()
        fine_classes = list(self._artifact.classes) if loaded else None
        return {
            "name": self.name,
            "version": self._version(),
            "classes": ["snia", "nonIa_snlike", "other"],
            **({"classes_fine": fine_classes} if fine_classes else {}),
            "input": "per-detection photometry sequence (mjd, band, mag, magerr, snr), truncated prefix",
            "surveys": ["ZTF", "LSST"],
            "available": loaded,
            "model_dir": str(self._model_dir),
            "oof_fold_routing": bool(self._fold_map),
            "n_fold_mapped_objects": len(self._fold_map),
            **({} if loaded else {"reason": self._load_failed}),
        }

    def _version(self) -> str:
        if self._artifact is not None:
            return str(self._artifact.meta.get("model_version", "seq_v10"))
        return "unavailable"
