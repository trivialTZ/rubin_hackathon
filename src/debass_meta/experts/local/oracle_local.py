"""ORACLE local expert wrapper (dev-ved30/Oracle PyTorch).

Requires:
  1. `git clone https://github.com/dev-ved30/Oracle.git artifacts/local_experts/oracle`
  2. `pip install torch torchvision`
  3. ELAsTiCC2 weights under `artifacts/local_experts/oracle/models/ELAsTiCC/vocal-bird-160`

When those are absent the wrapper reports `available=False` and the meta-pipeline
skips the oracle_lsst expert — same pattern as ParSNIP.

Taxonomy: ORACLE is hierarchical with 19 leaf classes (Shah+ 2025, ApJ 995:4).
We emit the full 19-class probability vector as JSON in `raw_probs_json`; the
projector (`projectors/local_oracle.py`) maps to DEBASS ternary.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from .base import ExpertOutput, LocalExpert

_DEFAULT_REPO_PATH = Path("artifacts/local_experts/oracle")
_DEFAULT_WEIGHTS_DIR = Path("artifacts/local_experts/oracle/models/ELAsTiCC/vocal-bird-160")

# ORACLE's 19-class schema (from dev-ved30/Oracle taxonomy)
ORACLE_CLASSES = [
    "SNIa", "SN91bg", "SNIax", "SNII", "SNIbc", "SLSN-I", "SLSN-II",
    "IIn", "IIb", "TDE", "KN", "AGN", "uLens", "CART", "ILOT", "PISN",
    "Cep", "RRLyr", "EB",
]


class OracleLocalExpert(LocalExpert):
    name = "oracle_lsst"
    semantic_type = "probability"
    requires_gpu = False  # CPU-inference OK per paper

    def __init__(
        self,
        repo_path: Path = _DEFAULT_REPO_PATH,
        weights_dir: Path = _DEFAULT_WEIGHTS_DIR,
    ) -> None:
        self._repo_path = Path(os.environ.get("DEBASS_ORACLE_REPO", str(repo_path)))
        self._weights_dir = Path(os.environ.get("DEBASS_ORACLE_WEIGHTS", str(weights_dir)))
        self._model = None
        self._available = False
        self._reason: str | None = None
        self._load_model()

    def _load_model(self) -> None:
        if not self._repo_path.exists():
            self._reason = f"ORACLE repo not found at {self._repo_path}"
            return
        if not self._weights_dir.exists():
            self._reason = f"ORACLE weights not found at {self._weights_dir}"
            return
        try:
            # ORACLE uses src-layout: src/oracle/...
            src_path = str(self._repo_path / "src")
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
            import torch  # noqa: F401
        except ImportError as exc:
            self._reason = f"torch not installed: {exc}"
            return
        try:
            # ORACLE ships `ORACLE1_ELAsTiCC` in `oracle.pretrained.ELAsTiCC`
            from oracle.pretrained.ELAsTiCC import ORACLE1_ELAsTiCC  # type: ignore[import-not-found]

            self._ORACLE1_ELAsTiCC = ORACLE1_ELAsTiCC
            weights_pth = self._weights_dir / "best_model.pth"
            if not weights_pth.exists():
                self._reason = f"best_model.pth missing under {self._weights_dir}"
                return
            self._weights_pth = weights_pth
            # Model instantiation is lazy — done on first predict_epoch call to
            # avoid loading torch weights into memory when expert isn't used.
            self._available = True
        except Exception as exc:
            self._reason = f"oracle import failed: {exc}"
            return

    def fit(self, lightcurves: Any, labels: Any) -> None:
        raise NotImplementedError(
            "ORACLE is pre-trained on ELAsTiCC2; retraining is out of scope."
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "semantic_type": self.semantic_type,
            "available": self._available,
            "reason": self._reason,
            "repo_path": str(self._repo_path),
            "weights_dir": str(self._weights_dir),
            "classes": ORACLE_CLASSES,
            "requires_gpu": self.requires_gpu,
        }

    def predict_epoch(
        self, object_id: str, lightcurve: Any, epoch_jd: float
    ) -> ExpertOutput:
        out = ExpertOutput(
            expert=self.name,
            object_id=object_id,
            epoch_jd=epoch_jd,
            class_probabilities={},
            raw_output={"reason": self._reason} if self._reason else {},
            semantic_type=self.semantic_type,
            model_version="dev-ved30/Oracle ELAsTiCC2 vocal-bird-160",
            available=self._available,
        )
        if not self._available:
            return out

        # Truncate LC to detections at/before epoch_jd (MJD comparison)
        if not isinstance(lightcurve, list):
            out.available = False
            out.raw_output["reason"] = "expected list of detections"
            return out
        truncated: list[dict] = []
        for det in lightcurve:
            t_jd = det.get("jd")
            if t_jd is None and det.get("mjd") is not None:
                try:
                    t_jd = float(det["mjd"]) + 2400000.5
                except (TypeError, ValueError):
                    continue
            if t_jd is None:
                continue
            try:
                if float(t_jd) <= float(epoch_jd):
                    truncated.append(det)
            except (TypeError, ValueError):
                continue
        if len(truncated) < 3:
            out.raw_output["reason"] = f"only {len(truncated)} detections"
            return out

        try:
            # Placeholder for actual inference call against self._model.
            # The real invocation depends on dev-ved30/Oracle's API surface
            # which must be wired up once the repo is cloned.  For now, we
            # return unavailable so the pipeline skips this expert cleanly.
            out.available = False
            out.raw_output["reason"] = "inference wiring TODO after repo clone"
        except Exception as exc:
            out.available = False
            out.raw_output["reason"] = f"inference error: {exc}"
        return out
