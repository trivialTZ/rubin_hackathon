"""ORACLE local expert wrapper (dev-ved30/Oracle PyTorch).

Uses the lightcurve-only variant `ORACLE1_ELAsTiCC_lite`, which classifies
LSST-band time series into a 19-class hierarchical taxonomy without needing
host-context static features (the full ORACLE1_ELAsTiCC requires 18 static
fields that DP1 doesn't carry).

Setup
-----
1. `pip install git+https://github.com/dev-ved30/Oracle.git`
2. Download weights into `artifacts/local_experts/oracle/models/ELAsTiCC-lite/revived-star-159/`:
       curl -L -o best_model_f1.pth \
         https://github.com/dev-ved30/Oracle/raw/main/models/ELAsTiCC-lite/revived-star-159/best_model_f1.pth
3. Optional: override paths via env vars
       DEBASS_ORACLE_WEIGHTS=/path/to/.../revived-star-159

Input → ORACLE shape
--------------------
ORACLE was trained on ELAsTiCC2 SNANA tables (FLUXCAL units at zero-point
27.5 mag, capital-Y for y-band). DP1 LCs are in nanojansky with lowercase
'y'. We multiply flux by 10**((27.5 - 31.4) / 2.5) ≈ 0.02754 and uppercase
the y band before invoking the model.

Caveat — domain mismatch on DP1
-------------------------------
ORACLE1_ELAsTiCC_lite is trained on synthetic 3-year LSST LCs. DP1
commissioning LCs span weeks-to-months and the snapshot-ladder evaluation
truncates to a few detections. ORACLE often returns confident periodic /
fast classifications on short DP1 LCs even for spectroscopically-confirmed
SNe. The trust-head architecture is designed to discount such systematic
mis-calibration; in practice we expect a low trust score on DP1 but a
useful contribution once LSST DP2/DR1 baselines arrive.
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
_DEFAULT_WEIGHTS_DIR = Path(
    "artifacts/local_experts/oracle/models/ELAsTiCC-lite/revived-star-159"
)

# SNANA FLUXCAL ZP=27.5 mag in counts; LSST psfFlux is in nanojansky (AB ZP=31.4 mag).
_NJY_TO_FLUXCAL = 10 ** ((27.5 - 31.4) / 2.5)

# ORACLE leaf taxonomy (19 classes) emitted at depth -1.
ORACLE_CLASSES = [
    "SNIa", "SN91bg", "SNIax", "SNII", "SNIbc", "SLSN-I", "SLSN-II",
    "IIn", "IIb", "TDE", "KN", "AGN", "uLens", "CART", "ILOT", "PISN",
    "Cep", "RRLyr", "EB",
]


def _band_to_oracle(band: str) -> str:
    """LSST band string → ORACLE-expected string (capital Y for y-band)."""
    b = str(band).strip()
    return "Y" if b == "y" else b


class OracleLocalExpert(LocalExpert):
    name = "oracle_lsst"
    semantic_type = "probability"
    requires_gpu = False  # CPU inference is OK per Shah+ 2025

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
        if not self._weights_dir.exists():
            self._reason = f"ORACLE lite weights not found at {self._weights_dir}"
            return
        if not (self._weights_dir / "best_model_f1.pth").exists():
            self._reason = f"best_model_f1.pth missing under {self._weights_dir}"
            return
        try:
            import torch  # noqa: F401
            from oracle.pretrained.ELAsTiCC import ORACLE1_ELAsTiCC_lite
        except ImportError as exc:
            self._reason = f"oracle/torch import failed: {exc}"
            return
        try:
            self._model = ORACLE1_ELAsTiCC_lite(model_dir=str(self._weights_dir))
            self._available = True
        except Exception as exc:
            self._reason = f"ORACLE_lite load failed: {type(exc).__name__}: {exc}"

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
            "weights_dir": str(self._weights_dir),
            "classes": ORACLE_CLASSES,
            "requires_gpu": self.requires_gpu,
            "model_variant": "ORACLE1_ELAsTiCC_lite (lightcurve-only)",
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
            model_version="dev-ved30/Oracle ELAsTiCC-lite revived-star-159",
            available=self._available,
        )
        if not self._available or self._model is None:
            return out
        if not isinstance(lightcurve, list):
            out.available = False
            out.raw_output["reason"] = "expected list of detections"
            return out

        # Truncate to detections at/before epoch_jd (MJD compare; tolerate
        # both `mjd` and `jd` keys per the normalize_lightcurve schema).
        truncated: list[dict] = []
        for det in lightcurve:
            t_mjd = det.get("mjd")
            if t_mjd is None and det.get("jd") is not None:
                try:
                    t_mjd = float(det["jd"]) - 2400000.5
                except (TypeError, ValueError):
                    continue
            if t_mjd is None:
                continue
            try:
                t_jd = float(t_mjd) + 2400000.5
                if t_jd <= float(epoch_jd):
                    truncated.append(det)
            except (TypeError, ValueError):
                continue
        if len(truncated) < 3:
            out.raw_output["reason"] = f"only {len(truncated)} detections"
            return out
        truncated.sort(key=lambda d: float(d.get("mjd", 0.0)))

        try:
            import numpy as np
            from astropy.table import Table
        except Exception as exc:
            out.available = False
            out.raw_output["reason"] = f"numpy/astropy import failed: {exc}"
            return out

        try:
            mjd = np.array([float(d["mjd"]) for d in truncated], dtype=float)
            flux = np.array([float(d.get("flux", float("nan"))) for d in truncated], dtype=float)
            ferr = np.array([float(d.get("fluxerr", float("nan"))) for d in truncated], dtype=float)
            band = np.array([_band_to_oracle(d.get("band", "")) for d in truncated], dtype=object)
            mask = np.isfinite(flux) & np.isfinite(ferr) & (ferr > 0) & (band != "")
            if int(mask.sum()) < 3:
                out.raw_output["reason"] = f"only {int(mask.sum())} usable rows"
                return out
            tbl = Table({
                "MJD":         mjd[mask],
                "FLUXCAL":     flux[mask] * _NJY_TO_FLUXCAL,
                "FLUXCALERR":  ferr[mask] * _NJY_TO_FLUXCAL,
                "BAND":        band[mask],
                "PHOTFLAG":    np.full(int(mask.sum()), 4096, dtype=np.int64),
            })
            scores = self._model.score(tbl)
            leaf_depth = max(scores)
            leaf_df = scores[leaf_depth]
            row = leaf_df.iloc[0].to_dict()
            probs = {str(k): float(v) for k, v in row.items() if math.isfinite(float(v))}
            if not probs:
                out.raw_output["reason"] = "empty leaf probs"
                return out
            out.class_probabilities = probs
            out.raw_output["raw_probs_json"] = json.dumps(probs)
            out.raw_output["leaf_depth"] = int(leaf_depth)
            top_class = max(probs, key=lambda k: probs[k])
            out.raw_output["top_class"] = top_class
            out.raw_output["top_prob"] = float(probs[top_class])
        except Exception as exc:
            out.available = False
            out.raw_output["reason"] = f"inference error: {type(exc).__name__}: {exc}"
        return out
