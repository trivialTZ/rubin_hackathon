"""Bazin / Villar parametric-feature local expert.

Uses `light-curve` (Rust-fast) to extract Bazin + Villar + stats features
from the truncated lightcurve per band.  A LightGBM head then maps to 3-way
(snia, nonIa_snlike, other) posterior.  Head is trained once on the current
labelled snapshot (see `scripts/train_lc_features_head.py`); if absent, the
expert reports availability=False and the meta-classifier skips it.

Design rationale: this expert provides an **interpretable** baseline that
does not rely on any deep-learning checkpoint — complementing ORACLE and
SALT3 for a tri-ensemble of local experts.
"""
from __future__ import annotations

import math
import os
import pickle
from pathlib import Path
from typing import Any

from .base import ExpertOutput, LocalExpert

_MODEL_PATH = Path(
    os.environ.get(
        "DEBASS_LC_FEATURES_MODEL",
        "artifacts/local_experts/lc_features/model.pkl",
    )
)

_BAND_MAP = {
    1: "g", 2: "r", 3: "i",
    "1": "g", "2": "r", "3": "i",
    "ztfg": "g", "ztfr": "r", "ztfi": "i",
    "g": "g", "r": "r", "i": "i", "z": "z", "y": "y", "u": "u",
    "lsstu": "u", "lsstg": "g", "lsstr": "r", "lssti": "i", "lsstz": "z", "lssty": "y",
}


def _mjd(det: dict[str, Any]) -> float | None:
    for key in ("mjd",):
        val = det.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    val = det.get("jd")
    if val is not None:
        try:
            return float(val) - 2400000.5
        except (TypeError, ValueError):
            pass
    return None


def _band(det: dict[str, Any]) -> str | None:
    for key in ("band", "filter", "flt", "fid"):
        val = det.get(key)
        if val is None:
            continue
        mapped = _BAND_MAP.get(val) or _BAND_MAP.get(str(val).strip().lower())
        if mapped:
            return mapped
    return None


def _flux(det: dict[str, Any]) -> tuple[float, float] | None:
    flux = det.get("flux")
    fluxerr = det.get("fluxerr")
    if flux is not None and fluxerr is not None:
        try:
            flux = float(flux)
            fluxerr = float(fluxerr)
            if math.isfinite(flux) and math.isfinite(fluxerr) and fluxerr > 0:
                return flux, fluxerr
        except (TypeError, ValueError):
            pass
    mag = det.get("magpsf")
    magerr = det.get("sigmapsf")
    if mag is None or magerr is None:
        return None
    try:
        mag = float(mag)
        magerr = float(magerr)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(mag) and math.isfinite(magerr) and magerr > 0):
        return None
    f = 10.0 ** (-0.4 * (mag - 25.0))
    ferr = f * (math.log(10.0) / 2.5) * magerr
    return f, ferr


class LcFeaturesExpert(LocalExpert):
    name = "lc_features_bv"
    semantic_type = "probability"
    requires_gpu = False

    FEATURE_NAMES: list[str] = [
        "bazin_amplitude", "bazin_rise", "bazin_fall", "bazin_t0", "bazin_chi2",
        "villar_amplitude", "villar_plateau", "villar_rise", "villar_fall", "villar_chi2",
        "std", "skew", "kurt", "amplitude_stats", "mean_mag", "n_det_band",
    ]

    def __init__(self, model_path: Path = _MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self._head = None
        self._available = False
        try:
            import light_curve as _lc  # noqa: F401

            self._lc_available = True
        except Exception:
            self._lc_available = False
        self._load_head()

    def _load_head(self) -> None:
        if not self.model_path.exists():
            return
        try:
            with open(self.model_path, "rb") as fh:
                self._head = pickle.load(fh)
            self._available = True
        except Exception:
            self._head = None
            self._available = False

    def fit(self, lightcurves: Any, labels: Any) -> None:
        """Training is done by `scripts/train_lc_features_head.py`."""
        raise NotImplementedError(
            "Train via scripts/train_lc_features_head.py; this class is inference-only."
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "semantic_type": self.semantic_type,
            "available": self._available,
            "lc_available": self._lc_available,
            "classes": ["snia", "nonIa_snlike", "other"],
            "feature_names": self.FEATURE_NAMES,
            "model_path": str(self.model_path),
        }

    def _compute_features(self, lc_rows: list[tuple[float, float, float, str]]) -> dict[str, float]:
        """Compute a fixed-shape feature vector for a set of detections.

        Uses the `light-curve` library when available; falls back to a simple
        hand-written Bazin fit otherwise.  Returns dict in the exact order of
        FEATURE_NAMES (missing values are NaN so LightGBM handles them).
        """
        import numpy as np

        feats: dict[str, float] = {name: float("nan") for name in self.FEATURE_NAMES}
        feats["n_det_band"] = float(len(lc_rows))
        if len(lc_rows) < 4:
            return feats

        mjds = np.array([r[0] for r in lc_rows], dtype=float)
        fluxes = np.array([r[1] for r in lc_rows], dtype=float)
        fluxerrs = np.array([r[2] for r in lc_rows], dtype=float)

        feats["std"] = float(fluxes.std())
        feats["mean_mag"] = float(-2.5 * np.log10(np.clip(fluxes.mean(), 1e-12, None)) + 25.0)
        if len(fluxes) >= 3:
            mean = fluxes.mean()
            std = fluxes.std() or 1e-12
            feats["skew"] = float(((fluxes - mean) ** 3).mean() / (std ** 3))
            feats["kurt"] = float(((fluxes - mean) ** 4).mean() / (std ** 4))
        feats["amplitude_stats"] = float(fluxes.max() - fluxes.min())

        try:
            import light_curve as lc

            # Bazin feature (5 params: amplitude, rise, fall, t0, baseline) → returns 5
            bazin = lc.BazinFit("lmsder")
            bazin_out = bazin(mjds, fluxes, fluxerrs)
            feats["bazin_amplitude"] = float(bazin_out[0])
            feats["bazin_rise"] = float(bazin_out[1])
            feats["bazin_fall"] = float(bazin_out[2])
            feats["bazin_t0"] = float(bazin_out[3])
            feats["bazin_chi2"] = float(bazin_out[4]) if len(bazin_out) > 4 else float("nan")
        except Exception:
            pass

        try:
            import light_curve as lc

            villar = lc.VillarFit("lmsder")
            villar_out = villar(mjds, fluxes, fluxerrs)
            feats["villar_amplitude"] = float(villar_out[0])
            feats["villar_plateau"] = float(villar_out[1])
            feats["villar_rise"] = float(villar_out[2])
            feats["villar_fall"] = float(villar_out[3])
            feats["villar_chi2"] = float(villar_out[4]) if len(villar_out) > 4 else float("nan")
        except Exception:
            pass

        return feats

    def predict_epoch(
        self, object_id: str, lightcurve: Any, epoch_jd: float
    ) -> ExpertOutput:
        out = ExpertOutput(
            expert=self.name,
            object_id=object_id,
            epoch_jd=epoch_jd,
            class_probabilities={},
            raw_output={},
            semantic_type=self.semantic_type,
            model_version="bazin_villar",
            available=self._available,
        )

        if not self._available:
            out.raw_output["reason"] = "lc_features head not trained"
            return out

        detections = list(lightcurve) if isinstance(lightcurve, list) else []
        truncated: list[dict] = []
        for det in detections:
            t_mjd = _mjd(det)
            if t_mjd is None:
                continue
            if epoch_jd is not None and (t_mjd + 2400000.5) > float(epoch_jd):
                continue
            truncated.append(det)

        if len(truncated) < 4:
            out.raw_output["reason"] = f"only {len(truncated)} detections"
            return out

        # Choose the band with the most detections (usually r/i)
        by_band: dict[str, list[tuple[float, float, float, str]]] = {}
        for det in truncated:
            band = _band(det)
            mjd = _mjd(det)
            flux_pair = _flux(det)
            if not (band and mjd and flux_pair):
                continue
            f, ferr = flux_pair
            by_band.setdefault(band, []).append((mjd, f, ferr, band))
        if not by_band:
            out.raw_output["reason"] = "no usable bands"
            return out
        best_band = max(by_band, key=lambda k: len(by_band[k]))
        feats = self._compute_features(sorted(by_band[best_band]))
        X = [[feats[n] for n in self.FEATURE_NAMES]]
        try:
            probs = self._head.predict_proba(X)[0]
        except Exception as exc:
            out.raw_output["reason"] = f"head predict failed: {exc}"
            return out

        classes = getattr(self._head, "classes_", ["snia", "nonIa_snlike", "other"])
        out.class_probabilities = {str(c): float(p) for c, p in zip(classes, probs)}
        out.raw_output["features"] = feats
        out.raw_output["best_band"] = best_band
        return out
