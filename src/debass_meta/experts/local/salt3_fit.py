"""SALT3 χ² local expert — physics-motivated SN Ia consistency feature.

Fits the SALT3 (T. Kenworthy et al. 2021) SN Ia SED model to the truncated
lightcurve via `sncosmo` and records χ²/ndof.  For comparison, also fits a
Nugent-II-P template as a non-Ia control.  The difference

    Δχ² = χ²_nonIa - χ²_Ia

is a Bayes-factor-like score: positive favours Ia.  The projector maps
Δχ² → p_snia = sigmoid(-Δχ²/2).

No training weights needed — sncosmo ships SALT3 and Nugent templates.
Works on ZTF (ztfg/ztfr) and LSST (lsst*) bandpasses.
"""
from __future__ import annotations

import math
from typing import Any

from .base import ExpertOutput, LocalExpert

_BAND_MAP = {
    # ZTF
    1: "ztfg", 2: "ztfr", 3: "ztfi",
    "1": "ztfg", "2": "ztfr", "3": "ztfi",
    "g": "ztfg", "r": "ztfr", "i": "ztfi",
    "ztfg": "ztfg", "ztfr": "ztfr", "ztfi": "ztfi",
    # LSST (sncosmo ships these)
    "u": "lsstu", "z": "lsstz", "y": "lssty",
    "lsstu": "lsstu", "lsstg": "lsstg", "lsstr": "lsstr",
    "lssti": "lssti", "lsstz": "lsstz", "lssty": "lssty",
}

_IA_MODEL = "salt3"
_NONIA_MODEL = "nugent-sn2p"


def _mag_to_flux(mag: float, magerr: float, zp: float = 25.0) -> tuple[float, float]:
    flux = 10.0 ** (-0.4 * (mag - zp))
    fluxerr = flux * (math.log(10.0) / 2.5) * magerr
    return flux, fluxerr


def _band_name(det: dict[str, Any]) -> str | None:
    for key in ("band", "filter", "flt", "fid"):
        val = det.get(key)
        if val is None:
            continue
        mapped = _BAND_MAP.get(val) or _BAND_MAP.get(str(val).strip().lower())
        if mapped:
            return mapped
    return None


def _obs_time(det: dict[str, Any]) -> float | None:
    for key in ("mjd",):
        val = det.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    val = det.get("jd")
    if val is not None:
        try:
            return float(val) - 2400000.5
        except (TypeError, ValueError):
            pass
    return None


class Salt3Chi2Expert(LocalExpert):
    name = "salt3_chi2"
    semantic_type = "probability"
    requires_gpu = False

    def __init__(self, redshift: float | None = None) -> None:
        self._redshift = redshift
        try:
            import sncosmo  # noqa: F401

            self._available = True
        except Exception:
            self._available = False

    def fit(self, lightcurves: Any, labels: Any) -> None:
        # Template-based; no training required.
        pass

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "semantic_type": self.semantic_type,
            "available": self._available,
            "ia_model": _IA_MODEL,
            "nonia_model": _NONIA_MODEL,
            "classes": ["Ia", "II"],
            "bands_supported": ["ztfg", "ztfr", "lsstu", "lsstg", "lsstr", "lssti", "lsstz", "lssty"],
        }

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
            model_version=f"{_IA_MODEL}+{_NONIA_MODEL}",
            available=self._available,
        )
        if not self._available:
            return out

        try:
            import numpy as np
            import sncosmo
            from astropy.table import Table
        except Exception as exc:
            out.available = False
            out.raw_output["reason"] = f"import failed: {exc}"
            return out

        detections = list(lightcurve) if isinstance(lightcurve, list) else []
        # Truncate to detections at/before epoch
        truncated: list[dict] = []
        for det in detections:
            t_mjd = _obs_time(det)
            if t_mjd is None:
                continue
            if epoch_jd is not None and (t_mjd + 2400000.5) > float(epoch_jd):
                continue
            truncated.append(det)
        if len(truncated) < 3:
            out.raw_output["reason"] = f"only {len(truncated)} detections at epoch"
            return out

        rows: list[tuple[float, str, float, float, float]] = []
        for det in truncated:
            band = _band_name(det)
            mjd = _obs_time(det)
            if band is None or mjd is None:
                continue
            flux = det.get("flux")
            fluxerr = det.get("fluxerr")
            if flux is None or fluxerr is None:
                mag = det.get("magpsf")
                magerr = det.get("sigmapsf")
                if mag is None or magerr is None:
                    continue
                try:
                    mag = float(mag)
                    magerr = float(magerr)
                except (TypeError, ValueError):
                    continue
                if not (math.isfinite(mag) and math.isfinite(magerr) and magerr > 0):
                    continue
                flux, fluxerr = _mag_to_flux(mag, magerr)
            try:
                flux = float(flux)
                fluxerr = float(fluxerr)
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(flux) and math.isfinite(fluxerr) and fluxerr > 0):
                continue
            rows.append((mjd, band, flux, fluxerr, 25.0))

        if len(rows) < 3:
            out.raw_output["reason"] = f"only {len(rows)} usable detections"
            return out

        tbl = Table(
            rows=rows,
            names=("mjd", "band", "flux", "fluxerr", "zp"),
        )
        tbl["zpsys"] = "ab"

        def _fit(model_name: str) -> dict[str, float | None]:
            try:
                model = sncosmo.Model(source=model_name)
                if self._redshift is not None and math.isfinite(self._redshift):
                    model.set(z=self._redshift)
                    free_params = [p for p in model.param_names if p != "z"]
                    bounds = None
                else:
                    free_params = [p for p in model.param_names if p != "mwebv"]
                    bounds = {"z": (0.0, 0.5)} if "z" in model.param_names else None
                t0_guess = float(np.median(tbl["mjd"]))
                if "t0" in model.param_names:
                    model.set(t0=t0_guess)
                result, _ = sncosmo.fit_lc(
                    tbl, model, free_params, bounds=bounds, modelcov=False
                )
                chisq = float(result.chisq)
                ndof = int(result.ndof) if result.ndof else max(len(rows) - len(free_params), 1)
                params = {n: float(model.get(n)) for n in model.param_names if n in ("t0", "x1", "c", "x0")}
                return {"chi2": chisq, "ndof": ndof, **params}
            except Exception as exc:
                return {"error": str(exc), "chi2": None, "ndof": None}

        ia_fit = _fit(_IA_MODEL)
        nonia_fit = _fit(_NONIA_MODEL)
        out.raw_output["ia_fit"] = ia_fit
        out.raw_output["nonia_fit"] = nonia_fit

        if ia_fit.get("chi2") is not None and nonia_fit.get("chi2") is not None:
            delta = nonia_fit["chi2"] - ia_fit["chi2"]  # positive favours Ia
            out.raw_output["delta_chi2"] = delta
            # sigmoid-based Bayes-factor mapping with a gentle temperature
            prob_ia = 1.0 / (1.0 + math.exp(-delta / 2.0))
            out.class_probabilities = {
                "Ia": prob_ia,
                "II": 1.0 - prob_ia,
            }
        else:
            out.raw_output["reason"] = "fit failed"
        return out
