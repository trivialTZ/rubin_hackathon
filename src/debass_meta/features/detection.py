"""Survey-aware detection normalization.

Converts ZTF and LSST detection dicts into a common schema so that
downstream feature extraction and model training work identically
regardless of the source survey.

ZTF detections use integer ``fid`` (1=g, 2=r), PSF magnitudes,
and real-bogus scores (``rb``/``drb``).

LSST detections (from Lasair diaSource records) use string ``band``
(u/g/r/i/z/y), PSF fluxes in nanojansky (``psfFlux``), and a
``reliability`` score.
"""
from __future__ import annotations

import math
from typing import Any

# Canonical LSST band order
CANONICAL_BANDS = ("u", "g", "r", "i", "z", "y")
BAND_TO_IDX = {b: i for i, b in enumerate(CANONICAL_BANDS)}

# ZTF filter id → canonical band name
ZTF_FID_TO_BAND = {1: "g", 2: "r", 3: "i"}

# AB magnitude zero-point: 3631 Jy = 3.631e9 nJy → mag = -2.5*log10(flux_nJy) + 31.4
_AB_ZEROPOINT = 31.4
_LN10_OVER_2P5 = math.log(10.0) / 2.5  # ≈ 0.9210


def flux_to_mag(flux_njy: float, fluxerr_njy: float | None = None) -> tuple[float, float | None]:
    """Convert nanojansky flux to AB magnitude.

    Returns (mag, magerr).  magerr is None when fluxerr is not provided
    or flux is non-positive.
    """
    if flux_njy is None or not math.isfinite(flux_njy) or flux_njy <= 0:
        return (math.nan, None)
    mag = -2.5 * math.log10(flux_njy) + _AB_ZEROPOINT
    magerr = None
    if fluxerr_njy is not None and math.isfinite(fluxerr_njy) and fluxerr_njy > 0:
        magerr = 2.5 / (math.log(10.0)) * fluxerr_njy / flux_njy
    return (mag, magerr)


def mag_to_flux(mag: float, magerr: float | None = None) -> tuple[float, float | None]:
    """Convert AB magnitude to nanojansky flux.

    Returns (flux_njy, fluxerr_njy).
    """
    if mag is None or not math.isfinite(mag):
        return (math.nan, None)
    flux = 10.0 ** ((_AB_ZEROPOINT - mag) / 2.5)
    fluxerr = None
    if magerr is not None and math.isfinite(magerr) and magerr > 0:
        fluxerr = flux * math.log(10.0) / 2.5 * magerr
    return (flux, fluxerr)


def _detect_survey(det: dict[str, Any]) -> str:
    """Heuristic: if the dict has ``psfFlux`` or ``midpointMjdTai``, it's LSST."""
    if "psfFlux" in det or "midpointMjdTai" in det or "diaSourceId" in det:
        return "LSST"
    return "ZTF"


def _safe_float(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        f = float(value)
        return f if math.isfinite(f) else math.nan
    except (TypeError, ValueError):
        return math.nan


def normalize_detection(det: dict[str, Any], *, survey: str = "auto") -> dict[str, Any]:
    """Convert a single ZTF or LSST detection dict to the common schema.

    Parameters
    ----------
    det : dict
        Raw detection dict from either survey.
    survey : str
        ``"ZTF"``, ``"LSST"``, or ``"auto"`` (auto-detect from keys).

    Returns
    -------
    dict with keys: mjd, band, band_idx, mag, magerr, flux, fluxerr,
    is_positive, quality, snr, survey.  Original keys are preserved
    under their original names for provenance.
    """
    if survey == "auto":
        survey = _detect_survey(det)

    out: dict[str, Any] = dict(det)  # preserve originals
    out["survey"] = survey

    if survey == "LSST":
        _normalize_lsst(det, out)
    else:
        _normalize_ztf(det, out)

    return out


def _normalize_ztf(det: dict[str, Any], out: dict[str, Any]) -> None:
    # Time
    out["mjd"] = _safe_float(det.get("mjd") or det.get("jd"))

    # Band
    fid = det.get("fid")
    band = ZTF_FID_TO_BAND.get(int(fid), "r") if fid is not None else "r"
    out["band"] = band
    out["band_idx"] = BAND_TO_IDX.get(band, -1)

    # Magnitude (primary) → flux (derived)
    mag = _safe_float(det.get("magpsf"))
    magerr = _safe_float(det.get("sigmapsf"))
    out["mag"] = mag
    out["magerr"] = magerr if math.isfinite(magerr) else None
    flux, fluxerr = mag_to_flux(mag, magerr if math.isfinite(magerr) else None)
    out["flux"] = flux
    out["fluxerr"] = fluxerr

    # Detection flag
    isdiffpos = det.get("isdiffpos")
    out["is_positive"] = isdiffpos in ("t", "1", 1, True, "true")

    # Quality (prefer drb over rb)
    drb = det.get("drb")
    rb = det.get("rb")
    quality = _safe_float(drb if drb is not None else rb)
    out["quality"] = quality if math.isfinite(quality) else None

    # SNR
    if math.isfinite(mag) and math.isfinite(magerr) and magerr > 0:
        out["snr"] = 1.0 / (_LN10_OVER_2P5 * magerr)
    else:
        out["snr"] = None


def _normalize_lsst(det: dict[str, Any], out: dict[str, Any]) -> None:
    # Time
    out["mjd"] = _safe_float(det.get("midpointMjdTai") or det.get("mjd") or det.get("jd"))

    # Band (already a string in LSST)
    band = str(det.get("band") or "").strip().lower()
    if band not in BAND_TO_IDX:
        band = "r"  # fallback
    out["band"] = band
    out["band_idx"] = BAND_TO_IDX.get(band, -1)

    # Flux (primary) → magnitude (derived)
    flux = _safe_float(det.get("psfFlux"))
    fluxerr = _safe_float(det.get("psfFluxErr"))
    out["flux"] = flux
    out["fluxerr"] = fluxerr if math.isfinite(fluxerr) else None
    mag, magerr = flux_to_mag(flux, fluxerr if math.isfinite(fluxerr) else None)
    out["mag"] = mag
    out["magerr"] = magerr

    # Detection flag (LSST uses isNegative — inverted)
    is_negative = det.get("isNegative", False)
    out["is_positive"] = not bool(is_negative)

    # Quality
    reliability = _safe_float(det.get("reliability"))
    out["quality"] = reliability if math.isfinite(reliability) else None

    # SNR
    snr = _safe_float(det.get("snr"))
    out["snr"] = snr if math.isfinite(snr) else None


def normalize_lightcurve(
    detections: list[dict[str, Any]],
    *,
    survey: str = "auto",
) -> list[dict[str, Any]]:
    """Normalize a list of detection dicts.

    If *survey* is ``"auto"``, each detection is auto-detected independently
    (allows mixed-survey lightcurves, though unusual).
    """
    return [normalize_detection(d, survey=survey) for d in detections]
