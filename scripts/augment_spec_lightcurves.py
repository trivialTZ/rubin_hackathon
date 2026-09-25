"""WP4 — GP / parametric lightcurve augmentation of spec-labeled objects.

Boone (2019) / PLAsTiCC-winner technique.  We have very few spec-labeled LSST
objects but many spec-labeled *real* lightcurves overall (ZTF g,r; some LSST).
For each spec-labeled source object we fit a smooth flux-vs-time model per band
(Gaussian Process in flux space, with a Bazin parametric fallback), then resample
that fit onto *real* LSST cadence templates with realistic per-epoch noise —
generating ``K`` augmented labeled lightcurves per source.  This multiplies the
scarce type-level supervision while letting us control the ``n_det`` distribution
per class (which also fights the length-shortcut failure mode).

Design notes
------------
* FIT is done in **flux space** (not magnitude) so negative difference fluxes
  are legal and the baseline / fading tail is modelled directly.
* Per band: >=4 finite-flux epochs → try a Matern GP; if the GP is ill-conditioned
  (raises, or predicts non-finite) fall back to a Bazin fit.  3 epochs → Bazin.
  <3 epochs → the band is skipped (we never fabricate a band the fit barely saw).
* Objects with <5 total finite-flux detections are skipped.
* ZTF sources are only ever resampled in bands ``g`` and ``r`` (the only bands
  a ZTF fit ever sees); LSST sources use whichever of ``ugrizy`` they carry.
* RESAMPLE draws a real LSST cadence template (delta-t sequence, band sequence,
  per-epoch flux-error magnitudes), restricted to the source's fitted bands,
  aligns the source peak to a jittered phase of the template, evaluates the fit,
  and adds Gaussian noise with the template's per-epoch flux errors.  Optional
  ``--time-dilate`` applies a small ``1+dz`` stretch about the peak.
* OUTPUT lightcurve JSONs are written in the EXACT normalized-LSST schema the
  real cached files use (``psfFlux``/``psfFluxErr``/``band``/``mjd``/``isNegative``
  + all the derived fields ``normalize_detection`` adds) so that
  ``normalize_lightcurve`` + the v11 sequence tokenizer consume them unmodified.
  This is asserted at write time by round-tripping one output through the
  tokenizer (``sequence_arrays(schema="v11")``).
* Augmented truth rows carry ``label_quality='augmented'`` /
  ``label_source='gp_augment_v1'`` so they can NEVER enter any eval frame — the
  trainer's ``--extra-truth`` path is what folds them into training only.
* ``--seed`` makes the whole run bit-reproducible (per-object / per-augmentation
  RNG streams are spawned deterministically from a single SeedSequence).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from debass_meta.features.detection import (  # noqa: E402
    CANONICAL_BANDS,
    normalize_detection,
    normalize_lightcurve,
)

# ── constants ────────────────────────────────────────────────────────────────
MIN_TOTAL_DETECTIONS = 5          # objects with fewer finite-flux dets are skipped
MIN_BAND_FOR_GP = 4               # >=4 dets in a band → try GP first
MIN_BAND_FOR_FIT = 3              # 3 dets → Bazin only; <3 → skip band
ZTF_RESAMPLE_BANDS = ("g", "r")   # never fabricate bands a ZTF fit never saw
LABEL_QUALITY = "augmented"
LABEL_SOURCE = "gp_augment_v1"
DEFAULT_PEAK_JITTER_DAYS = 3.0
_BAND_MAP = {"6": "u", "1": "g", "2": "r", "3": "i", "4": "z", "5": "y"}
_BAND_IDX = {b: i for i, b in enumerate(CANONICAL_BANDS)}


# ── survey / series extraction ────────────────────────────────────────────────
def infer_source_survey(detections: Sequence[dict[str, Any]]) -> str:
    """Return "LSST" if any raw detection looks like an LSST diaSource, else "ZTF"."""
    for det in detections:
        if any(k in det for k in ("psfFlux", "midpointMjdTai", "diaSourceId")):
            return "LSST"
        if "band" in det and "psfFlux" in det:
            return "LSST"
    # ZTF detections carry integer fid
    for det in detections:
        if "fid" in det:
            return "ZTF"
    return "LSST"


@dataclass
class BandSeries:
    mjd: np.ndarray
    flux: np.ndarray
    fluxerr: np.ndarray


def collect_band_series(
    normalized: Sequence[dict[str, Any]],
    *,
    allowed_bands: Sequence[str] | None = None,
) -> dict[str, BandSeries]:
    """Group finite-flux detections by band → (mjd, flux, fluxerr) arrays.

    Uses the normalized ``flux`` (nJy, signed) so negative difference fluxes and
    the baseline are kept in the fit.  ``allowed_bands`` restricts the bands
    considered (used to clamp ZTF sources to g,r).
    """
    buckets: dict[str, list[tuple[float, float, float]]] = {}
    for det in normalized:
        band = str(det.get("band") or "").lower()
        if band not in _BAND_IDX:
            continue
        if allowed_bands is not None and band not in allowed_bands:
            continue
        mjd = det.get("mjd")
        flux = det.get("flux")
        if mjd is None or flux is None:
            continue
        if not (math.isfinite(mjd) and math.isfinite(flux)):
            continue
        ferr = det.get("fluxerr")
        ferr = float(ferr) if ferr is not None and math.isfinite(ferr) else math.nan
        buckets.setdefault(band, []).append((float(mjd), float(flux), ferr))
    out: dict[str, BandSeries] = {}
    for band, rows in buckets.items():
        rows.sort(key=lambda r: r[0])
        mjd = np.array([r[0] for r in rows], dtype=float)
        flux = np.array([r[1] for r in rows], dtype=float)
        ferr = np.array([r[2] for r in rows], dtype=float)
        # fill missing errors with a robust fraction of the flux scale
        scale = np.nanstd(flux) if np.isfinite(np.nanstd(flux)) else 0.0
        fill = max(scale * 0.1, 1.0)
        ferr = np.where(np.isfinite(ferr) & (ferr > 0), ferr, fill)
        out[band] = BandSeries(mjd=mjd, flux=flux, fluxerr=ferr)
    return out


# ── parametric (Bazin) model ──────────────────────────────────────────────────
def bazin(t: np.ndarray, A: float, t0: float, t_rise: float, t_fall: float, B: float) -> np.ndarray:
    """Bazin (2011) SN flux model.  Vectorized, numerically guarded."""
    t = np.asarray(t, dtype=float)
    x = (t - t0)
    # guard the exponentials against overflow
    fall = np.exp(-np.clip(x / t_fall, -50.0, 50.0))
    rise = 1.0 + np.exp(-np.clip(x / t_rise, -50.0, 50.0))
    return A * fall / rise + B


def fit_bazin(series: BandSeries) -> Callable[[np.ndarray], np.ndarray] | None:
    """Least-squares Bazin fit (flux space).  Returns a callable or None."""
    from scipy.optimize import curve_fit

    t, flux, ferr = series.mjd, series.flux, series.fluxerr
    n = len(t)
    if n < MIN_BAND_FOR_FIT:
        return None
    B0 = float(np.median(flux))
    peak_i = int(np.argmax(flux))
    A0 = float(flux[peak_i] - B0)
    if A0 <= 0:
        A0 = float(np.ptp(flux)) or 1.0
    t0 = float(t[peak_i])
    span = float(np.ptp(t)) or 10.0
    p0 = [A0, t0, max(span / 10.0, 1.0), max(span / 3.0, 2.0), B0]
    frange = float(np.ptp(flux)) or abs(A0) or 1.0
    lower = [0.0, t.min() - span, 0.1, 0.5, B0 - 5 * frange]
    upper = [10 * abs(A0) + 10 * frange, t.max() + span, 200.0, 400.0, B0 + 5 * frange]
    # keep bounds ordered even in degenerate cases
    p0 = [min(max(p, lo), hi) for p, lo, hi in zip(p0, lower, upper)]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                bazin, t, flux, p0=p0, sigma=ferr, absolute_sigma=True,
                bounds=(lower, upper), maxfev=20000,
            )
    except Exception:
        return None
    if not np.all(np.isfinite(popt)):
        return None

    def model(tt: np.ndarray, _p=popt) -> np.ndarray:
        return bazin(tt, *_p)

    # sanity: prediction must be finite on the training grid
    if not np.all(np.isfinite(model(t))):
        return None
    return model


def fit_gp(series: BandSeries, *, seed: int) -> Callable[[np.ndarray], np.ndarray] | None:
    """Matern GP fit in a self-standardized flux space.  Returns callable or None.

    Standardizing flux (and shifting time) ourselves keeps the kernel matrix well
    conditioned and makes ``alpha`` (the per-point noise variance) commensurate
    with the kernel amplitude regardless of the raw nJy scale.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        ConstantKernel,
        Matern,
        WhiteKernel,
    )

    t, flux, ferr = series.mjd, series.flux, series.fluxerr
    if len(t) < MIN_BAND_FOR_GP:
        return None
    t0 = float(t.min())
    tc = (t - t0).reshape(-1, 1)
    fmean = float(np.mean(flux))
    fstd = float(np.std(flux))
    if not math.isfinite(fstd) or fstd <= 0:
        return None
    ys = (flux - fmean) / fstd
    alpha = np.clip((ferr / fstd) ** 2, 1e-4, None)
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2))
        * Matern(length_scale=10.0, length_scale_bounds=(1.0, 200.0), nu=1.5)
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3, 1e1))
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp = GaussianProcessRegressor(
                kernel=kernel, alpha=alpha, normalize_y=False,
                n_restarts_optimizer=1, random_state=seed,
            )
            gp.fit(tc, ys)
            check = gp.predict(tc)
    except Exception:
        return None
    if not np.all(np.isfinite(check)):
        return None

    def model(tt: np.ndarray, _gp=gp, _t0=t0, _m=fmean, _s=fstd) -> np.ndarray:
        tt = np.asarray(tt, dtype=float).reshape(-1, 1) - _t0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = _gp.predict(tt)
        return _s * pred + _m

    return model


def fit_band(series: BandSeries, *, seed: int) -> Callable[[np.ndarray], np.ndarray] | None:
    """Fit one band: GP first (if >=4 dets), Bazin fallback."""
    n = len(series.mjd)
    if n >= MIN_BAND_FOR_GP:
        model = fit_gp(series, seed=seed)
        if model is not None:
            return model
    return fit_bazin(series)


@dataclass
class ObjectFit:
    models: dict[str, Callable[[np.ndarray], np.ndarray]]
    peak_mjd: float
    t_min: float
    t_max: float

    @property
    def fitted_bands(self) -> tuple[str, ...]:
        return tuple(self.models.keys())


def fit_object(
    normalized: Sequence[dict[str, Any]],
    *,
    survey: str,
    seed: int,
) -> ObjectFit | None:
    """Fit every band of one source object.  Returns None when the object is
    unusable (too few detections, or no band could be fit)."""
    allowed = ZTF_RESAMPLE_BANDS if survey.upper() == "ZTF" else None
    series = collect_band_series(normalized, allowed_bands=allowed)
    total = sum(len(s.mjd) for s in series.values())
    if total < MIN_TOTAL_DETECTIONS:
        return None
    models: dict[str, Callable[[np.ndarray], np.ndarray]] = {}
    t_lo, t_hi = math.inf, -math.inf
    for i, band in enumerate(sorted(series)):
        model = fit_band(series[band], seed=seed + i)
        if model is None:
            continue
        models[band] = model
        t_lo = min(t_lo, float(series[band].mjd.min()))
        t_hi = max(t_hi, float(series[band].mjd.max()))
    if not models:
        return None
    # global peak: densely evaluate each fitted band, take the brightest epoch
    grid = np.linspace(t_lo, t_hi, 500)
    best_t, best_f = float(grid[0]), -math.inf
    for band, model in models.items():
        f = model(grid)
        j = int(np.argmax(f))
        if f[j] > best_f:
            best_f, best_t = float(f[j]), float(grid[j])
    return ObjectFit(models=models, peak_mjd=best_t, t_min=t_lo, t_max=t_hi)


# ── cadence templates ─────────────────────────────────────────────────────────
@dataclass
class CadenceTemplate:
    mjd: np.ndarray       # absolute template epochs, sorted
    band: list[str]       # per-epoch band
    fluxerr: np.ndarray   # per-epoch |flux error| magnitude (noise sigma)


def load_cadence_templates(cadence_dir: Path) -> list[CadenceTemplate]:
    """Read every LSST lightcurve JSON in ``cadence_dir`` into a cadence template
    (epoch times, bands, per-epoch flux-error magnitudes)."""
    templates: list[CadenceTemplate] = []
    for path in sorted(cadence_dir.glob("*.json")):
        try:
            raw = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(raw, list) or not raw:
            continue
        rows: list[tuple[float, str, float]] = []
        for det in raw:
            band = str(det.get("band") or "").lower()
            if band not in _BAND_IDX:
                continue
            mjd = det.get("mjd") or det.get("midpointMjdTai")
            ferr = det.get("psfFluxErr")
            if ferr is None:
                ferr = det.get("fluxerr")
            try:
                mjd = float(mjd)
                ferr = abs(float(ferr))
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(mjd) and math.isfinite(ferr) and ferr > 0):
                continue
            rows.append((mjd, band, ferr))
        if not rows:
            continue
        rows.sort(key=lambda r: r[0])
        templates.append(
            CadenceTemplate(
                mjd=np.array([r[0] for r in rows], dtype=float),
                band=[r[1] for r in rows],
                fluxerr=np.array([r[2] for r in rows], dtype=float),
            )
        )
    return templates


def restrict_template(template: CadenceTemplate, bands: Sequence[str]) -> CadenceTemplate | None:
    """Keep only the template epochs whose band is in ``bands`` (the source's
    fitted bands).  Returns None if nothing remains."""
    keep = [i for i, b in enumerate(template.band) if b in bands]
    if not keep:
        return None
    idx = np.array(keep, dtype=int)
    return CadenceTemplate(
        mjd=template.mjd[idx],
        band=[template.band[i] for i in keep],
        fluxerr=template.fluxerr[idx],
    )


# ── resampling ────────────────────────────────────────────────────────────────
def _make_detection(
    *,
    object_id: str,
    band: str,
    mjd: float,
    flux: float,
    fluxerr: float,
    is_negative: bool,
) -> dict[str, Any]:
    """Build one augmented detection in the raw LSST schema, then normalize it so
    the written dict is byte-identical to a cached normalized LSST detection."""
    snr = abs(flux) / fluxerr if fluxerr > 0 else 0.0
    raw = {
        "band": band,
        "band_map": dict(_BAND_MAP),
        "mjd": float(mjd),
        "psfFlux": float(flux),
        "psfFluxErr": float(fluxerr),
        "snr": float(snr),
        "reliability": 1.0,
        "isNegative": bool(is_negative),
        "diaObjectId": object_id,
        "oid": object_id,
        "survey": "LSST",
        "survey_id": "lsst",
        "_source": LABEL_SOURCE,
    }
    return normalize_detection(raw, survey="LSST")


def resample_once(
    fit: ObjectFit,
    template: CadenceTemplate,
    *,
    object_id: str,
    rng: np.random.Generator,
    time_dilate: float | None,
    peak_jitter_days: float,
) -> list[dict[str, Any]]:
    """Produce ONE augmented lightcurve (list of normalized detections)."""
    rel = template.mjd - template.mjd[0]           # template phase, t0 = 0
    span = float(rel[-1]) if len(rel) > 1 else 1.0
    phase_frac = float(rng.uniform(0.15, 0.5))     # where the peak lands in span
    jitter = float(rng.normal(0.0, peak_jitter_days))
    stretch = 1.0
    if time_dilate:
        stretch = 1.0 + float(rng.uniform(-time_dilate, time_dilate))
    # source-time of each template epoch, peak-aligned + jittered + stretched
    src_t = fit.peak_mjd + (rel - phase_frac * span) * stretch + jitter

    dets: list[dict[str, Any]] = []
    model_flux = np.empty(len(src_t))
    for k, band in enumerate(template.band):
        model = fit.models[band]
        mf = float(model(np.array([src_t[k]]))[0])
        model_flux[k] = mf
        ferr = float(template.fluxerr[k])
        sampled = mf + float(rng.normal(0.0, ferr))
        # a positive detection needs flux > 0 AND >= ~1 sigma; else it is a
        # negative / non-detection token (mirrors LSST isNegative semantics and
        # feeds the v11 negative-token channels).
        is_neg = not (sampled > 0.0 and sampled >= ferr)
        dets.append(
            _make_detection(
                object_id=object_id, band=band, mjd=float(src_t[k]),
                flux=sampled, fluxerr=ferr, is_negative=is_neg,
            )
        )

    # guarantee >=1 positive detection so the LSST tokenizer yields >0 tokens
    if not any(d.get("is_positive") for d in dets):
        k = int(np.argmax(model_flux))
        ferr = float(template.fluxerr[k])
        forced_flux = model_flux[k] if model_flux[k] > 0 else abs(model_flux[k]) + ferr
        forced_flux = max(forced_flux, ferr * 1.5)
        dets[k] = _make_detection(
            object_id=object_id, band=template.band[k], mjd=float(src_t[k]),
            flux=float(forced_flux), fluxerr=ferr, is_negative=False,
        )
    return dets


def augment_object(
    fit: ObjectFit,
    templates: Sequence[CadenceTemplate],
    *,
    object_id: str,
    k_per_object: int,
    seed_seq: np.random.SeedSequence,
    time_dilate: float | None,
    peak_jitter_days: float,
) -> list[list[dict[str, Any]]]:
    """Produce ``k_per_object`` augmented lightcurves for one fitted source."""
    bands = set(fit.fitted_bands)
    usable = [t for t in (restrict_template(t, bands) for t in templates) if t is not None]
    if not usable:
        return []
    children = seed_seq.spawn(k_per_object)
    out: list[list[dict[str, Any]]] = []
    for child in children:
        rng = np.random.default_rng(child)
        # a few template draws so a degenerate (0-positive) draw can be retried
        best: list[dict[str, Any]] | None = None
        best_pos = -1
        for _ in range(8):
            template = usable[int(rng.integers(len(usable)))]
            dets = resample_once(
                fit, template, object_id=object_id, rng=rng,
                time_dilate=time_dilate, peak_jitter_days=peak_jitter_days,
            )
            npos = sum(1 for d in dets if d.get("is_positive"))
            if npos > best_pos:
                best, best_pos = dets, npos
            if npos >= 1:
                break
        if best is not None:
            out.append(best)
    return out


# ── truth loading ─────────────────────────────────────────────────────────────
def load_spec_truth(truth_path: Path):
    import pandas as pd

    df = pd.read_parquet(truth_path)
    required = {"object_id", "final_class_ternary", "label_quality"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"truth parquet missing columns: {sorted(missing)}")
    spec = df[df["label_quality"] == "spectroscopic"].copy()
    spec = spec[spec["final_class_ternary"].notna()]
    return spec


def find_lightcurve(lc_dirs: Sequence[Path], object_id: str) -> Path | None:
    for d in lc_dirs:
        p = d / f"{object_id}.json"
        if p.exists():
            return p
    return None


# ── tokenizer round-trip guard ────────────────────────────────────────────────
def assert_tokenizes(dets: list[dict[str, Any]]) -> int:
    """Round-trip one augmented lightcurve through the v11 tokenizer; return the
    number of tokens (must be > 0).  Raises AssertionError on failure."""
    from debass_meta.features.sequence_dataset import sequence_arrays

    cont, bands = sequence_arrays(dets, schema="v11")
    assert cont.ndim == 2 and cont.shape[0] > 0, "augmented LC produced 0 tokens"
    assert cont.shape[1] == 11, f"expected 11-dim v11 tensor, got {cont.shape[1]}"
    assert np.all(np.isfinite(cont)), "non-finite values in tokenized tensor"
    assert len(bands) == cont.shape[0], "band/continuous length mismatch"
    return int(cont.shape[0])


# ── driver ────────────────────────────────────────────────────────────────────
def run(
    *,
    truth_path: Path,
    lc_dirs: Sequence[Path],
    cadence_dir: Path,
    out_dir: Path,
    k_per_object: int,
    seed: int,
    limit: int | None,
    time_dilate: float | None,
    peak_jitter_days: float,
) -> dict[str, Any]:
    import pandas as pd

    spec = load_spec_truth(truth_path)
    if limit is not None:
        spec = spec.head(limit)

    templates = load_cadence_templates(cadence_dir)
    if not templates:
        raise ValueError(f"no cadence templates found in {cadence_dir}")

    out_lc_dir = out_dir / "lightcurves"
    out_lc_dir.mkdir(parents=True, exist_ok=True)

    # deterministic per-object RNG streams
    rows = list(spec.itertuples(index=False))
    root = np.random.SeedSequence(seed)
    obj_seeds = root.spawn(len(rows))

    truth_rows: list[dict[str, Any]] = []
    n_fitted = n_skipped_nolc = n_skipped_fit = n_files = 0
    skipped_ids: list[str] = []

    for row, obj_seq in zip(rows, obj_seeds):
        source_id = str(getattr(row, "object_id"))
        ternary = getattr(row, "final_class_ternary")
        lc_path = find_lightcurve(lc_dirs, source_id)
        if lc_path is None:
            n_skipped_nolc += 1
            skipped_ids.append(source_id)
            continue
        try:
            raw = json.loads(lc_path.read_text())
        except Exception:
            n_skipped_nolc += 1
            skipped_ids.append(source_id)
            continue
        if not isinstance(raw, list) or not raw:
            n_skipped_nolc += 1
            skipped_ids.append(source_id)
            continue
        survey = infer_source_survey(raw)
        normalized = normalize_lightcurve(raw, survey=survey)
        # fit seed derived deterministically from this object's stream
        fit_seed = int(obj_seq.generate_state(1)[0] % (2**31 - 1))
        fit = fit_object(normalized, survey=survey, seed=fit_seed)
        if fit is None:
            n_skipped_fit += 1
            skipped_ids.append(source_id)
            continue

        augmented = augment_object(
            fit, templates, object_id=source_id, k_per_object=k_per_object,
            seed_seq=obj_seq, time_dilate=time_dilate, peak_jitter_days=peak_jitter_days,
        )
        if not augmented:
            n_skipped_fit += 1
            skipped_ids.append(source_id)
            continue

        n_fitted += 1
        for k, dets in enumerate(augmented):
            aug_id = f"{source_id}_aug{k:02d}"
            # rewrite the object identifiers to the augmented id
            for d in dets:
                d["diaObjectId"] = aug_id
                d["oid"] = aug_id
            # write-time tokenizer round-trip guard (fail fast if schema drifts)
            assert_tokenizes(dets)
            (out_lc_dir / f"{aug_id}.json").write_text(json.dumps(dets))
            n_files += 1
            truth_rows.append(
                {
                    "object_id": aug_id,
                    "source_object_id": source_id,
                    "final_class_ternary": ternary,
                    "label_quality": LABEL_QUALITY,
                    "label_source": LABEL_SOURCE,
                    "survey": survey,
                }
            )

    truth_df = pd.DataFrame(
        truth_rows,
        columns=[
            "object_id", "source_object_id", "final_class_ternary",
            "label_quality", "label_source", "survey",
        ],
    )
    truth_out = out_dir / "truth_augmented.parquet"
    truth_df.to_parquet(truth_out, index=False)

    return {
        "n_source_spec": len(rows),
        "n_fitted": n_fitted,
        "n_skipped_no_lc": n_skipped_nolc,
        "n_skipped_fit": n_skipped_fit,
        "n_files_written": n_files,
        "truth_parquet": str(truth_out),
        "out_lc_dir": str(out_lc_dir),
        "n_cadence_templates": len(templates),
        "skipped_ids": skipped_ids,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--truth", required=True, type=Path,
                   help="truth parquet (object_id/final_class_ternary/label_quality)")
    p.add_argument("--lc-dir", action="append", required=True, type=Path, dest="lc_dirs",
                   help="source lightcurve dir (repeatable)")
    p.add_argument("--cadence-dir", required=True, type=Path,
                   help="dir of REAL LSST lightcurve JSONs to draw cadence templates from")
    p.add_argument("--per-object", type=int, default=20, dest="k_per_object",
                   help="K augmented lightcurves per source object (default 20)")
    p.add_argument("--out-dir", type=Path, default=Path("data/augmented_lsst"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="smoke: cap #source objects")
    p.add_argument("--time-dilate", type=float, default=None,
                   help="max fractional 1+dz stretch about the peak (e.g. 0.1)")
    p.add_argument("--peak-jitter-days", type=float, default=DEFAULT_PEAK_JITTER_DAYS,
                   help="stddev (days) of the peak-alignment jitter")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    stats = run(
        truth_path=args.truth,
        lc_dirs=args.lc_dirs,
        cadence_dir=args.cadence_dir,
        out_dir=args.out_dir,
        k_per_object=args.k_per_object,
        seed=args.seed,
        limit=args.limit,
        time_dilate=args.time_dilate,
        peak_jitter_days=args.peak_jitter_days,
    )
    printable = {k: v for k, v in stats.items() if k != "skipped_ids"}
    print(json.dumps(printable, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
