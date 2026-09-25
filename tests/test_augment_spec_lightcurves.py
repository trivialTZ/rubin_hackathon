"""Tests for WP4 GP/parametric lightcurve augmentation (scripts/augment_spec_lightcurves.py).

Covers:
  * GP + Bazin fits recover a synthetic Bazin curve (corr > 0.95 on a dense grid)
  * augmented JSONs tokenize through the v11 sequence schema with > 0 tokens
  * truth_augmented.parquet rows are well-formed (label tier locked to augmented)
  * per-object K is respected
  * a ZTF source only ever emits g/r epochs (never fabricated bands)
  * a --seed run is bit-reproducible
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for _p in (str(_SRC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# import the script module by path (scripts/ is not a package)
_spec = importlib.util.spec_from_file_location(
    "augment_spec_lightcurves", _ROOT / "scripts" / "augment_spec_lightcurves.py"
)
aug = importlib.util.module_from_spec(_spec)
sys.modules["augment_spec_lightcurves"] = aug  # needed for dataclass annotation resolution
_spec.loader.exec_module(aug)

from debass_meta.features.sequence_dataset import sequence_arrays  # noqa: E402


# ── synthetic data helpers ────────────────────────────────────────────────────
def _bazin_series(seed: int = 0, n: int = 30, noise: float = 20.0) -> aug.BandSeries:
    rng = np.random.default_rng(seed)
    t0 = 55000.0
    true = dict(A=1000.0, t0=t0 + 20.0, t_rise=3.0, t_fall=15.0, B=50.0)
    t = np.sort(rng.uniform(t0, t0 + 60.0, size=n))
    clean = aug.bazin(t, **true)
    flux = clean + rng.normal(0.0, noise, size=n)
    ferr = np.full(n, noise)
    return aug.BandSeries(mjd=t, flux=flux, fluxerr=ferr), true


def _dense_corr(model, true: dict) -> float:
    grid = np.linspace(55000.0, 55060.0, 400)
    fitted = np.asarray(model(grid), dtype=float)
    truth = aug.bazin(grid, **true)
    return float(np.corrcoef(fitted, truth)[0, 1])


def _lsst_source_json(n_per_band: int = 12, bands=("g", "r", "i", "z"), seed: int = 1):
    """A real-schema LSST source lightcurve (list of raw diaSource dicts)."""
    rng = np.random.default_rng(seed)
    dets = []
    t0 = 61000.0
    for bi, band in enumerate(bands):
        t = np.sort(rng.uniform(t0, t0 + 60.0, size=n_per_band))
        peak = t0 + 20.0 + bi
        flux = 5e5 * np.exp(-((t - peak) ** 2) / (2 * 12.0**2)) + 1e3
        flux = flux + rng.normal(0, 2e3, size=n_per_band)
        for tt, ff in zip(t, flux):
            dets.append(
                {
                    "band": band,
                    "mjd": float(tt),
                    "psfFlux": float(ff),
                    "psfFluxErr": 2e3,
                    "isNegative": False,
                    "snr": float(abs(ff) / 2e3),
                    "reliability": 0.9,
                    "diaObjectId": 999,
                }
            )
    return dets


def _ztf_source_json(n_per_band: int = 12, seed: int = 2):
    """A real-schema ZTF source lightcurve (fid/magpsf), with a g,r,i mix."""
    rng = np.random.default_rng(seed)
    dets = []
    t0 = 58000.0
    for fid in (1, 2, 3):  # g, r, i
        t = np.sort(rng.uniform(t0, t0 + 60.0, size=n_per_band))
        peak = t0 + 20.0
        mag = 18.0 + 1.5 * ((t - peak) / 20.0) ** 2  # brightens then fades
        for tt, mm in zip(t, mag):
            dets.append(
                {
                    "fid": fid,
                    "mjd": float(tt),
                    "magpsf": float(mm),
                    "sigmapsf": 0.05,
                    "isdiffpos": "t",
                    "rb": 0.9,
                    "drb": 0.9,
                }
            )
    return dets


def _cadence_files(tmp: Path, n: int = 6, seed: int = 5) -> Path:
    """Write a handful of real-schema LSST cadence templates (ugrizy)."""
    rng = np.random.default_rng(seed)
    d = tmp / "cadence"
    d.mkdir(parents=True, exist_ok=True)
    for j in range(n):
        dets = []
        t0 = 61500.0 + 10 * j
        n_ep = int(rng.integers(8, 25))
        for _ in range(n_ep):
            band = str(rng.choice(list("ugrizy")))
            dets.append(
                {
                    "band": band,
                    "mjd": float(t0 + rng.uniform(0, 55)),
                    "psfFlux": float(rng.normal(1e4, 5e3)),
                    "psfFluxErr": float(abs(rng.normal(1500, 300)) + 200),
                    "isNegative": False,
                }
            )
        (d / f"cad{j}.json").write_text(json.dumps(dets))
    return d


# ── tests ─────────────────────────────────────────────────────────────────────
def test_bazin_fit_recovers_synthetic():
    series, true = _bazin_series(seed=0)
    model = aug.fit_bazin(series)
    assert model is not None
    assert _dense_corr(model, true) > 0.95


def test_gp_fit_recovers_synthetic():
    series, true = _bazin_series(seed=3, n=40, noise=15.0)
    model = aug.fit_gp(series, seed=7)
    assert model is not None
    # GP is only trusted where it interpolates; correlation over the observed
    # window recovers the underlying Bazin shape.
    grid = np.linspace(series.mjd.min(), series.mjd.max(), 400)
    fitted = np.asarray(model(grid), dtype=float)
    truth = aug.bazin(grid, **true)
    assert float(np.corrcoef(fitted, truth)[0, 1]) > 0.95


def test_fit_band_prefers_gp_then_bazin():
    # >=4 points → a fit exists; 3 points → Bazin; <3 → None
    series, _ = _bazin_series(seed=1, n=20)
    assert aug.fit_band(series, seed=0) is not None
    three = aug.BandSeries(series.mjd[:3], series.flux[:3], series.fluxerr[:3])
    assert aug.fit_band(three, seed=0) is not None  # Bazin fallback
    two = aug.BandSeries(series.mjd[:2], series.flux[:2], series.fluxerr[:2])
    assert aug.fit_band(two, seed=0) is None  # too few → skip band


def test_object_with_too_few_detections_is_skipped():
    from debass_meta.features.detection import normalize_lightcurve

    dets = _lsst_source_json(n_per_band=1, bands=("g", "r"))  # 2 total
    norm = normalize_lightcurve(dets, survey="LSST")
    assert aug.fit_object(norm, survey="LSST", seed=0) is None


def _write_source_and_truth(tmp: Path, entries):
    """entries: list of (object_id, ternary, raw_dets). Returns (truth_path, lc_dir)."""
    import pandas as pd

    lc_dir = tmp / "src_lc"
    lc_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for oid, ternary, dets in entries:
        (lc_dir / f"{oid}.json").write_text(json.dumps(dets))
        rows.append(
            {"object_id": oid, "final_class_ternary": ternary, "label_quality": "spectroscopic"}
        )
    # add a decoy non-spec row that must be ignored
    rows.append(
        {"object_id": "decoy", "final_class_ternary": "snia", "label_quality": "context"}
    )
    truth_path = tmp / "truth.parquet"
    pd.DataFrame(rows).to_parquet(truth_path, index=False)
    return truth_path, lc_dir


def test_end_to_end_augment_lsst(tmp_path):
    import pandas as pd

    entries = [
        ("obj_a", "snia", _lsst_source_json(seed=10)),
        ("obj_b", "nonIa_snlike", _lsst_source_json(seed=11)),
    ]
    truth_path, lc_dir = _write_source_and_truth(tmp_path, entries)
    cadence = _cadence_files(tmp_path)
    out_dir = tmp_path / "out"
    K = 5
    stats = aug.run(
        truth_path=truth_path, lc_dirs=[lc_dir], cadence_dir=cadence,
        out_dir=out_dir, k_per_object=K, seed=42, limit=None,
        time_dilate=0.05, peak_jitter_days=3.0,
    )
    # both objects fit; K files each; decoy ignored
    assert stats["n_fitted"] == 2
    assert stats["n_files_written"] == 2 * K

    # per-object K respected
    files = sorted((out_dir / "lightcurves").glob("*.json"))
    a_files = [f for f in files if f.name.startswith("obj_a_aug")]
    b_files = [f for f in files if f.name.startswith("obj_b_aug")]
    assert len(a_files) == K and len(b_files) == K
    assert {f.stem for f in a_files} == {f"obj_a_aug{k:02d}" for k in range(K)}

    # every augmented JSON tokenizes via v11 with > 0 tokens
    for f in files:
        dets = json.loads(f.read_text())
        cont, bands = sequence_arrays(dets, schema="v11")
        assert cont.shape[0] > 0
        assert cont.shape[1] == 11
        assert np.all(np.isfinite(cont))

    # truth rows well-formed, tier locked to augmented
    truth = pd.read_parquet(out_dir / "truth_augmented.parquet")
    assert len(truth) == 2 * K
    assert set(truth["label_quality"]) == {"augmented"}
    assert set(truth["label_source"]) == {"gp_augment_v1"}
    assert set(truth["final_class_ternary"]) == {"snia", "nonIa_snlike"}
    assert set(truth["source_object_id"]) == {"obj_a", "obj_b"}
    assert truth["object_id"].is_unique


def test_ztf_source_only_emits_g_r(tmp_path):
    entries = [("ztf_x", "snia", _ztf_source_json(seed=20))]
    truth_path, lc_dir = _write_source_and_truth(tmp_path, entries)
    cadence = _cadence_files(tmp_path, seed=9)
    out_dir = tmp_path / "out_ztf"
    stats = aug.run(
        truth_path=truth_path, lc_dirs=[lc_dir], cadence_dir=cadence,
        out_dir=out_dir, k_per_object=6, seed=1, limit=None,
        time_dilate=None, peak_jitter_days=2.0,
    )
    assert stats["n_fitted"] == 1
    seen_bands = set()
    for f in (out_dir / "lightcurves").glob("*.json"):
        for det in json.loads(f.read_text()):
            seen_bands.add(det["band"])
    assert seen_bands, "no epochs emitted"
    assert seen_bands <= {"g", "r"}, f"ZTF source fabricated bands: {seen_bands}"


def test_seed_reproducible(tmp_path):
    entries = [("obj_a", "snia", _lsst_source_json(seed=10))]
    truth_path, lc_dir = _write_source_and_truth(tmp_path, entries)
    cadence = _cadence_files(tmp_path)

    def _one(out):
        aug.run(
            truth_path=truth_path, lc_dirs=[lc_dir], cadence_dir=cadence,
            out_dir=out, k_per_object=4, seed=123, limit=None,
            time_dilate=0.1, peak_jitter_days=3.0,
        )
        return sorted((out / "lightcurves").glob("*.json"))

    a = _one(tmp_path / "r1")
    b = _one(tmp_path / "r2")
    assert [p.name for p in a] == [p.name for p in b]
    for pa, pb in zip(a, b):
        assert pa.read_text() == pb.read_text()
