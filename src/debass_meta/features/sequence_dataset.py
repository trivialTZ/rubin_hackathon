"""Leak-free per-detection sequence tensors for the v9 sequence encoder.

Turns a cached lightcurve into a causal per-detection feature matrix that the
GRU encoder (:mod:`debass_meta.models.seq_encoder`) consumes.  The detection
list is preprocessed EXACTLY like the gold builder
(:func:`debass_meta.features.lightcurve.extract_features_at_each_epoch`):
normalize → keep positive detections → sort by MJD → truncate.  Every feature
of detection *k* is a function of detections 1..k only (dt/dmag look strictly
backwards; the magnitude anchor is the FIRST detection, which is inside every
prefix), so the row-k feature vector is identical no matter how many future
detections exist — the encoder's prefix state at step k therefore inherits the
no-leakage proof.  This is asserted by ``tests/test_seq_encoder_v9.py``.

No torch imports here: this module is pure numpy so the gold/scoring side can
load it without a torch installation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from debass_meta.features.lightcurve import _ensure_normalized

# Band vocabulary for the embedding layer (index 6 = unknown/other).
BAND_TO_IDX = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5}
UNKNOWN_BAND_IDX = 6
N_BANDS = 7

# Continuous per-detection features, in order.
SEQ_CONTINUOUS_FIELDS = (
    "log_dt_prev",        # log1p(days since previous detection); 0 for the first
    "dmag_prev",          # mag_k - mag_{k-1}; 0 for the first / non-finite pairs
    "mag_minus_first",    # mag_k - mag_1 (causal anchor: first det is in every prefix)
    "magerr",             # photometric error (clipped); fill 0.2 when missing
    "log_snr",            # log1p(snr) (clipped); fill 0 when missing
    "is_first",           # 1.0 for the first detection
    "missing_magerr",     # 1.0 when magerr was missing/non-finite
    "missing_snr",        # 1.0 when snr was missing/non-finite
    "survey_is_lsst",     # 1.0 for LSST detections
)
SEQ_CONTINUOUS_DIM = len(SEQ_CONTINUOUS_FIELDS)

# Indices of the z-scored dims (flags stay raw).
_ZSCORE_DIMS = (0, 1, 2, 3, 4)

# Per-detection survey flag dim (keys the optional per-survey normalization).
_SURVEY_DIM = SEQ_CONTINUOUS_FIELDS.index("survey_is_lsst")
# A survey needs at least this many corpus sequences to earn dedicated
# z-score statistics; below it the pooled stats are used (v10 fallback).
PER_SURVEY_MIN_SEQUENCES = 200

_MAGERR_FILL = 0.2
_MAGERR_CLIP = 1.5
_LOG_SNR_CLIP = 8.0


def truncated_positive_detections(
    detections: list[dict[str, Any]],
    *,
    max_len: int | None = None,
) -> list[dict[str, Any]]:
    """Replicate the gold builder's preprocessing (normalize → positive filter
    → MJD sort → truncate).  Must stay in lockstep with
    ``_truncated_detection_lists`` in scripts/build_snapshots_fusion.py."""
    ndets = [_ensure_normalized(d) for d in detections]
    pos = [d for d in ndets if d.get("is_positive", True)]
    if not pos:
        pos = ndets
    pos.sort(key=lambda d: d.get("mjd") or 0)
    if max_len is not None:
        pos = pos[:max_len]
    return pos


def _finite(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def sequence_arrays(
    detections: list[dict[str, Any]],
    *,
    max_len: int | None = None,
    pre_truncated: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (continuous (L, 9) float32, band_idx (L,) int64) for one object.

    ``pre_truncated=True`` skips preprocessing when the caller already holds
    the truncated/normalized prefix (the gold builder's own lists).
    """
    dets = detections if pre_truncated else truncated_positive_detections(
        detections, max_len=max_len
    )
    L = len(dets)
    cont = np.zeros((L, SEQ_CONTINUOUS_DIM), dtype=np.float32)
    bands = np.full((L,), UNKNOWN_BAND_IDX, dtype=np.int64)
    prev_mjd: float | None = None
    prev_mag: float | None = None
    first_mag: float | None = None
    for k, det in enumerate(dets):
        mjd = _finite(det.get("mjd"))
        mag = _finite(det.get("mag"))
        magerr = _finite(det.get("magerr"))
        snr = _finite(det.get("snr"))
        band = str(det.get("band") or "").lower()
        bands[k] = BAND_TO_IDX.get(band, UNKNOWN_BAND_IDX)
        if first_mag is None and mag is not None:
            first_mag = mag

        dt = 0.0
        if prev_mjd is not None and mjd is not None:
            dt = max(mjd - prev_mjd, 0.0)
        dmag = 0.0
        if prev_mag is not None and mag is not None:
            dmag = mag - prev_mag
        anchor = 0.0
        if first_mag is not None and mag is not None:
            anchor = mag - first_mag

        cont[k, 0] = math.log1p(dt)
        cont[k, 1] = float(np.clip(dmag, -5.0, 5.0))
        cont[k, 2] = float(np.clip(anchor, -8.0, 8.0))
        cont[k, 3] = float(np.clip(magerr, 0.0, _MAGERR_CLIP)) if magerr is not None else _MAGERR_FILL
        cont[k, 4] = float(np.clip(math.log1p(max(snr, 0.0)), 0.0, _LOG_SNR_CLIP)) if snr is not None else 0.0
        cont[k, 5] = 1.0 if k == 0 else 0.0
        cont[k, 6] = 1.0 if magerr is None else 0.0
        cont[k, 7] = 1.0 if snr is None else 0.0
        cont[k, 8] = 1.0 if str(det.get("survey") or "").upper() == "LSST" else 0.0

        if mjd is not None:
            prev_mjd = mjd
        if mag is not None:
            prev_mag = mag
    return cont, bands


def load_object_sequence(
    lc_dir: Path,
    object_id: str,
    *,
    max_len: int | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load + tensorize one object's cached lightcurve; None when absent/empty.

    Imports the gold loader lazily to keep this module import-light.
    """
    from debass_meta.ingest.gold import _load_lightcurve, _resolve_lightcurve_path

    lc_path, _source = _resolve_lightcurve_path(lc_dir, object_id=object_id, associations=None)
    if lc_path is None:
        return None
    detections = _load_lightcurve(lc_path)
    if not detections:
        return None
    cont, bands = sequence_arrays(detections, max_len=max_len)
    if len(cont) == 0:
        return None
    return cont, bands


def sequence_survey(cont: np.ndarray) -> str:
    """Classify one sequence: "lsst" when any detection carries the LSST
    flag, else "ztf" (shared by the --surveys filters in the v10 trainers)."""
    if len(cont) and bool(np.max(cont[:, _SURVEY_DIM]) > 0.5):
        return "lsst"
    return "ztf"


def _fit_moments(stacked: np.ndarray) -> tuple[list[float], list[float]]:
    mean = [0.0] * SEQ_CONTINUOUS_DIM
    std = [1.0] * SEQ_CONTINUOUS_DIM
    for d in _ZSCORE_DIMS:
        col = stacked[:, d]
        col = col[np.isfinite(col)]
        if len(col) >= 2:
            mean[d] = float(np.mean(col))
            s = float(np.std(col))
            std[d] = s if s > 1e-6 else 1.0
    return mean, std


@dataclass
class NormStats:
    """Z-score statistics for the continuous dims, fit on the SSL corpus and
    frozen into the encoder artifact (applied identically at export time).

    ``per_survey`` (v10, optional) keys dedicated statistics by survey
    ("ztf" / "lsst"), dispatched row-wise on the per-detection
    ``survey_is_lsst`` flag.  ZTF and LSST differ in cadence (log_dt_prev)
    and depth (magerr, log_snr); pooled z-scoring absorbs that only via the
    survey flag.  A survey missing from ``per_survey`` (too few corpus
    sequences at fit time, or a pre-v10 artifact) falls back to the pooled
    statistics — an empty dict is byte-identical to pre-v10 behaviour."""

    mean: list[float] = field(default_factory=lambda: [0.0] * SEQ_CONTINUOUS_DIM)
    std: list[float] = field(default_factory=lambda: [1.0] * SEQ_CONTINUOUS_DIM)
    per_survey: dict[str, dict[str, list[float]]] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        arrays: list[np.ndarray],
        *,
        per_survey: bool = False,
        min_survey_sequences: int = PER_SURVEY_MIN_SEQUENCES,
    ) -> "NormStats":
        arrays = [a for a in arrays if len(a)]
        stacked = np.concatenate(arrays, axis=0)
        mean, std = _fit_moments(stacked)
        out = cls(mean=mean, std=std)
        if not per_survey:
            return out
        row_is_lsst = stacked[:, _SURVEY_DIM] > 0.5
        for survey, want_lsst in (("ztf", False), ("lsst", True)):
            n_seqs = sum(
                1 for a in arrays if np.any((a[:, _SURVEY_DIM] > 0.5) == want_lsst)
            )
            rows = stacked[row_is_lsst == want_lsst]
            if n_seqs < min_survey_sequences or len(rows) < 2:
                continue  # pooled fallback for this survey
            s_mean, s_std = _fit_moments(rows)
            out.per_survey[survey] = {"mean": s_mean, "std": s_std}
        return out

    def apply(self, cont: np.ndarray) -> np.ndarray:
        out = cont.astype(np.float32, copy=True)
        if not self.per_survey:
            for d in _ZSCORE_DIMS:
                out[:, d] = (out[:, d] - self.mean[d]) / self.std[d]
            return out
        is_lsst = cont[:, _SURVEY_DIM] > 0.5
        for survey, sel in (("ztf", ~is_lsst), ("lsst", is_lsst)):
            if not np.any(sel):
                continue
            entry = self.per_survey.get(survey)
            mean = entry["mean"] if entry else self.mean
            std = entry["std"] if entry else self.std
            for d in _ZSCORE_DIMS:
                out[sel, d] = (out[sel, d] - mean[d]) / std[d]
        return out

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"mean": list(self.mean), "std": list(self.std)}
        if self.per_survey:
            payload["per_survey"] = {
                survey: {"mean": list(entry["mean"]), "std": list(entry["std"])}
                for survey, entry in self.per_survey.items()
            }
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "NormStats":
        per_survey = {
            str(survey): {
                "mean": [float(v) for v in entry["mean"]],
                "std": [float(v) for v in entry["std"]],
            }
            for survey, entry in (payload.get("per_survey") or {}).items()
        }
        return cls(mean=[float(v) for v in payload["mean"]],
                   std=[float(v) for v in payload["std"]],
                   per_survey=per_survey)


def save_norm_stats(stats: NormStats, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats.to_json(), indent=1))


def load_norm_stats(path: Path) -> NormStats:
    return NormStats.from_json(json.loads(path.read_text()))
