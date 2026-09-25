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

# ── v11 schema (opt-in) ──────────────────────────────────────────────────────
# The nine v9 channels PLUS two negative-detection channels.  v11 tokenizes the
# negatives that fall in the epoch window (t ≤ t(Nth positive)) instead of
# dropping them: fading tails / faint variables are signal for the "other"
# class, and the signed difference flux carries information a magnitude-only
# encoding throws away (a negative-flux detection has no finite magnitude).
# The first nine channels are computed by IDENTICAL code to schema="v9", so a
# v11 sequence over the positive-only prefix reduces to the v9 tensor plus two
# appended columns — the versioning (schema= + seq_schema artifact meta) keeps
# deployed v9/v10 encoders (cont_dim=9 persisted) byte-safe.
SEQ_CONTINUOUS_FIELDS_V11 = SEQ_CONTINUOUS_FIELDS + (
    "is_negative",        # 1.0 when the token is a negative (non-)detection
    "signed_flux",        # asinh(signed diff flux); sign = brighten(+)/fade(−)
)
SEQ_CONTINUOUS_DIM_V11 = len(SEQ_CONTINUOUS_FIELDS_V11)

SEQ_SCHEMAS = ("v9", "v11")
_IS_NEGATIVE_DIM = SEQ_CONTINUOUS_FIELDS_V11.index("is_negative")   # 9
_SIGNED_FLUX_DIM = SEQ_CONTINUOUS_FIELDS_V11.index("signed_flux")   # 10


def _cont_dim_for_schema(schema: str) -> int:
    if schema == "v11":
        return SEQ_CONTINUOUS_DIM_V11
    if schema == "v9":
        return SEQ_CONTINUOUS_DIM
    raise ValueError(f"unknown sequence schema {schema!r} (expected one of {SEQ_SCHEMAS})")


# Indices of the z-scored dims (flags stay raw), keyed by continuous width so
# a v9 NormStats (9-dim mean/std) and a v11 NormStats (11-dim) each pick the
# right columns.  v9 keeps its exact historical set — dim 9 (is_negative) is a
# flag (raw) and dim 10 (signed_flux) IS z-scored.
_ZSCORE_DIMS = (0, 1, 2, 3, 4)                          # v9 (kept for back-compat)
_ZSCORE_DIMS_V11 = (0, 1, 2, 3, 4, _SIGNED_FLUX_DIM)
_ZSCORE_DIMS_BY_DIM = {
    SEQ_CONTINUOUS_DIM: _ZSCORE_DIMS,
    SEQ_CONTINUOUS_DIM_V11: _ZSCORE_DIMS_V11,
}


def _zscore_dims_for(dim: int) -> tuple[int, ...]:
    """Z-scored continuous dims for a `dim`-wide sequence (pooled fallback to
    the v9 set for any unexpected width)."""
    return _ZSCORE_DIMS_BY_DIM.get(dim, _ZSCORE_DIMS)


# Per-detection survey flag dim (keys the optional per-survey normalization).
# Index 8 in BOTH schemas — v11 appends after it, so per-survey routing is
# schema-agnostic.
_SURVEY_DIM = SEQ_CONTINUOUS_FIELDS.index("survey_is_lsst")
# A survey needs at least this many corpus sequences to earn dedicated
# z-score statistics; below it the pooled stats are used (v10 fallback).
PER_SURVEY_MIN_SEQUENCES = 200

_MAGERR_FILL = 0.2
_MAGERR_CLIP = 1.5
_LOG_SNR_CLIP = 8.0


def _mjd_key(det: dict[str, Any]) -> float:
    return det.get("mjd") or 0


def _local_truncated_windows(
    detections: list[dict[str, Any]],
    *,
    survey: str,
    max_n_det: int | None,
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """Fallback replica of P2's canonical ``truncated_detection_windows`` used
    ONLY while P2 has not yet landed the single implementation (parallel dev;
    P5→P2 dependency).  Matches the pinned contract exactly:

    * per epoch N: ``(pos_prefix, full_window)``;
    * ``pos_prefix`` = first N positive dets (MJD-sorted).  For ``survey`` !=
      ``"lsst"`` an all-negative lightcurve falls back to the negatives
      (ZTF-identical to the pre-v11 recipe); LSST with 0 positives → ``[]``;
    * ``full_window`` = all dets (positive OR negative) with t ≤ t(Nth positive).
    """
    ndets = [_ensure_normalized(d) for d in detections]
    pos = [d for d in ndets if d.get("is_positive", True)]
    if not pos:
        if str(survey).lower() == "lsst":
            return []
        pos = list(ndets)  # ZTF all-negative fallback (11% of ZTF LCs depend on it)
    pos = sorted(pos, key=_mjd_key)
    ordered = sorted(ndets, key=_mjd_key)
    n_max = len(pos) if max_n_det is None else min(len(pos), max_n_det)
    windows: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
    for n in range(1, n_max + 1):
        pos_prefix = pos[:n]
        t_n = pos_prefix[-1].get("mjd")
        full = list(pos_prefix) if t_n is None else [d for d in ordered if _mjd_key(d) <= t_n]
        windows.append((pos_prefix, full))
    return windows


def _detection_windows(
    detections: list[dict[str, Any]],
    *,
    survey: str,
    max_n_det: int | None,
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """Delegate to P2's single canonical helper
    (``debass_meta.features.lightcurve.truncated_detection_windows``); fall back
    to the local replica until P2 lands it.  ONE truncation implementation is
    the B10 contract — the lockstep copies at ``lightcurve.py``,
    ``build_snapshots_fusion.py`` and here all route through it.

    ``max_n_det=None`` means "all positives" (the historical v9 default); P2's
    canonical helper types ``max_n_det`` as a plain ``int`` and does
    ``min(len(pos), max_n_det)``, so None must be resolved to a concrete bound
    on THIS (P5) side before delegating — ``len(detections)`` is ≥ the positive
    count, so ``min`` still yields every epoch window.  The local replica
    already accepts None, but we normalise uniformly for both paths."""
    if max_n_det is None:
        max_n_det = len(detections)
    try:
        from debass_meta.features.lightcurve import (  # noqa: E402
            truncated_detection_windows as _canonical,
        )
    except ImportError:
        return _local_truncated_windows(detections, survey=survey, max_n_det=max_n_det)
    return _canonical(detections, survey=survey, max_n_det=max_n_det)


def truncated_positive_detections(
    detections: list[dict[str, Any]],
    *,
    max_len: int | None = None,
    survey: str = "ztf",
) -> list[dict[str, Any]]:
    """First ``max_len`` positive detections (normalize → positive filter → MJD
    sort → truncate), via P2's canonical windowing helper (lockstep copy dead).

    ``survey`` defaults to ``"ztf"`` so the historical unconditional
    all-negative fallback is preserved (v9 bit-identity): the canonical helper's
    ``survey != "lsst"`` branch is contractually ZTF-identical to the pre-v11
    recipe.  Pass ``survey="LSST"`` (or ``"lsst"``) to get the LSST 0-positives
    → empty behaviour."""
    windows = _detection_windows(detections, survey=survey, max_n_det=max_len)
    return list(windows[-1][0]) if windows else []


def _infer_survey(detections: list[dict[str, Any]]) -> str:
    """"lsst" when any detection is an LSST detection, else "ztf" — keys the
    survey-gated truncation fallback for the v11 window."""
    for det in detections:
        nd = _ensure_normalized(det)
        if str(nd.get("survey") or "").upper() == "LSST":
            return "lsst"
    return "ztf"


def _v11_window(
    detections: list[dict[str, Any]],
    *,
    max_len: int | None = None,
) -> list[dict[str, Any]]:
    """Full epoch window (positives + in-window negatives, MJD-sorted) at the
    deepest epoch N = min(#positives, max_len) — the token stream the v11
    sequence encoder consumes."""
    windows = _detection_windows(detections, survey=_infer_survey(detections), max_n_det=max_len)
    return list(windows[-1][1]) if windows else []


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
    schema: str = "v9",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (continuous (L, cont_dim) float32, band_idx (L,) int64) for one
    object.  ``schema="v9"`` → the historical 9-dim positives-only tensor
    (byte-identical: same channels, same preprocessing); ``schema="v11"`` →
    the 11-dim tensor over the full epoch window (positives + in-window
    negatives) with the appended ``is_negative`` + ``signed_flux`` channels.

    ``pre_truncated=True`` skips preprocessing when the caller already holds the
    windowed/normalized token list (the gold builder's own lists — v9 the
    positives-only prefix, v11 the full window).
    """
    cont_dim = _cont_dim_for_schema(schema)
    if pre_truncated:
        dets = detections
    elif schema == "v11":
        dets = _v11_window(detections, max_len=max_len)
    else:
        dets = truncated_positive_detections(detections, max_len=max_len)
    L = len(dets)
    cont = np.zeros((L, cont_dim), dtype=np.float32)
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

        if schema == "v11":
            is_pos = bool(det.get("is_positive", True))
            cont[k, _IS_NEGATIVE_DIM] = 0.0 if is_pos else 1.0
            flux = _finite(det.get("flux"))
            if flux is not None:
                # Sign encodes brighten(+)/fade(−); asinh compresses the wide
                # (nJy-scale LSST vs mag-derived ZTF) dynamic range while
                # passing through zero smoothly.  Per-survey NormStats then
                # z-scores this dim, absorbing the cross-survey scale gap.
                signed = abs(flux) if is_pos else -abs(flux)
                cont[k, _SIGNED_FLUX_DIM] = float(np.arcsinh(signed))

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
    schema: str = "v9",
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load + tensorize one object's cached lightcurve; None when absent/empty.

    ``schema`` selects the v9 (positives-only, 9-dim) or v11 (full-window,
    11-dim) tensorization.  Imports the gold loader lazily to keep this module
    import-light.
    """
    from debass_meta.ingest.gold import _load_lightcurve, _resolve_lightcurve_path

    lc_path, _source = _resolve_lightcurve_path(lc_dir, object_id=object_id, associations=None)
    if lc_path is None:
        return None
    detections = _load_lightcurve(lc_path)
    if not detections:
        return None
    cont, bands = sequence_arrays(detections, max_len=max_len, schema=schema)
    if len(cont) == 0:
        return None
    return cont, bands


def epoch_window_tokens(
    detections: list[dict[str, Any]],
    *,
    schema: str = "v9",
    max_len: int | None = None,
) -> list[dict[str, Any]]:
    """Ordered token-dict list the given ``schema`` tensorizes at the deepest
    epoch (v9 → positives-only prefix; v11 → full epoch window incl. in-window
    negatives).  ADDITIVE train-time helper: because tokenization is strictly
    causal (token *k* depends only on tokens ``0..k``), slicing the RESULT and
    re-tensorizing with ``sequence_arrays(..., pre_truncated=True)`` re-anchors
    a contiguous sub-window correctly — the basis for random-phase augmentation
    without disturbing the public tokenizer signatures."""
    if schema == "v11":
        return list(_v11_window(detections, max_len=max_len))
    if schema == "v9":
        return list(truncated_positive_detections(detections, max_len=max_len))
    raise ValueError(f"unknown sequence schema {schema!r} (expected one of {SEQ_SCHEMAS})")


def load_object_tokens(
    lc_dir: Path,
    object_id: str,
    *,
    max_len: int | None = None,
    schema: str = "v9",
) -> list[dict[str, Any]] | None:
    """Ordered token dicts for one cached object (None when absent/empty).

    Uses the SAME loader/windowing as :func:`load_object_sequence`; the returned
    list re-tensorizes 1:1 via ``sequence_arrays(toks, pre_truncated=True)`` and
    a contiguous slice re-tensorizes as a re-anchored window (train-time
    random-phase augmentation)."""
    from debass_meta.ingest.gold import _load_lightcurve, _resolve_lightcurve_path

    lc_path, _source = _resolve_lightcurve_path(lc_dir, object_id=object_id, associations=None)
    if lc_path is None:
        return None
    detections = _load_lightcurve(lc_path)
    if not detections:
        return None
    toks = epoch_window_tokens(detections, schema=schema, max_len=max_len)
    return toks or None


def sequence_survey(cont: np.ndarray) -> str:
    """Classify one sequence: "lsst" when any detection carries the LSST
    flag, else "ztf" (shared by the --surveys filters in the v10 trainers)."""
    if len(cont) and bool(np.max(cont[:, _SURVEY_DIM]) > 0.5):
        return "lsst"
    return "ztf"


def _fit_moments(stacked: np.ndarray) -> tuple[list[float], list[float]]:
    dim = stacked.shape[1]
    mean = [0.0] * dim
    std = [1.0] * dim
    for d in _zscore_dims_for(dim):
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
        zdims = _zscore_dims_for(len(self.mean))
        if not self.per_survey:
            for d in zdims:
                out[:, d] = (out[:, d] - self.mean[d]) / self.std[d]
            return out
        is_lsst = cont[:, _SURVEY_DIM] > 0.5
        for survey, sel in (("ztf", ~is_lsst), ("lsst", is_lsst)):
            if not np.any(sel):
                continue
            entry = self.per_survey.get(survey)
            mean = entry["mean"] if entry else self.mean
            std = entry["std"] if entry else self.std
            for d in zdims:
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
