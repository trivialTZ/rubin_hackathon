"""Mondrian split-conformal Adaptive Prediction Sets (APS) for fusion_v8.

Per-class thresholds are fit on the CAL split only (split-conformal: cal is
never used for model fitting, so the exchangeability argument holds for the
locked test split and deployment rows).  Mondrian strata are
``true class × survey × n_det bucket {1-4, 5-9, 10-20}``; buckets with fewer
than ``min_count`` cal objects fall back up the hierarchy
``class×survey×n_det -> class×survey -> class -> global``.

Nonconformity score (APS, Romano et al. 2020, non-randomized):
``s(x, y) = Σ_{c: p̂_c >= p̂_y} p̂_c`` — the cumulative calibrated probability
mass at-or-above the candidate class.  The per-bucket threshold is the
conformal quantile ``q̂ = ⌈(n+1)(1−α)⌉/n`` of the true-class scores; the
deployment prediction set is ``{c : s(x, c) <= q̂_bucket(c, survey, n_det)}``.

Honest behavior: data-starved strata inherit coarser thresholds (wide sets =
calibrated "I don't know"), and ``coverage_report`` makes empirical coverage
vs the nominal 1−α directly inspectable per stratum.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .multiclass_followup import CLASSES

# n_det buckets per design §4.2: {1-4, 5-9, 10-20}; n_det > 20 clamps into
# the last bucket (same epoch regime).
N_DET_BUCKET_LABELS = ("1-4", "5-9", "10-20")

_LEVEL_NAMES = ("class_survey_ndet", "class_survey", "class", "global")


def ndet_bucket(n_det) -> np.ndarray:
    """Vectorized n_det -> bucket index {0, 1, 2}."""
    n = np.asarray(n_det, dtype=float)
    return np.where(n <= 4, 0, np.where(n <= 9, 1, 2)).astype(int)


def _aps_score_matrix(proba: np.ndarray) -> np.ndarray:
    """(N, 3) matrix of APS scores: S[i, c] = Σ_{c': p_ic' >= p_ic} p_ic'."""
    P = np.asarray(proba, dtype=float)
    S = np.zeros_like(P)
    for c in range(P.shape[1]):
        S[:, c] = np.where(P >= P[:, [c]], P, 0.0).sum(axis=1)
    return S


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile ⌈(n+1)(1−α)⌉/n (method='higher').

    Returns 1.0 (full set) when n is too small for the requested level —
    the honest conservative limit."""
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    if n == 0:
        return 1.0
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return 1.0
    return float(scores[k - 1])


def _validate_strata(strata: pd.DataFrame, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    for col in ("survey", "n_det"):
        if col not in strata.columns:
            raise KeyError(f"strata frame must have a '{col}' column")
    if len(strata) != n_rows:
        raise ValueError(
            f"strata rows ({len(strata)}) != proba rows ({n_rows})"
        )
    survey = strata["survey"].astype(str).to_numpy()
    bucket = ndet_bucket(strata["n_det"].to_numpy())
    return survey, bucket


class MondrianAPS:
    """Mondrian split-conformal APS with hierarchy fallback (pinned API)."""

    def __init__(self, alpha: float = 0.1, min_count: int = 30) -> None:
        self.alpha = float(alpha)
        self.min_count = int(min_count)
        # thresholds per hierarchy level (filled by fit)
        self._t3: dict[tuple[int, str, int], float] = {}
        self._t2: dict[tuple[int, str], float] = {}
        self._t1: dict[int, float] = {}
        self._t0: float | None = None
        # cal-object counts per stratum (diagnostics / metadata)
        self._n3: dict[tuple[int, str, int], int] = {}
        self._n2: dict[tuple[int, str], int] = {}
        self._n1: dict[int, int] = {}
        self._n0: int = 0

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        proba_cal: np.ndarray,
        y_cal: np.ndarray,
        strata_cal: pd.DataFrame,
        alpha: float = 0.1,
    ) -> "MondrianAPS":
        """Fit per-stratum thresholds on the CAL split.

        ``strata_cal`` must carry ``survey`` and ``n_det`` columns; an
        optional ``object_id`` column makes the >= ``min_count`` guard count
        unique cal OBJECTS (per spec) instead of rows.
        """
        self.alpha = float(alpha)
        P = np.asarray(proba_cal, dtype=float)
        y = np.asarray(y_cal, dtype=int)
        if len(P) == 0:
            raise ValueError("MondrianAPS.fit requires at least one cal row")
        survey, bucket = _validate_strata(strata_cal, len(P))
        if "object_id" in strata_cal.columns:
            object_ids = strata_cal["object_id"].astype(str).to_numpy()
        else:
            object_ids = np.arange(len(P)).astype(str)

        scores = _aps_score_matrix(P)[np.arange(len(P)), y]

        def _count(mask: np.ndarray) -> int:
            return int(len(np.unique(object_ids[mask])))

        self._t3.clear(); self._t2.clear(); self._t1.clear()
        self._n3.clear(); self._n2.clear(); self._n1.clear()
        for c in range(len(CLASSES)):
            mask_c = y == c
            n_c = _count(mask_c)
            if n_c >= self.min_count:
                self._t1[c] = _conformal_quantile(scores[mask_c], self.alpha)
                self._n1[c] = n_c
            for sv in np.unique(survey[mask_c]) if mask_c.any() else []:
                mask_cs = mask_c & (survey == sv)
                n_cs = _count(mask_cs)
                if n_cs >= self.min_count:
                    self._t2[(c, str(sv))] = _conformal_quantile(
                        scores[mask_cs], self.alpha
                    )
                    self._n2[(c, str(sv))] = n_cs
                for b in np.unique(bucket[mask_cs]) if mask_cs.any() else []:
                    mask_csb = mask_cs & (bucket == b)
                    n_csb = _count(mask_csb)
                    if n_csb >= self.min_count:
                        self._t3[(c, str(sv), int(b))] = _conformal_quantile(
                            scores[mask_csb], self.alpha
                        )
                        self._n3[(c, str(sv), int(b))] = n_csb
        self._t0 = _conformal_quantile(scores, self.alpha)
        self._n0 = _count(np.ones(len(P), dtype=bool))
        return self

    # --------------------------------------------------------------- lookup
    def threshold_for(
        self, class_idx: int, survey: str, n_det: int
    ) -> tuple[float, str]:
        """Most specific available threshold + the hierarchy level used."""
        if self._t0 is None:
            raise RuntimeError("MondrianAPS is not fitted")
        b = int(ndet_bucket(np.asarray([n_det]))[0])
        key3 = (int(class_idx), str(survey), b)
        if key3 in self._t3:
            return self._t3[key3], "class_survey_ndet"
        key2 = (int(class_idx), str(survey))
        if key2 in self._t2:
            return self._t2[key2], "class_survey"
        if int(class_idx) in self._t1:
            return self._t1[int(class_idx)], "class"
        return float(self._t0), "global"

    # -------------------------------------------------------------- predict
    def predict_sets(
        self, proba: np.ndarray, strata: pd.DataFrame
    ) -> np.ndarray:
        """(N, 3) boolean conformal-set membership in ``CLASSES`` order."""
        if self._t0 is None:
            raise RuntimeError("MondrianAPS is not fitted")
        P = np.asarray(proba, dtype=float)
        if len(P) == 0:
            return np.zeros((0, len(CLASSES)), dtype=bool)
        survey, bucket = _validate_strata(strata, len(P))
        S = _aps_score_matrix(P)

        thresholds = np.empty_like(S)
        # Threshold lookup is constant within (survey, bucket) groups.
        for sv in np.unique(survey):
            for b in np.unique(bucket):
                mask = (survey == sv) & (bucket == b)
                if not mask.any():
                    continue
                # representative n_det for this bucket index
                n_det_rep = {0: 3, 1: 7, 2: 15}[int(b)]
                for c in range(len(CLASSES)):
                    thresholds[mask, c] = self.threshold_for(
                        c, str(sv), n_det_rep
                    )[0]
        return S <= thresholds + 1e-12

    # ------------------------------------------------------------- coverage
    def coverage_report(
        self, proba: np.ndarray, y: np.ndarray, strata: pd.DataFrame
    ) -> pd.DataFrame:
        """Empirical class-conditional coverage + mean set size per stratum.

        One row per hierarchy level x stratum observed in the eval data plus
        a global row; compare ``coverage`` against the nominal ``1 - alpha``.
        """
        P = np.asarray(proba, dtype=float)
        y = np.asarray(y, dtype=int)
        survey, bucket = _validate_strata(strata, len(P))
        sets = self.predict_sets(P, strata)
        covered = sets[np.arange(len(y)), y]
        set_size = sets.sum(axis=1)

        rows: list[dict[str, Any]] = []

        def _add(level: str, mask: np.ndarray, class_name, sv, bucket_label):
            if not mask.any():
                return
            rows.append({
                "level": level,
                "class_name": class_name,
                "survey": sv,
                "n_det_bucket": bucket_label,
                "n": int(mask.sum()),
                "coverage": float(covered[mask].mean()),
                "mean_set_size": float(set_size[mask].mean()),
            })

        _add("global", np.ones(len(y), dtype=bool), None, None, None)
        for c, name in enumerate(CLASSES):
            mask_c = y == c
            _add("class", mask_c, name, None, None)
            for sv in sorted(set(survey[mask_c])) if mask_c.any() else []:
                mask_cs = mask_c & (survey == sv)
                _add("class_survey", mask_cs, name, sv, None)
                for b in sorted(set(bucket[mask_cs])) if mask_cs.any() else []:
                    mask_csb = mask_cs & (bucket == b)
                    _add(
                        "class_survey_ndet", mask_csb, name, sv,
                        N_DET_BUCKET_LABELS[int(b)],
                    )
        return pd.DataFrame(rows)

    # ----------------------------------------------------------- save/load
    def save(self, path) -> None:
        payload = {
            "alpha": self.alpha,
            "min_count": self.min_count,
            "classes": list(CLASSES),
            "t0": self._t0,
            "n0": self._n0,
            "t1": [
                {"class_idx": c, "threshold": t, "n": self._n1.get(c)}
                for c, t in sorted(self._t1.items())
            ],
            "t2": [
                {"class_idx": c, "survey": sv, "threshold": t,
                 "n": self._n2.get((c, sv))}
                for (c, sv), t in sorted(self._t2.items())
            ],
            "t3": [
                {"class_idx": c, "survey": sv, "bucket": b, "threshold": t,
                 "n": self._n3.get((c, sv, b))}
                for (c, sv, b), t in sorted(self._t3.items())
            ],
        }
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, path) -> "MondrianAPS":
        with open(Path(path)) as fh:
            payload = json.load(fh)
        obj = cls(alpha=float(payload["alpha"]),
                  min_count=int(payload["min_count"]))
        obj._t0 = float(payload["t0"])
        obj._n0 = int(payload.get("n0", 0))
        for entry in payload.get("t1", []):
            c = int(entry["class_idx"])
            obj._t1[c] = float(entry["threshold"])
            if entry.get("n") is not None:
                obj._n1[c] = int(entry["n"])
        for entry in payload.get("t2", []):
            key = (int(entry["class_idx"]), str(entry["survey"]))
            obj._t2[key] = float(entry["threshold"])
            if entry.get("n") is not None:
                obj._n2[key] = int(entry["n"])
        for entry in payload.get("t3", []):
            key = (
                int(entry["class_idx"]), str(entry["survey"]),
                int(entry["bucket"]),
            )
            obj._t3[key] = float(entry["threshold"])
            if entry.get("n") is not None:
                obj._n3[key] = int(entry["n"])
        return obj
