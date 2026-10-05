"""Probability calibration helpers.

TemperatureScaler — multiclass post-hoc calibration (grid-search temperature).
IsotonicCalibrator — binary post-hoc calibration via isotonic regression.
BetaCalibrator — binary post-hoc calibration via beta calibration (smooth, monotone).

References:
  Niculescu-Mizil & Caruana (2005), "Predicting Good Probabilities With
  Supervised Learning", ICML.  Isotonic regression outperforms Platt
  scaling when N_cal >= ~1000.
  Kull, Silva Filho & Flach (2017), "Beta calibration: a well-founded and
  easily implemented improvement on logistic calibration for binary
  classifiers", AISTATS.
"""
from __future__ import annotations

import numpy as np


class TemperatureScaler:
    """Simple grid-search temperature scaling for multiclass probabilities."""

    name = "temperature_scaling"

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = float(temperature)

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        probs = np.asarray(probs, dtype=float)
        y_true = np.asarray(y_true, dtype=int)
        probs = np.clip(probs, 1e-8, 1.0)
        logits = np.log(probs)

        best_t = 1.0
        best_loss = float("inf")
        for temperature in np.linspace(0.5, 3.0, 26):
            scaled = self._softmax(logits / float(temperature))
            loss = -np.mean(np.log(np.clip(scaled[np.arange(len(y_true)), y_true], 1e-8, 1.0)))
            if loss < best_loss:
                best_loss = float(loss)
                best_t = float(temperature)
        self.temperature = best_t
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=float)
        probs = np.clip(probs, 1e-8, 1.0)
        logits = np.log(probs)
        return self._softmax(logits / self.temperature)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)


class IsotonicCalibrator:
    """Isotonic regression calibrator for binary probabilities.

    Non-parametric: learns a monotonic step function mapping raw scores
    to calibrated probabilities.  More flexible than Platt scaling and
    preferred when the calibration set has >= ~1000 samples.
    """

    name = "isotonic"

    def __init__(self) -> None:
        self._ir = None

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression

        y_prob = np.asarray(y_prob, dtype=float)
        y_true = np.asarray(y_true, dtype=int)
        self._ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self._ir.fit(y_prob, y_true)
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        y_prob = np.asarray(y_prob, dtype=float)
        if self._ir is None:
            return y_prob
        return self._ir.predict(y_prob)


class BetaCalibrator:
    """Beta calibration (Kull, Silva Filho & Flach 2017, AISTATS) for binary
    probabilities, with optional per-row weights.

    ``logit(p_cal) = c + a ln(p) - b ln(1 - p)``: a weighted logistic regression
    on ``[ln p, -ln(1 - p)]`` with ``p`` clipped to ``[1e-6, 1 - 1e-6]``.  A map
    is monotone only for ``a, b >= 0``; when a coefficient comes out negative the
    regression is refitted without that feature (as the ``betacal`` package
    does), and when no feature survives the map is the weighted base rate.  The
    map is smooth, so it has none of the plateaus (ties) of an isotonic fit.
    """

    name = "beta"
    _EPS = 1e-6

    def __init__(self) -> None:
        self.a = 0.0
        self.b = 0.0
        self.c = 0.0
        self.fitted = False

    @classmethod
    def _features(cls, y_prob: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(y_prob, dtype=float), cls._EPS, 1.0 - cls._EPS)
        return np.column_stack([np.log(p), -np.log1p(-p)])

    @staticmethod
    def _lr(x: np.ndarray, y: np.ndarray, w: np.ndarray | None):
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=42)
        lr.fit(x, y, sample_weight=w)
        return lr

    def fit(self, y_prob, y_true, sample_weight=None) -> "BetaCalibrator":
        x = self._features(y_prob)
        y = np.asarray(y_true, dtype=int)
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.a = self.b = self.c = 0.0
        self.fitted = True
        if len(np.unique(y)) < 2:
            return self
        lr = self._lr(x, y, w)
        coef = [float(v) for v in lr.coef_[0]]
        c = float(lr.intercept_[0])
        keep = [0, 1]
        if min(coef) < 0.0:             # refit without the negative feature
            keep = [i for i, v in enumerate(coef) if v >= 0.0]
            coef = [0.0, 0.0]
            if keep:
                lr = self._lr(x[:, keep], y, w)
                sub = [float(v) for v in lr.coef_[0]]
                if min(sub) >= 0.0:
                    for i, v in zip(keep, sub):
                        coef[i] = v
                    c = float(lr.intercept_[0])
                else:
                    keep = []
            if not keep:                # no monotone feature left: the weighted base rate
                pw = np.ones(len(y)) if w is None else w
                rate = float(np.clip((pw * y).sum() / max(pw.sum(), 1e-12), 1e-6, 1.0 - 1e-6))
                c = float(np.log(rate / (1.0 - rate)))
        self.a, self.b, self.c = coef[0], coef[1], c
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return np.asarray(y_prob, dtype=float)
        x = self._features(y_prob)
        z = self.c + self.a * x[:, 0] + self.b * x[:, 1]
        return 1.0 / (1.0 + np.exp(-z))
