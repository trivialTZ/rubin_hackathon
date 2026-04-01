"""Probability calibration helpers.

TemperatureScaler — multiclass post-hoc calibration (grid-search temperature).
IsotonicCalibrator — binary post-hoc calibration via isotonic regression.

References:
  Niculescu-Mizil & Caruana (2005), "Predicting Good Probabilities With
  Supervised Learning", ICML.  Isotonic regression outperforms Platt
  scaling when N_cal >= ~1000.
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
