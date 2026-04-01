"""Baseline classifiers for the DEBASS meta-classifier benchmark.

All baselines share the same fit(X, y) / predict_proba(X) interface
so they can be swapped into the evaluation pipeline without changes.

Target: ternary labels  0=snia  1=nonIa_snlike  2=other
"""
from __future__ import annotations

from typing import Any

import numpy as np

LABEL_MAP = {"snia": 0, "nonIa_snlike": 1, "other": 2}
LABEL_NAMES = ["snia", "nonIa_snlike", "other"]
N_CLASSES = 3


class SingleExpertBaseline:
    """Use one expert's probability column(s) as the entire classifier."""

    def __init__(self, expert_col: str) -> None:
        self.expert_col = expert_col
        self.name = f"single_{expert_col}"

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> None:
        self._feature_names = feature_names or []
        if feature_names and self.expert_col in feature_names:
            self._col_idx = feature_names.index(self.expert_col)
        else:
            self._col_idx = 0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_ia = np.clip(X[:, self._col_idx], 0, 1)
        p_other = 1 - p_ia
        # Ternary: split non-Ia evenly between nonIa_snlike and other
        return np.column_stack([p_ia, p_other * 0.5, p_other * 0.5])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


class AverageEnsemble:
    """Unweighted average of all available probability columns."""

    name = "average_ensemble"

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> None:
        self._n_features = X.shape[1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        mean_score = np.nanmean(np.clip(X, 0, 1), axis=1)
        p_other = 1 - mean_score
        return np.column_stack([mean_score, p_other * 0.5, p_other * 0.5])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


class WeightedEnsemble:
    """Weighted average; weights learned by minimising NLL on val set."""

    name = "weighted_ensemble"

    def __init__(self) -> None:
        self._weights: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> None:
        from scipy.optimize import minimize

        n = X.shape[1]
        X_clean = np.nan_to_num(np.clip(X, 1e-6, 1 - 1e-6), nan=1.0 / N_CLASSES)

        def neg_log_likelihood(w):
            w = np.abs(w) / np.abs(w).sum()
            scores = X_clean @ w
            scores = np.clip(scores, 1e-6, 1 - 1e-6)
            # Binary cross-entropy against SN Ia label
            y_ia = (y == 0).astype(float)
            return -np.mean(y_ia * np.log(scores) + (1 - y_ia) * np.log(1 - scores))

        result = minimize(neg_log_likelihood, np.ones(n) / n, method="L-BFGS-B")
        w = np.abs(result.x)
        self._weights = w / w.sum()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("Call fit() first")
        X_clean = np.nan_to_num(np.clip(X, 0, 1), nan=1.0 / N_CLASSES)
        score = X_clean @ self._weights
        p_other = 1 - score
        return np.column_stack([score, p_other * 0.5, p_other * 0.5])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


class StackedEnsemble:
    """Logistic regression or LightGBM stacker over broker feature matrix."""

    def __init__(self, method: str = "logistic") -> None:
        self.method = method
        self.name = f"stacked_{method}"
        self._clf = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> None:
        X_clean = np.nan_to_num(X, nan=1.0 / N_CLASSES)
        if self.method == "logistic":
            from sklearn.linear_model import LogisticRegression
            self._clf = LogisticRegression(max_iter=1000, multi_class="multinomial", C=1.0)
        elif self.method == "lgbm":
            from lightgbm import LGBMClassifier
            self._clf = LGBMClassifier(n_estimators=200, num_leaves=31, verbose=-1)
        else:
            raise ValueError(f"Unknown stacking method: {self.method}")
        self._clf.fit(X_clean, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Call fit() first")
        X_clean = np.nan_to_num(X, nan=1.0 / N_CLASSES)
        return self._clf.predict_proba(X_clean)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
