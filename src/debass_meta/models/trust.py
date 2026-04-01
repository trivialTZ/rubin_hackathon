"""Legacy per-broker trust head.

This module is retained for baseline compatibility only.
The production trust-aware path uses per-expert training in
``debass_meta.models.expert_trust``.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class TrustHead:
    """Legacy logistic trust model for a single broker/expert.

    Prefer ``debass_meta.models.expert_trust`` for the production trust-aware stack.
    """

    def __init__(self, broker_name: str) -> None:
        self.broker_name = broker_name
        self.name = f"trust_{broker_name}"
        self._clf = None
        self._feature_names: list[str] = []

    def fit(
        self,
        phi: np.ndarray,
        m: np.ndarray,
        y_helpful: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> None:
        from sklearn.linear_model import LogisticRegression
        self._feature_names = feature_names or []
        if len(np.unique(y_helpful)) < 2:
            return
        X = np.column_stack([phi, m.reshape(-1, 1)])
        self._clf = LogisticRegression(max_iter=500, C=1.0)
        self._clf.fit(X, y_helpful)

    def predict_trust(
        self,
        phi: np.ndarray,
        m: np.ndarray,
    ) -> np.ndarray:
        if self._clf is None:
            return m.astype(float)
        X = np.column_stack([phi, m.reshape(-1, 1)])
        return self._clf.predict_proba(X)[:, 1]

    def metadata(self) -> dict[str, Any]:
        return {
            "broker": self.broker_name,
            "fitted": self._clf is not None,
            "feature_names": self._feature_names,
            "status": "legacy_baseline_only",
        }


def build_trust_features(
    broker_scores: np.ndarray,
    availability: np.ndarray,
    epoch_indices: np.ndarray,
) -> np.ndarray:
    """Construct legacy broker-level trust features.

    Prefer expert-specific features in ``debass_meta.models.expert_trust``.
    """
    scores_clean = np.nan_to_num(broker_scores, nan=0.5)
    mean_score = scores_clean.mean(axis=1) if scores_clean.ndim > 1 else scores_clean
    std_score = scores_clean.std(axis=1) if scores_clean.ndim > 1 else np.zeros_like(mean_score)
    return np.column_stack([
        epoch_indices.astype(float),
        availability.astype(float),
        mean_score,
        std_score,
    ])
