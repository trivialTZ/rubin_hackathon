"""Legacy trust-weighted meta-classifier.

This module is retained for baseline compatibility only.
The production trust-aware path uses expert-specific trust heads plus the
follow-up payload pipeline (`scripts/train_expert_trust.py`,
`scripts/train_followup.py`, `scripts/score_nightly.py`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .trust import TrustHead, build_trust_features
from .baselines import LABEL_NAMES, N_CLASSES


class MetaClassifier:
    """Legacy broker-level trust-weighted meta-classifier.

    Prefer the expert-trust + followup stack for production use.
    """

    def __init__(
        self,
        broker_names: list[str],
        method: str = "lgbm",
    ) -> None:
        self.broker_names = broker_names
        self.method = method
        self.name = f"meta_{method}"
        self._trust_heads: dict[str, TrustHead] = {
            b: TrustHead(b) for b in broker_names
        }
        self._clf = None
        self._feature_names: list[str] = []

    def fit(
        self,
        X_brokers: dict[str, np.ndarray],
        availability: dict[str, np.ndarray],
        epoch_indices: np.ndarray,
        y: np.ndarray,
        y_helpful: dict[str, np.ndarray] | None = None,
    ) -> None:
        for b in self.broker_names:
            scores = X_brokers.get(b, np.zeros((len(y), 1)))
            avail = availability.get(b, np.zeros(len(y)))
            phi = build_trust_features(scores, avail, epoch_indices)
            if y_helpful and b in y_helpful:
                helpful = y_helpful[b]
            else:
                helpful = avail.astype(int)
            self._trust_heads[b].fit(phi, avail, helpful)

        Z = self._build_weighted_features(X_brokers, availability, epoch_indices)
        self._feature_names = self._make_feature_names()

        Z_clean = np.nan_to_num(Z, nan=1.0 / N_CLASSES)
        if self.method == "logistic":
            from sklearn.linear_model import LogisticRegression
            self._clf = LogisticRegression(max_iter=1000, multi_class="multinomial", C=1.0)
        else:
            from lightgbm import LGBMClassifier
            self._clf = LGBMClassifier(
                n_estimators=300, num_leaves=31,
                learning_rate=0.05, verbose=-1,
            )
        self._clf.fit(Z_clean, y)

    def predict_proba(
        self,
        X_brokers: dict[str, np.ndarray],
        availability: dict[str, np.ndarray],
        epoch_indices: np.ndarray,
    ) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Call fit() first")
        Z = self._build_weighted_features(X_brokers, availability, epoch_indices)
        Z_clean = np.nan_to_num(Z, nan=1.0 / N_CLASSES)
        return self._clf.predict_proba(Z_clean)

    def predict(
        self,
        X_brokers: dict[str, np.ndarray],
        availability: dict[str, np.ndarray],
        epoch_indices: np.ndarray,
    ) -> np.ndarray:
        return np.argmax(self.predict_proba(X_brokers, availability, epoch_indices), axis=1)

    def save(self, model_dir: Path) -> None:
        import pickle
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "meta_clf.pkl", "wb") as fh:
            pickle.dump(self._clf, fh)
        for b, th in self._trust_heads.items():
            with open(model_dir / f"trust_{b}.pkl", "wb") as fh:
                pickle.dump(th, fh)

    @classmethod
    def load(cls, model_dir: Path, broker_names: list[str], method: str = "lgbm") -> "MetaClassifier":
        import pickle
        model_dir = Path(model_dir)
        obj = cls(broker_names=broker_names, method=method)
        with open(model_dir / "meta_clf.pkl", "rb") as fh:
            obj._clf = pickle.load(fh)
        for b in broker_names:
            p = model_dir / f"trust_{b}.pkl"
            if p.exists():
                with open(p, "rb") as fh:
                    obj._trust_heads[b] = pickle.load(fh)
        return obj

    def _build_weighted_features(
        self,
        X_brokers: dict[str, np.ndarray],
        availability: dict[str, np.ndarray],
        epoch_indices: np.ndarray,
    ) -> np.ndarray:
        parts = []
        for b in self.broker_names:
            scores = X_brokers.get(b, np.zeros((len(epoch_indices), 1)))
            avail = availability.get(b, np.zeros(len(epoch_indices)))
            phi = build_trust_features(scores, avail, epoch_indices)
            trust = self._trust_heads[b].predict_trust(phi, avail)
            scores_clean = np.nan_to_num(scores, nan=0.5)
            weighted = scores_clean * trust.reshape(-1, 1)
            parts.append(weighted)
            parts.append(trust.reshape(-1, 1))
        parts.append(epoch_indices.reshape(-1, 1))
        return np.hstack(parts)

    def _make_feature_names(self) -> list[str]:
        names = []
        for b in self.broker_names:
            names.extend([f"{b}_weighted_score", f"{b}_trust"])
        names.append("epoch_index")
        return names
