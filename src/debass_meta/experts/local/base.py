"""Abstract base class for local expert classifiers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExpertOutput:
    """Standardised output from a local expert at a single epoch."""
    expert: str
    object_id: str
    epoch_jd: float          # Julian date of the epoch scored
    # Probability dict: class_name -> float in [0,1]
    class_probabilities: dict[str, float] = field(default_factory=dict)
    # Raw model output (preserved verbatim alongside probabilities)
    raw_output: dict[str, Any] = field(default_factory=dict)
    semantic_type: str = "probability"
    model_version: str = "unknown"
    available: bool = True   # False when model weights not loaded


class LocalExpert(ABC):
    """Interface for local expert classifiers."""

    name: str          # subclass must set
    semantic_type: str = "probability"
    requires_gpu: bool = False  # True for torch-based experts (SNN, ParSNIP)

    @abstractmethod
    def fit(self, lightcurves: Any, labels: Any) -> None:
        """Train or fine-tune the expert on historical data."""

    @abstractmethod
    def predict_epoch(self, object_id: str, lightcurve: Any, epoch_jd: float) -> ExpertOutput:
        """Score a single object at a given epoch (truncated lightcurve)."""

    def predict_proba(self, object_id: str, lightcurve: Any, epoch_jd: float) -> dict[str, float]:
        """Convenience wrapper — returns class_probabilities dict only."""
        return self.predict_epoch(object_id, lightcurve, epoch_jd).class_probabilities

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return expert metadata: name, version, classes, input requirements."""
