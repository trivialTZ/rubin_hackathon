"""Model utilities for DEBASS."""

from .calibrate import TemperatureScaler
from .early_meta import DEFAULT_FEATURES, EarlyMetaClassifier, LABEL_MAP, LABEL_NAMES, N_CLASSES
from .expert_trust import ExpertTrustArtifact, train_expert_trust_suite
from .followup import FollowupArtifact, train_followup_model
from .splitters import GroupSplit, group_train_cal_test_split

__all__ = [
    "DEFAULT_FEATURES",
    "EarlyMetaClassifier",
    "ExpertTrustArtifact",
    "FollowupArtifact",
    "GroupSplit",
    "LABEL_MAP",
    "LABEL_NAMES",
    "N_CLASSES",
    "TemperatureScaler",
    "group_train_cal_test_split",
    "train_expert_trust_suite",
    "train_followup_model",
]
