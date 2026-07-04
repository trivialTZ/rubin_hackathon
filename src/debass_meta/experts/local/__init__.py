from .supernnova import SuperNNovaExpert
from .parsnip import ParSNIPExpert
from .ampel_runner import AmpelRunner
from .snguess import AmpelSNGuessExpert
from .alerce_lc import AlerceLCExpert
from .salt3_fit import Salt3Chi2Expert
from .lc_features import LcFeaturesExpert
from .seq_v9 import SeqV9Expert

ALL_LOCAL_EXPERTS = [
    SuperNNovaExpert,
    ParSNIPExpert,
    AmpelRunner,
    AmpelSNGuessExpert,
    AlerceLCExpert,
    Salt3Chi2Expert,
    LcFeaturesExpert,
    SeqV9Expert,
]

# Convenience lists for CPU vs GPU pipeline stages
CPU_LOCAL_EXPERTS = [cls for cls in ALL_LOCAL_EXPERTS if not cls.requires_gpu]
GPU_LOCAL_EXPERTS = [cls for cls in ALL_LOCAL_EXPERTS if cls.requires_gpu]
