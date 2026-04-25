from .supernnova import SuperNNovaExpert
from .parsnip import ParSNIPExpert
from .ampel_runner import AmpelRunner
from .alerce_lc import AlerceLCExpert
from .salt3_fit import Salt3Chi2Expert
from .lc_features import LcFeaturesExpert

ALL_LOCAL_EXPERTS = [
    SuperNNovaExpert,
    ParSNIPExpert,
    AmpelRunner,
    AlerceLCExpert,
    Salt3Chi2Expert,
    LcFeaturesExpert,
]

# Convenience lists for CPU vs GPU pipeline stages
CPU_LOCAL_EXPERTS = [cls for cls in ALL_LOCAL_EXPERTS if not cls.requires_gpu]
GPU_LOCAL_EXPERTS = [cls for cls in ALL_LOCAL_EXPERTS if cls.requires_gpu]
