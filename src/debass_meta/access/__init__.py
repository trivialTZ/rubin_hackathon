from .alerce import AlerceAdapter
from .ampel import AmpelAdapter
from .antares import AntaresAdapter
from .babamul import BabamulAdapter
from .fink import FinkAdapter
from .lasair import LasairAdapter
from .pitt import PittAdapter

ALL_ADAPTERS = [
    AlerceAdapter,
    FinkAdapter,
    LasairAdapter,
    PittAdapter,
    AntaresAdapter,
    AmpelAdapter,
    BabamulAdapter,
]
