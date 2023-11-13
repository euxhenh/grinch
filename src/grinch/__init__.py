# flake8: noqa: E402
import warnings

from numba.core.errors import (
    NumbaDeprecationWarning,
    NumbaPendingDeprecationWarning,
)

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

from . import custom_types as typing
from . import processors as pr
from . import shortcuts
from .aliases import ADK, OBS, OBSM, OBSP, UNS, VAR, VARM, VARP, AnnDataKeys
from .base import StorageMixin
from .cond_filter import Filter, StackedFilter
from .conf import BaseConfigurable
from .filters import FilterCells, FilterGenes, VarianceFilter
from .main import instantiate_config
from .normalizers import Combat, Harmony, Log1P, NormalizeTotal, Scale
from .pipeline import GRPipeline, MultiRead
from .processors import *  # noqa
from .reporter import Report, Reporter
from .shortcuts import *  # noqa

__all__ = [
    'typing',
    'ADK',
    'OBS',
    'OBSM',
    'VAR',
    'VARM',
    'OBSP',
    'VARP',
    'UNS',
    'StorageMixin',
    'AnnDataKeys',
    'BaseConfigurable',
    'GRPipeline',
    'MultiRead',
    'FilterCells',
    'FilterGenes',
    'VarianceFilter',
    'Report',
    'Reporter',
    'Filter',
    'StackedFilter',
    'Combat',
    'Harmony',
    'Log1P',
    'Scale',
    'NormalizeTotal',
    'instantiate_config',
]

__all__.extend(pr.__all__)
__all__.extend(shortcuts.__all__)
