import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from hydra.utils import instantiate
from omegaconf import OmegaConf

from grinch import VAR, OBS, OBSM

from ._utils import assert_allclose, to_view

X = np.random.random((100, 600))
