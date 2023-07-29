import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData
from hydra.utils import instantiate
from omegaconf import OmegaConf

from ._utils import to_view

X = np.array([
    [1.0, 4, 4, 2],
    [0, 1, 2, 0],
    [3, 2, 1, 7.0],
    [3, 2, 1, 8.0],
], dtype=np.float32)


X_mods = [X, sp.csr_matrix(X), to_view(X)]


@pytest.mark.parametrize("X", X_mods)
def test_fnan(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.FilterNaN.Config",
            "read_key": "obs.vals",
        }
    )
    cfg = instantiate(cfg)
    fnan = cfg.create()
    adata = AnnData(X)
    adata.obs['vals'] = [0, 1, np.nan, 4]
    fnan(adata)

    assert adata.shape[0] == 3


@pytest.mark.parametrize("X", X_mods)
def test_fnan_2(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.FilterNaN.Config",
            "read_key": "obs.vals",
        }
    )
    cfg = instantiate(cfg)
    fnan = cfg.create()
    adata = AnnData(X)
    adata.obs['vals'] = ['0', '1', 'nan', '4']
    fnan(adata)

    assert adata.shape[0] == 3
