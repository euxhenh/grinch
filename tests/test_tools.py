import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData
from hydra.utils import instantiate
from omegaconf import OmegaConf

from ._utils import assert_allclose, to_view

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


@pytest.mark.parametrize("X", X_mods)
def test_applyop(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.ApplyOp.Config",
            "read_key": "obs.vals",
            "op": "square",
        }
    )
    cfg = instantiate(cfg)
    op = cfg.create()
    adata = AnnData(X)
    adata.obs['vals'] = [2, 3, 4, 5]
    op(adata)

    assert_allclose(adata.obs['vals'], [4, 9, 16, 25])
