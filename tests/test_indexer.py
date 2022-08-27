import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData
from hydra.utils import instantiate
from omegaconf import OmegaConf

from ._utils import assert_allclose, to_view

X = np.array([
    [1.0, 4, 0, 2],
    [0, 3, 2, 0],
    [3, 2, 1, 7.0],
], dtype=np.float32)


X_mods = [X, sp.csr_matrix(X), sp.csc_matrix(X), to_view(X)]


@pytest.mark.parametrize("X", X_mods)
def test_indexer_obs(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.InplaceIndexer.Config",
            "mask_key": "obs.mask",
        }
    )
    cfg = instantiate(cfg)
    index = cfg.initialize()
    adata = AnnData(X)
    X_original = adata.X.copy()
    adata.obs['mask'] = [0, 1, 1]
    index(adata)
    assert_allclose(X_original, X)
    assert_allclose(adata.X, np.array([
        [0, 3, 2, 0],
        [3, 2, 1, 7]
    ]))


@pytest.mark.parametrize("X", X_mods)
def test_indexer_var(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.InplaceIndexer.Config",
            "axis": 'var',
            "mask_key": "var.mask",
        }
    )
    cfg = instantiate(cfg)
    index = cfg.initialize()
    adata = AnnData(X)
    X_original = adata.X.copy()
    adata.var['mask'] = [2, 0, 1, 1]
    index(adata)
    assert_allclose(X_original, X)
    assert_allclose(adata.X, np.array([
        [1, 0, 2],
        [0, 2, 0],
        [3, 1, 7]
    ]))


@pytest.mark.parametrize("X", X_mods)
def test_indexer_obs_no_mask(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.InplaceIndexer.Config",
            "mask_key": "uns.mask",
            "as_bool": False,
        }
    )
    cfg = instantiate(cfg)
    index = cfg.initialize()
    adata = AnnData(X)
    X_original = adata.X.copy()
    adata.uns['mask'] = [1, 0]
    index(adata)
    assert_allclose(X_original, X)
    assert_allclose(adata.X, np.array([
        [0, 3, 2, 0],
        [1, 4, 0, 2],
    ]))


@pytest.mark.parametrize("X", X_mods)
def test_indexer_var_no_mask(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.InplaceIndexer.Config",
            "axis": 'var',
            "mask_key": "uns.mask",
            "as_bool": False,
        }
    )
    cfg = instantiate(cfg)
    index = cfg.initialize()
    adata = AnnData(X)
    X_original = adata.X.copy()
    adata.uns['mask'] = [2, 1]
    index(adata)
    assert_allclose(X_original, X)
    assert_allclose(adata.X, np.array([
        [0, 4],
        [2, 3],
        [1, 2]
    ]))
