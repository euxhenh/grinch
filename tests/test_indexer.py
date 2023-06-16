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
    fcfg = OmegaConf.create(
        {
            "_target_": "src.grinch.Filter",
            "key": "obs.pick",
        }
    )
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.InplaceIndexer.Config",
            "filter_by": [fcfg],
        }
    )
    cfg = instantiate(cfg, _convert_='all')
    index = cfg.initialize()
    adata = AnnData(X)
    X_original = adata.X.copy()
    adata.obs['pick'] = [0, 1, 1]
    index(adata)
    assert_allclose(X_original, X)
    assert_allclose(adata.X, np.array([
        [0, 3, 2, 0],
        [3, 2, 1, 7]
    ]))


@pytest.mark.parametrize("X", X_mods)
def test_indexer_var(X):
    fcfg = OmegaConf.create(
        {
            "_target_": "src.grinch.Filter",
            "key": "var.pick",
        }
    )
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.InplaceIndexer.Config",
            "filter_by": [fcfg],
            "axis": 'var',
        }
    )
    cfg = instantiate(cfg, _convert_='all')
    index = cfg.initialize()
    adata = AnnData(X)
    X_original = adata.X.copy()
    adata.var['pick'] = [2, 0, 1, 1]
    index(adata)
    assert_allclose(X_original, X)
    assert_allclose(adata.X, np.array([
        [1, 0, 2],
        [0, 2, 0],
        [3, 1, 7]
    ]))


@pytest.mark.parametrize("X", X_mods)
def test_indexer_obs_multiple(X):
    fcfg1 = OmegaConf.create(
        {
            "_target_": "src.grinch.Filter",
            "key": "obs.pick",
        }
    )
    fcfg2 = OmegaConf.create(
        {
            "_target_": "src.grinch.Filter",
            "key": "obs.pick2",
        }
    )
    fcfg3 = OmegaConf.create(
        {
            "_target_": "src.grinch.Filter",
            "key": "obs.pick3",
            "cutoff": 0.5,
            "greater_is_True": True,
        }
    )
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.InplaceIndexer.Config",
            "filter_by": [fcfg1, fcfg2, fcfg3]
        }
    )
    cfg = instantiate(cfg, _convert_='all')
    index = cfg.initialize()
    adata = AnnData(X)
    X_original = adata.X.copy()
    adata.obs['pick'] = [0, 1, 1]
    adata.obs['pick2'] = [1, 1, 0]
    adata.obs['pick3'] = [1, 1, 0]
    index(adata)
    assert_allclose(X_original, X)
    assert_allclose(adata.X, np.array([
        [0, 3, 2, 0],
    ]))
