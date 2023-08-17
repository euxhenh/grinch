import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData
from hydra.utils import instantiate
from omegaconf import OmegaConf

from ._utils import assert_allclose, to_view

X = np.array([
    [1, 2, 3, 4],
    [2, 2, 2, 2],
    [0, 0, 0, 1],
    [0, 0, 0, 0],
], dtype=np.float32)


X_mods = [X, sp.csr_matrix(X), to_view(X)]


@pytest.mark.parametrize("X", X_mods)
def test_normalize_total(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.NormalizeTotal.Config",
            "total_counts": 10,
            "save_input": True,
            "input_layer_name": 'x_raw',
        }
    )
    cfg = instantiate(cfg)
    normalize_total = cfg.create()
    adata = AnnData(X.copy())
    normalize_total(adata)
    X_normalized = np.array([
        [1, 2, 3, 4],
        [2.5, 2.5, 2.5, 2.5],
        [0, 0, 0, 10],
        [0, 0, 0, 0],
    ])
    assert_allclose(X_normalized, adata.X)
    assert_allclose(X, adata.layers['x_raw'])


@pytest.mark.parametrize("X", X_mods)
def test_normalize_median(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.NormalizeTotal.Config",
            "total_counts": None,
            "save_input": True,
        }
    )
    cfg = instantiate(cfg)
    normalize_total = cfg.create()
    adata = AnnData(X.copy())
    normalize_total(adata)
    X_normalized = np.array([
        np.array([1, 2, 3, 4]) * (8 / 10),
        [2, 2, 2, 2],
        [0, 0, 0, 8],
        [0, 0, 0, 0],
    ])
    assert_allclose(X_normalized, adata.X)
    assert_allclose(X, adata.layers['pre_normalizetotal'])


@pytest.mark.parametrize("X", X_mods)
def test_log1p(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.Log1P.Config",
            "save_input": True,
        }
    )
    cfg = instantiate(cfg)
    log1p = cfg.create()
    adata = AnnData(X.copy())
    log1p(adata)
    X_logged = np.log1p(X)
    assert_allclose(X_logged, adata.X)
    assert_allclose(X, adata.layers['pre_log1p'])


X_scaled = (X - X.mean(axis=0)) / X.var(axis=0) ** (1/2)


@pytest.mark.parametrize("X", X_mods)
def test_scale(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.Scale.Config",
            "save_input": True,
        }
    )
    cfg = instantiate(cfg)
    scale = cfg.create()
    adata = AnnData(X.copy())
    scale(adata)
    assert_allclose(X_scaled, adata.X)
    assert_allclose(X, adata.layers['pre_scale'])
