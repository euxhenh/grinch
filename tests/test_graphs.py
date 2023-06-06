import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData
from hydra.utils import instantiate
from omegaconf import OmegaConf

from grinch import OBSP

from ._utils import assert_allclose

X = np.array([[0], [3], [1]], dtype=np.float32)
X_mods = [X, sp.csr_matrix(X), sp.csc_matrix(X)]


@pytest.mark.parametrize("X", X_mods)
def test_knn_connectivity(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.KNNGraph.Config",
            "x_key": "X",
            "n_neighbors": 1,
            "mode": "connectivity",
        }
    )
    cfg = instantiate(cfg)
    knn = cfg.initialize()
    adata = AnnData(X)
    knn(adata)

    ans = np.array([[0, 0, 1], [0, 0, 1], [1, 0, 0]])
    assert_allclose(ans, adata.obsp[OBSP.KNN])
    assert isinstance(adata.obsp[OBSP.KNN], sp.csr_matrix)


@pytest.mark.parametrize("X", X_mods)
def test_knn_distance(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.KNNGraph.Config",
            "x_key": "X",
            "n_neighbors": 1,
        }
    )
    cfg = instantiate(cfg)
    knn = cfg.initialize()
    adata = AnnData(X)
    knn(adata)

    ans = np.array([[0, 0, 1], [0, 0, 2], [1, 0, 0]])
    assert_allclose(ans, adata.obsp[OBSP.KNN])
    assert isinstance(adata.obsp[OBSP.KNN], sp.csr_matrix)
