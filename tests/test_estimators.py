import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData
from hydra.utils import instantiate
from omegaconf import OmegaConf

from grinch import OBS, OBSM

from ._utils import assert_allclose, to_view

X = np.array([
    [1, 1, 0, 0, 0],
    [1, 2, 0, 0, 0],
    [0, 1, 5, 6, 5],
    [2, 1, 7, 9, 8],
    [0, 1, 5, 6, 7],
], dtype=np.float)


X_mods = [X, sp.csr_matrix(X), to_view(X)]
X_mods_no_sparse = [X, to_view(X)]


@pytest.mark.parametrize("X", X_mods)
def test_kmeans_x(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.estimators.KMeans.Config",
            "read_key": "X",
            "n_clusters": 2,
            "seed": 42,
        }
    )
    cfg = instantiate(cfg)
    kmeans = cfg.initialize()
    adata = AnnData(X)
    kmeans(adata)
    assert_allclose(adata.obs[OBS.KMEANS], [1, 1, 0, 0, 0])


@pytest.mark.parametrize("X", X_mods_no_sparse)
def test_kmeans_x_pca(X):
    adata = AnnData(X)
    cfg_pca = OmegaConf.create(
        {
            "_target_": "src.grinch.transformers.PCA.Config",
            "n_components": 2,
            "seed": 42,
            "save_key": f"obsm.{OBSM.X_PCA}",
        }
    )
    cfg_pca = instantiate(cfg_pca)
    pca = cfg_pca.initialize()
    pca(adata)

    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.estimators.KMeans.Config",
            "read_key": f"obsm.{OBSM.X_PCA}",
            "n_clusters": 2,
            "seed": 42,
        }
    )
    cfg = instantiate(cfg)
    kmeans = cfg.initialize()
    kmeans(adata)
    assert_allclose(adata.obs[OBS.KMEANS], [1, 1, 0, 0, 0])
