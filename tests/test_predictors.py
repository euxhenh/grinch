import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData
from hydra.utils import instantiate
from omegaconf import OmegaConf

from grinch import OBS, OBSM, OBSP, UNS

from ._utils import assert_allclose, to_view

X = np.array([
    [6, 8, 0, 0, 0],
    [5, 7, 0, 0, 0],
    [0, 1, 5, 6, 5],
    [2, 1, 7, 9, 8],
    [0, 1, 5, 6, 7],
], dtype=np.float32)

X_test = np.array([
    [0, -1, 5, 6, 5],
    [5, 6, 0, 1, 0],
    [1, 0, 5, 5, 5]
], dtype=np.float32)


X_mods = [X, sp.csr_matrix(X), to_view(X)]
X_mods_no_sparse = [X, to_view(X)]


@pytest.mark.parametrize("X", X_mods)
def test_kmeans_x(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.KMeans.Config",
            "x_key": "X",
            "n_clusters": 2,
            "seed": 42,
        }
    )
    cfg = instantiate(cfg)
    kmeans = cfg.initialize()
    adata = AnnData(X)
    kmeans(adata)
    outp = adata.obs[OBS.KMEANS]
    assert np.unique(outp[:2]).size == 1
    assert np.unique(outp[2:]).size == 1
    assert outp[0] != outp[-1]


@pytest.mark.parametrize("X", X_mods_no_sparse)
def test_kmeans_x_pca(X):
    adata = AnnData(X)
    cfg_pca = OmegaConf.create(
        {
            "_target_": "src.grinch.PCA.Config",
            "n_components": 2,
            "seed": 42,
            "x_emb_key": f"obsm.{OBSM.X_PCA}",
        }
    )
    cfg_pca = instantiate(cfg_pca)
    pca = cfg_pca.initialize()
    pca(adata)

    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.KMeans.Config",
            "x_key": f"obsm.{OBSM.X_PCA}",
            "n_clusters": 2,
            "seed": 42,
        }
    )
    cfg = instantiate(cfg)
    kmeans = cfg.initialize()
    kmeans(adata)
    outp = adata.obs[OBS.KMEANS]
    assert np.unique(outp[:2]).size == 1
    assert np.unique(outp[2:]).size == 1
    assert outp[0] != outp[-1]

    adata_test = AnnData(X_test)
    pca.transform(adata_test)
    kmeans.predict(adata_test)
    outp = adata_test.obs[OBS.KMEANS]
    assert outp[0] == outp[2]
    assert outp[0] != outp[1]


@pytest.mark.parametrize("X", X_mods_no_sparse)
def test_log_reg_x(X):
    adata = AnnData(X)
    cfg_pca = OmegaConf.create(
        {
            "_target_": "src.grinch.PCA.Config",
            "n_components": 2,
            "seed": 42,
            "x_emb_key": f"obsm.{OBSM.X_PCA}",
        }
    )
    cfg_pca = instantiate(cfg_pca)
    pca = cfg_pca.initialize()
    pca(adata)

    cfg_kmeans = OmegaConf.create(
        {
            "_target_": "src.grinch.KMeans.Config",
            "x_key": f"obsm.{OBSM.X_PCA}",
            "n_clusters": 2,
            "seed": 42,
        }
    )
    cfg_kmeans = instantiate(cfg_kmeans)
    kmeans = cfg_kmeans.initialize()
    kmeans(adata)

    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.LogisticRegression.Config",
            "x_key": f"obsm.{OBSM.X_PCA}",
            "y_key": f"obs.{OBS.KMEANS}",
            "seed": 42,
            "labels_key": f"obs.{OBS.LOG_REG}",
        }
    )
    # Need to start using convert all for lists and dicts
    cfg = instantiate(cfg, _convert_='all')
    lr = cfg.initialize()
    lr(adata)
    outp = adata.obs[OBS.LOG_REG]
    assert np.unique(outp[:2]).size == 1
    assert np.unique(outp[2:]).size == 1
    assert outp[0] != outp[-1]

    adata_test = AnnData(X_test)
    pca.transform(adata_test)
    lr.predict(adata_test)
    outp = adata_test.obs[OBS.LOG_REG]
    assert outp[0] == outp[2]
    assert outp[0] != outp[1]


@pytest.mark.parametrize("X", X_mods)
def test_leiden(X):
    adata = AnnData(X)
    cfg_knn = OmegaConf.create(
        {
            "_target_": "src.grinch.KNNGraph.Config",
            "x_key": "X",
            "n_neighbors": 1,
        }
    )
    cfg_knn = instantiate(cfg_knn)
    knn = cfg_knn.initialize()
    knn(adata)

    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.Leiden.Config",
            "x_key": f"obsp.{OBSP.KNN_DISTANCES}",
            "seed": 42,
        }
    )
    cfg = instantiate(cfg)
    leiden = cfg.initialize()
    leiden(adata)
    pred = adata.obs[OBS.LEIDEN]
    true = np.array([0, 0, 1, 1, 1])
    if pred[0] == 1:
        true = 1 - true
    assert_allclose(pred, true)

    centroids = {
        pred[0]: np.ravel(X[:2].mean(axis=0)),
        1 - pred[0]: np.ravel(X[2:].mean(axis=0)),
    }
    pred_centroid = adata.uns[UNS.LEIDEN_]["cluster_centers_"]
    assert_allclose(centroids[0], pred_centroid['0'])
    assert_allclose(centroids[1], pred_centroid['1'])
