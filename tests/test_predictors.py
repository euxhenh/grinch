import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData
from hydra.utils import instantiate
from omegaconf import OmegaConf

from grinch import OBS, OBSM, OBSP

from ._utils import assert_allclose, to_view

X = np.array([
    [6, 8, 0, 0, 0],
    [5, 7, 0, 0, 0],
    [6, 8, 1, 0, 0],
    [4, 7, 0, 0, 0],
    [0, 1, 5, 6, 8],
    [2, 1, 7, 9, 8],
    [0, 1, 8, 6, 7],
    [0, 1, 8, 6, 5],
    [2, 1, 7, 8, 8],
    [0, 1, 9, 6, 7],
], dtype=np.float32)

K_plus = 4

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
    kmeans = cfg.create()
    adata = AnnData(X)
    kmeans(adata)
    outp = adata.obs[OBS.KMEANS].to_numpy()
    assert np.unique(outp[:K_plus]).size == 1
    assert np.unique(outp[K_plus:]).size == 1
    assert outp[0] != outp[-1]


@pytest.mark.parametrize("X", X_mods_no_sparse)
def test_kmeans_x_pca(X):
    adata = AnnData(X)
    cfg_pca = OmegaConf.create(
        {
            "_target_": "src.grinch.PCA.Config",
            "n_components": 2,
            "seed": 42,
            "write_key": f"obsm.{OBSM.X_PCA}",
        }
    )
    cfg_pca = instantiate(cfg_pca)
    pca = cfg_pca.create()
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
    kmeans = cfg.create()
    kmeans(adata)
    outp = adata.obs[OBS.KMEANS].to_numpy()
    assert np.unique(outp[:K_plus]).size == 1
    assert np.unique(outp[K_plus:]).size == 1
    assert outp[0] != outp[-1]

    adata_test = AnnData(X_test)
    pca.transform(adata_test)
    kmeans.predict(adata_test)
    outp = adata_test.obs[OBS.KMEANS].to_numpy()
    assert outp[0] == outp[2]
    assert outp[0] != outp[1]


@pytest.mark.parametrize("X", X_mods_no_sparse)
def test_gmix_x(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.GaussianMixture.Config",
            "x_key": "X",
            "n_components": 2,
            "seed": 42,
        }
    )
    cfg = instantiate(cfg)
    kmeans = cfg.create()
    adata = AnnData(X)
    kmeans(adata)
    outp = adata.obs[OBS.GAUSSIAN_MIXTURE].to_numpy()
    assert np.unique(outp[:K_plus]).size == 1
    assert np.unique(outp[K_plus:]).size == 1
    assert outp[0] != outp[-1]
    proba = adata.obsm[OBSM.GAUSSIAN_MIXTURE_PROBA]
    assert (proba[:K_plus, 0] > proba[:K_plus, 1]).all()
    assert (proba[K_plus:, 0] < proba[K_plus:, 1]).all()


@pytest.mark.parametrize("X", X_mods_no_sparse)
@pytest.mark.parametrize(
    "classifier, key", [("LogisticRegression", OBS.LOG_REG),
                        ("XGBClassifier", OBS.XGB_CLASSIFIER)]
)
def test_classifiers_x(X, classifier, key):
    adata = AnnData(X)
    cfg_pca = OmegaConf.create(
        {
            "_target_": "src.grinch.PCA.Config",
            "n_components": 2,
            "seed": 42,
            "write_key": f"obsm.{OBSM.X_PCA}",
        }
    )
    cfg_pca = instantiate(cfg_pca)
    pca = cfg_pca.create()
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
    kmeans = cfg_kmeans.create()
    kmeans(adata)

    if classifier == "XGBClassifier":
        kwargs = {
            'n_estimators': 1,
            'max_depth': 1,
        }
    else:
        kwargs = {}

    cfg = OmegaConf.create(
        {
            "_target_": f"src.grinch.{classifier}.Config",
            "x_key": f"obsm.{OBSM.X_PCA}",
            "y_key": f"obs.{OBS.KMEANS}",
            "seed": 42,
            "labels_key": f"obs.{key}",
            **kwargs,
        }
    )
    # Need to start using convert all for lists and dicts
    cfg = instantiate(cfg, _convert_='all')
    lr = cfg.create()
    lr(adata)
    outp = adata.obs[key].to_numpy()
    assert np.unique(outp[:K_plus]).size == 1
    assert np.unique(outp[K_plus:]).size == 1
    assert outp[0] != outp[-1]

    adata_test = AnnData(X_test)
    pca.transform(adata_test)
    lr.predict(adata_test)
    outp = adata_test.obs[key].to_numpy()
    assert outp[0] == outp[2]
    assert outp[0] != outp[1]


@pytest.mark.parametrize("X", X_mods)
def test_leiden(X):
    adata = AnnData(X)
    cfg_knn = OmegaConf.create(
        {
            "_target_": "src.grinch.KNNGraph.Config",
            "x_key": "X",
            "n_neighbors": 3,
        }
    )
    cfg_knn = instantiate(cfg_knn)
    knn = cfg_knn.create()
    knn(adata)

    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.Leiden.Config",
            "x_key": f"obsp.{OBSP.KNN_DISTANCE}",
            "seed": 42,
            "resolution": 0.5,
        }
    )
    cfg = instantiate(cfg)
    leiden = cfg.create()
    leiden(adata)
    pred = adata.obs[OBS.LEIDEN].to_numpy()
    true = np.ones(X.shape[0])
    true[:K_plus] = 0
    if pred[0] == 1:
        true = 1 - true
    assert_allclose(pred, true)

    centroids = {
        pred[0]: np.ravel(X[:K_plus].mean(axis=0)),
        1 - pred[0]: np.ravel(X[K_plus:].mean(axis=0)),
    }
    pred_centroid = adata.uns['leiden_']["cluster_centers_"]
    assert_allclose(centroids[0], pred_centroid['0'])
    assert_allclose(centroids[1], pred_centroid['1'])
