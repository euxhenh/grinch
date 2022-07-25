import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.decomposition import PCA, TruncatedSVD
from umap import UMAP

from grinch import OBSM

from ._utils import assert_allclose, to_view

SEED = 42

X = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 1],
    [2, 3, 4, 5, 6],
    [7, 8, 9, 1, 2],
], dtype=np.float32)

rng = np.random.default_rng(SEED)
X_rand = rng.integers(0, 15, (10, 20)).astype(np.float32)

X_mods = [X, to_view(X), X_rand, to_view(X_rand)]
X_mods_sparse = [sp.csr_array(X), sp.csr_array(X_rand)]


@pytest.mark.parametrize("X", X_mods)
def test_pca(X):
    cfg = OmegaConf.create(
        {
            '_target_': 'src.grinch.PCA.Config',
            'n_components': 3,
            'seed': SEED,
        }
    )
    cfg = instantiate(cfg)
    pca = cfg.initialize()
    pca_sk = PCA(n_components=3, random_state=SEED)
    adata = AnnData(X)
    pca(adata)
    x_emb_sk = pca_sk.fit_transform(X)

    assert adata.uns[OBSM.X_PCA]['n_components'] == 3
    assert adata.uns[OBSM.X_PCA]['seed'] == SEED
    assert_allclose(adata.obsm[OBSM.X_PCA], x_emb_sk, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("X", X_mods + X_mods_sparse)
def test_truncated(X):
    cfg = OmegaConf.create(
        {
            '_target_': 'src.grinch.TruncatedSVD.Config',
            'n_components': 3,
            'seed': SEED,
        }
    )
    cfg = instantiate(cfg)
    tsvd = cfg.initialize()
    tsvd_sk = TruncatedSVD(n_components=3, random_state=SEED)
    adata = AnnData(X)
    tsvd(adata)
    x_emb_sk = tsvd_sk.fit_transform(X)

    assert adata.uns[OBSM.X_TRUNCATED_SVD]['n_components'] == 3
    assert adata.uns[OBSM.X_TRUNCATED_SVD]['seed'] == SEED
    assert_allclose(adata.obsm[OBSM.X_TRUNCATED_SVD], x_emb_sk, rtol=1e-4, atol=1e-4)


# @pytest.mark.parametrize("X", X_mods + X_mods_sparse)
@pytest.mark.parametrize("X", X_mods)
def test_umap(X):
    cfg = OmegaConf.create(
        {
            '_target_': 'src.grinch.UMAP.Config',
            'n_components': 2,
            'seed': SEED,
            'spread': 0.8,
            'n_neighbors': 2,
        }
    )
    cfg = instantiate(cfg)
    umap_up = UMAP(
        n_components=2,
        n_neighbors=2,
        spread=0.8,
        random_state=SEED,
        transform_seed=SEED
    )
    adata = AnnData(X)
    up = cfg.initialize()
    up(adata)
    x_emb_up = umap_up.fit_transform(X)

    assert adata.uns[OBSM.X_UMAP]['n_components'] == 2
    assert adata.uns[OBSM.X_UMAP]['seed'] == SEED
    assert adata.uns[OBSM.X_UMAP]['n_neighbors'] == 2
    assert adata.uns[OBSM.X_UMAP]['kwargs']['transform_seed'] == SEED
    assert_allclose(adata.obsm[OBSM.X_UMAP], x_emb_up, rtol=1e-4, atol=1e-4)
