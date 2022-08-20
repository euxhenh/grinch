import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split

from ._utils import assert_allclose, to_view

X = np.array([
    [6, 8, 0, 0, 0],
    [5, 7, 0, 0, 0],
    [0, 1, 5, 6, 5],
    [2, 1, 7, 9, 8],
    [0, 1, 5, 6, 7],
], dtype=np.float32)

X_mods = [X, sp.csr_matrix(X), to_view(X)]


@pytest.mark.parametrize("X", X_mods)
def test_splitter(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.Splitter.Config",
            "shuffle": False,
        }
    )
    cfg = instantiate(cfg)
    splitter = cfg.initialize()
    adata = AnnData(X)
    datasplitter = splitter(adata)
    assert_allclose(adata.X, datasplitter.TRAIN_SPLIT.X)


@pytest.mark.parametrize("X", X_mods)
def test_splitter_val(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.Splitter.Config",
            "val_fraction": 2/5,
            "seed": 42,
        }
    )
    cfg = instantiate(cfg)
    splitter = cfg.initialize()
    adata = AnnData(X)
    datasplitter = splitter(adata)
    idx_train, idx_val = train_test_split(np.arange(X.shape[0]), test_size=2/5, random_state=42)
    assert len(idx_train) == 3
    assert len(idx_val) == 2
    assert_allclose(adata.X[idx_train], datasplitter.TRAIN_SPLIT.X)
    assert_allclose(adata.X[idx_val], datasplitter.VAL_SPLIT.X)


@pytest.mark.parametrize("X", X_mods)
def test_splitter_test(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.Splitter.Config",
            "test_fraction": 3/5,
            "seed": 42,
        }
    )
    cfg = instantiate(cfg)
    splitter = cfg.initialize()
    adata = AnnData(X)
    datasplitter = splitter(adata)
    idx_train, idx_test = train_test_split(np.arange(X.shape[0]), test_size=3/5, random_state=42)
    assert len(idx_train) == 2
    assert len(idx_test) == 3
    assert_allclose(adata.X[idx_train], datasplitter.TRAIN_SPLIT.X)
    assert_allclose(adata.X[idx_test], datasplitter.TEST_SPLIT.X)


@pytest.mark.parametrize("X", X_mods)
def test_splitter_val_test(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.Splitter.Config",
            "val_fraction": 2/5,
            "test_fraction": 1/5,
            "seed": 42,
        }
    )
    cfg = instantiate(cfg)
    splitter = cfg.initialize()
    adata = AnnData(X)
    datasplitter = splitter(adata)
    idx_train, idx_val = train_test_split(np.arange(X.shape[0]), test_size=2/5, random_state=42)
    idx_train, idx_test = train_test_split(idx_train, test_size=1/3, random_state=42)
    assert len(idx_train) == 2
    assert len(idx_val) == 2
    assert len(idx_test) == 1
    assert len(set(idx_train).union(idx_val).union(idx_test)) == 5
    assert_allclose(adata.X[idx_train], datasplitter.TRAIN_SPLIT.X)
    assert_allclose(adata.X[idx_val], datasplitter.VAL_SPLIT.X)
    assert_allclose(adata.X[idx_test], datasplitter.TEST_SPLIT.X)
