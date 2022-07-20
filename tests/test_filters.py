import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData
from hydra.utils import instantiate
from omegaconf import OmegaConf

from grinch import OBS, VAR

from ._utils import assert_allclose, to_view

X = np.array([
    [1.0, 4, 0, 2],
    [0, 3, 2, 0],
    [3, 2, 1, 7.0],
], dtype=np.float32)


X_mods = [sp.csr_matrix(X), sp.csc_matrix(X), to_view(X)]


@pytest.mark.parametrize("X", X_mods)
def test_min_counts(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.FilterCells.Config",
            "min_counts": 6,
        }
    )
    cfg = instantiate(cfg)
    filter_cells = cfg.initialize()
    adata = AnnData(X)
    X_original = adata.X.copy()
    filter_cells(adata)
    assert_allclose(X_original, X)
    X_filtered = X[[0, 2]]
    assert_allclose(X_filtered, adata.X)
    assert_allclose(adata.obs[OBS.N_COUNTS], [7, 13])
    assert_allclose(adata.obs[OBS.N_GENES], [3, 4])


@pytest.mark.parametrize("X", X_mods)
def test_max_counts(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.FilterCells.Config",
            "max_counts": 7,
        }
    )
    cfg = instantiate(cfg)
    filter_cells = cfg.initialize()
    adata = AnnData(X)
    X_original = adata.X.copy()
    filter_cells(adata)
    assert_allclose(X_original, X)
    X_filtered = X[[0, 1]]
    assert_allclose(X_filtered, adata.X)
    assert_allclose(adata.obs[OBS.N_COUNTS], [7, 5])
    assert_allclose(adata.obs[OBS.N_GENES], [3, 2])


@pytest.mark.parametrize("X", X_mods)
def test_min_max_counts(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.FilterCells.Config",
            "min_counts": 7,
            "max_counts": 14,
        }
    )
    cfg = instantiate(cfg)
    filter_cells = cfg.initialize()
    adata = AnnData(X)
    X_original = adata.X.copy()
    filter_cells(adata)
    assert_allclose(X_original, X)
    X_filtered = X[[0, 2]]
    assert_allclose(X_filtered, adata.X)
    assert_allclose(adata.obs[OBS.N_COUNTS], [7, 13])
    assert_allclose(adata.obs[OBS.N_GENES], [3, 4])


@pytest.mark.parametrize("X", X_mods)
def test_min_max_genes(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.FilterCells.Config",
            "min_genes": 2,
            "max_genes": 3,
        }
    )
    cfg = instantiate(cfg)
    filter_cells = cfg.initialize()
    adata = AnnData(X)
    X_original = adata.X.copy()
    filter_cells(adata)
    assert_allclose(X_original, X)
    X_filtered = X[[0, 1]]
    assert_allclose(X_filtered, adata.X)
    assert_allclose(adata.obs[OBS.N_COUNTS], [7, 5])
    assert_allclose(adata.obs[OBS.N_GENES], [3, 2])


@pytest.mark.parametrize("X", X_mods)
def test_all(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.FilterCells.Config",
            "min_counts": 6,
            "min_genes": 3,
            "max_genes": 4,
        }
    )
    cfg = instantiate(cfg)
    filter_cells = cfg.initialize()
    adata = AnnData(X)
    X_original = adata.X.copy()
    filter_cells(adata)
    assert_allclose(X_original, X)
    X_filtered = X[[0, 2]]
    assert_allclose(X_filtered, adata.X)
    assert_allclose(adata.obs[OBS.N_COUNTS], [7, 13])
    assert_allclose(adata.obs[OBS.N_GENES], [3, 4])


@pytest.mark.parametrize("X", X_mods)
def test_inplace(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.FilterCells.Config",
            "inplace": False,
            "min_counts": 6,
            "min_genes": 3,
            "max_genes": 4,
        }
    )
    cfg = instantiate(cfg)
    filter_cells = cfg.initialize()
    adata = AnnData(X)
    adata_new = filter_cells(adata)
    X_filtered = X[[0, 2]]
    assert_allclose(X, adata.X)
    assert_allclose(X_filtered, adata_new.X)
    assert_allclose(adata_new.obs[OBS.N_COUNTS], [7, 13])
    assert_allclose(adata_new.obs[OBS.N_GENES], [3, 4])


@pytest.mark.parametrize("X", X_mods)
def test_gene_min_counts(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.FilterGenes.Config",
            "min_counts": 5,
        }
    )
    cfg = instantiate(cfg)
    filter_genes = cfg.initialize()
    adata = AnnData(X)
    X_original = adata.X.copy()
    filter_genes(adata)
    assert_allclose(X_original, X)
    X_filtered = X[:, [1, 3]]
    assert_allclose(X_filtered, adata.X)
    assert_allclose(adata.var[VAR.N_COUNTS], [9, 9])
    assert_allclose(adata.var[VAR.N_CELLS], [3, 2])


@pytest.mark.parametrize("X", X_mods)
def test_gene_max_counts(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.FilterGenes.Config",
            "max_counts": 7,
        }
    )
    cfg = instantiate(cfg)
    filter_genes = cfg.initialize()
    adata = AnnData(X)
    X_original = adata.X.copy()
    filter_genes(adata)
    assert_allclose(X_original, X)
    X_filtered = X[:, [0, 2]]
    assert_allclose(X_filtered, adata.X)
    assert_allclose(adata.var[VAR.N_COUNTS], [4, 3])
    assert_allclose(adata.var[VAR.N_CELLS], [2, 2])


@pytest.mark.parametrize("X", X_mods)
def test_gene_min_max_counts(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.FilterGenes.Config",
            "min_counts": 4,
            "max_counts": 8,
        }
    )
    cfg = instantiate(cfg)
    filter_genes = cfg.initialize()
    adata = AnnData(X)
    X_original = adata.X.copy()
    filter_genes(adata)
    assert_allclose(X_original, X)
    X_filtered = X[:, [0]]
    assert_allclose(X_filtered, adata.X)
    assert_allclose(adata.var[VAR.N_COUNTS], [4])
    assert_allclose(adata.var[VAR.N_CELLS], [2])


@pytest.mark.parametrize("X", X_mods)
def test_min_max_cells(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.FilterGenes.Config",
            "min_cells": 1,
            "max_cells": 2,
        }
    )
    cfg = instantiate(cfg)
    filter_genes = cfg.initialize()
    adata = AnnData(X)
    X_original = adata.X.copy()
    filter_genes(adata)
    assert_allclose(X_original, X)
    X_filtered = X[:, [0, 2, 3]]
    assert_allclose(X_filtered, adata.X)
    assert_allclose(adata.var[VAR.N_COUNTS], [4, 3, 9])
    assert_allclose(adata.var[VAR.N_CELLS], [2, 2, 2])


@pytest.mark.parametrize("X", X_mods)
def test_gene_all(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.FilterGenes.Config",
            "min_counts": 6,
            "min_cells": 1,
            "max_cells": 2,
        }
    )
    cfg = instantiate(cfg)
    filter_genes = cfg.initialize()
    adata = AnnData(X)
    X_original = adata.X.copy()
    filter_genes(adata)
    assert_allclose(X_original, X)
    X_filtered = X[:, [3]]
    assert_allclose(X_filtered, adata.X)
    assert_allclose(adata.var[VAR.N_COUNTS], [9])
    assert_allclose(adata.var[VAR.N_CELLS], [2])


@pytest.mark.parametrize("X", X_mods)
def test_gene_inplace(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.FilterGenes.Config",
            "inplace": False,
            "min_counts": 3,
            "min_cells": 3,
            "max_cells": 3,
        }
    )
    cfg = instantiate(cfg)
    filter_genes = cfg.initialize()
    adata = AnnData(X)
    adata_new = filter_genes(adata)
    X_filtered = X[:, [1]]
    assert_allclose(X, adata.X)
    assert_allclose(X_filtered, adata_new.X)
    assert_allclose(adata_new.var[VAR.N_COUNTS], [9])
    assert_allclose(adata_new.var[VAR.N_CELLS], [3])
