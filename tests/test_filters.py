import unittest

import numpy as np
from anndata import AnnData
from hydra.utils import instantiate
from numpy.testing import assert_allclose
from omegaconf import OmegaConf

from grinch import ADK


class TestFilterCells(unittest.TestCase):
    X = np.array([
        [1.0, 4, 0, 2],
        [0, 3, 2, 0],
        [3, 2, 1, 7.0],
    ], dtype=np.float32)

    def test_min_counts(self):
        cfg = OmegaConf.create(
            {
                "_target_": "src.grinch.filters.FilterCells.Config",
                "min_counts": 6,
            }
        )
        cfg = instantiate(cfg)
        filter_cells = cfg.initialize()
        adata = AnnData(self.X)
        filter_cells(adata)
        X_filtered = self.X[[0, 2]]
        assert_allclose(X_filtered, adata.X)
        assert_allclose(adata.obs[ADK.N_COUNTS], [7, 13])
        assert_allclose(adata.obs[ADK.N_GENES], [3, 4])

    def test_max_counts(self):
        cfg = OmegaConf.create(
            {
                "_target_": "src.grinch.filters.FilterCells.Config",
                "max_counts": 7,
            }
        )
        cfg = instantiate(cfg)
        filter_cells = cfg.initialize()
        adata = AnnData(self.X)
        filter_cells(adata)
        X_filtered = self.X[[0, 1]]
        assert_allclose(X_filtered, adata.X)
        assert_allclose(adata.obs[ADK.N_COUNTS], [7, 5])
        assert_allclose(adata.obs[ADK.N_GENES], [3, 2])

    def test_min_max_counts(self):
        cfg = OmegaConf.create(
            {
                "_target_": "src.grinch.filters.FilterCells.Config",
                "min_counts": 7,
                "max_counts": 14,
            }
        )
        cfg = instantiate(cfg)
        filter_cells = cfg.initialize()
        adata = AnnData(self.X)
        filter_cells(adata)
        X_filtered = self.X[[0, 2]]
        assert_allclose(X_filtered, adata.X)
        assert_allclose(adata.obs[ADK.N_COUNTS], [7, 13])
        assert_allclose(adata.obs[ADK.N_GENES], [3, 4])

    def test_min_max_genes(self):
        cfg = OmegaConf.create(
            {
                "_target_": "src.grinch.filters.FilterCells.Config",
                "min_genes": 2,
                "max_genes": 3,
            }
        )
        cfg = instantiate(cfg)
        filter_cells = cfg.initialize()
        adata = AnnData(self.X)
        filter_cells(adata)
        X_filtered = self.X[[0, 1]]
        assert_allclose(X_filtered, adata.X)
        assert_allclose(adata.obs[ADK.N_COUNTS], [7, 5])
        assert_allclose(adata.obs[ADK.N_GENES], [3, 2])

    def test_all(self):
        cfg = OmegaConf.create(
            {
                "_target_": "src.grinch.filters.FilterCells.Config",
                "min_counts": 6,
                "min_genes": 3,
                "max_genes": 4,
            }
        )
        cfg = instantiate(cfg)
        filter_cells = cfg.initialize()
        adata = AnnData(self.X)
        filter_cells(adata)
        X_filtered = self.X[[0, 2]]
        assert_allclose(X_filtered, adata.X)
        assert_allclose(adata.obs[ADK.N_COUNTS], [7, 13])
        assert_allclose(adata.obs[ADK.N_GENES], [3, 4])

    def test_inplace(self):
        cfg = OmegaConf.create(
            {
                "_target_": "src.grinch.filters.FilterCells.Config",
                "inplace": False,
                "min_counts": 6,
                "min_genes": 3,
                "max_genes": 4,
            }
        )
        cfg = instantiate(cfg)
        filter_cells = cfg.initialize()
        adata = AnnData(self.X)
        adata_new = filter_cells(adata)
        X_filtered = self.X[[0, 2]]
        assert_allclose(self.X, adata.X)
        assert_allclose(X_filtered, adata_new.X)
        assert_allclose(adata_new.obs[ADK.N_COUNTS], [7, 13])
        assert_allclose(adata_new.obs[ADK.N_GENES], [3, 4])


class TestFilterGenes(unittest.TestCase):
    X = np.array([
        [1.0, 4, 0, 2],
        [0, 3, 2, 0],
        [3, 2, 1, 7.0],
    ], dtype=np.float32)

    def test_min_counts(self):
        cfg = OmegaConf.create(
            {
                "_target_": "src.grinch.filters.FilterGenes.Config",
                "min_counts": 5,
            }
        )
        cfg = instantiate(cfg)
        filter_genes = cfg.initialize()
        adata = AnnData(self.X)
        filter_genes(adata)
        X_filtered = self.X[:, [1, 3]]
        assert_allclose(X_filtered, adata.X)
        assert_allclose(adata.var[ADK.N_COUNTS], [9, 9])
        assert_allclose(adata.var[ADK.N_CELLS], [3, 2])

    def test_max_counts(self):
        cfg = OmegaConf.create(
            {
                "_target_": "src.grinch.filters.FilterGenes.Config",
                "max_counts": 7,
            }
        )
        cfg = instantiate(cfg)
        filter_genes = cfg.initialize()
        adata = AnnData(self.X)
        filter_genes(adata)
        X_filtered = self.X[:, [0, 2]]
        assert_allclose(X_filtered, adata.X)
        assert_allclose(adata.var[ADK.N_COUNTS], [4, 3])
        assert_allclose(adata.var[ADK.N_CELLS], [2, 2])

    def test_min_max_counts(self):
        cfg = OmegaConf.create(
            {
                "_target_": "src.grinch.filters.FilterGenes.Config",
                "min_counts": 4,
                "max_counts": 8,
            }
        )
        cfg = instantiate(cfg)
        filter_genes = cfg.initialize()
        adata = AnnData(self.X)
        filter_genes(adata)
        X_filtered = self.X[:, [0]]
        assert_allclose(X_filtered, adata.X)
        assert_allclose(adata.var[ADK.N_COUNTS], [4])
        assert_allclose(adata.var[ADK.N_CELLS], [2])

    def test_min_max_genes(self):
        cfg = OmegaConf.create(
            {
                "_target_": "src.grinch.filters.FilterGenes.Config",
                "min_cells": 1,
                "max_cells": 2,
            }
        )
        cfg = instantiate(cfg)
        filter_genes = cfg.initialize()
        adata = AnnData(self.X)
        filter_genes(adata)
        X_filtered = self.X[:, [0, 2, 3]]
        assert_allclose(X_filtered, adata.X)
        assert_allclose(adata.var[ADK.N_COUNTS], [4, 3, 9])
        assert_allclose(adata.var[ADK.N_CELLS], [2, 2, 2])

    def test_all(self):
        cfg = OmegaConf.create(
            {
                "_target_": "src.grinch.filters.FilterGenes.Config",
                "min_counts": 6,
                "min_cells": 1,
                "max_cells": 2,
            }
        )
        cfg = instantiate(cfg)
        filter_genes = cfg.initialize()
        adata = AnnData(self.X)
        filter_genes(adata)
        X_filtered = self.X[:, [3]]
        assert_allclose(X_filtered, adata.X)
        assert_allclose(adata.var[ADK.N_COUNTS], [9])
        assert_allclose(adata.var[ADK.N_CELLS], [2])

    def test_inplace(self):
        cfg = OmegaConf.create(
            {
                "_target_": "src.grinch.filters.FilterGenes.Config",
                "inplace": False,
                "min_counts": 3,
                "min_cells": 3,
                "max_cells": 3,
            }
        )
        cfg = instantiate(cfg)
        filter_genes = cfg.initialize()
        adata = AnnData(self.X)
        adata_new = filter_genes(adata)
        X_filtered = self.X[:, [1]]
        assert_allclose(self.X, adata.X)
        assert_allclose(X_filtered, adata_new.X)
        assert_allclose(adata_new.var[ADK.N_COUNTS], [9])
        assert_allclose(adata_new.var[ADK.N_CELLS], [3])
