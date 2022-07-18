import unittest

import numpy as np
import scipy.sparse as sp
from _utils import assert_allclose, parametrize, to_view
from anndata import AnnData
from hydra.utils import instantiate
from omegaconf import OmegaConf

X = np.array([
    [1, 2, 3, 4],
    [2, 2, 2, 2],
    [0, 0, 0, 1],
    [0, 0, 0, 0],
], dtype=np.float32)


class TestNormalizeTotal(unittest.TestCase):

    @parametrize([X], to_apply=[sp.csr_matrix, to_view])
    def test_normalize_total(self, X):
        cfg = OmegaConf.create(
            {
                "_target_": "src.grinch.transforms.NormalizeTotal.Config",
                "total_counts": 10,
            }
        )
        cfg = instantiate(cfg)
        normalize_total = cfg.initialize()
        adata = AnnData(X)
        normalize_total(adata)
        X_normalized = np.array([
            [1, 2, 3, 4],
            [2.5, 2.5, 2.5, 2.5],
            [0, 0, 0, 10],
            [0, 0, 0, 0],
        ])
        assert_allclose(X_normalized, adata.X)


class TestLog1P(unittest.TestCase):

    @parametrize([X], to_apply=[sp.csr_matrix, to_view])
    def test_log1p(self, X):
        cfg = OmegaConf.create(
            {
                "_target_": "src.grinch.transforms.Log1P.Config",
            }
        )
        cfg = instantiate(cfg)
        log1p = cfg.initialize()
        adata = AnnData(X)
        xx = X.copy()
        log1p(adata)
        X_logged = np.log1p(X)
        assert_allclose(xx, X)
        assert_allclose(X_logged, adata.X)
