import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData
from hydra.utils import instantiate
from omegaconf import OmegaConf

from grinch import UNS

from ._utils import assert_allclose, to_view

X = np.array([
    [1, 5, 4, 45, 62],
    [5, 2, 4, 44, 75],
    [5, 2, 4, 44, 75],
    [75, 62, 4, 4, 2],
    [65, 64, 4, 6, 4],
    [65, 64, 4.5, 6, 4],
], dtype=np.float32)

X_mods = [X, sp.csr_matrix(X), to_view(X)]


@pytest.mark.parametrize("X", X_mods)
def test_bimodal_test(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.BimodalTest.Config",
        }
    )
    cfg = instantiate(cfg)
    bimodal = cfg.initialize()
    adata = AnnData(X)

    bimodal(adata)
    is_sig = adata.uns[UNS.BIMODALTEST_].qvals <= 0.05

    assert_allclose([True, True, False, True, True], is_sig)
