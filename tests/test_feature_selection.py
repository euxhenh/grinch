import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData
from hydra.utils import instantiate
from omegaconf import OmegaConf

from grinch import UNS, VAR

from ._utils import assert_allclose, to_view

X = np.array([
    [1, 1, 1, 1],
    [1, 1, 0, 1],
    [1, 0, 1, 1],
    [1, 0, 0, 1]
], dtype=np.float32)

X_mods = [X, sp.csr_matrix(X), to_view(X)]


@pytest.mark.parametrize("X", X_mods)
def test_phenotype_cover(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.PhenotypeCover.Config",
            "y_key": "obs.anns",
            "coverage": 10,
        }
    )
    cfg = instantiate(cfg)
    gpc = cfg.initialize()
    adata = AnnData(X)
    adata.obs['anns'] = [0, 1, 0, 1]
    gpc(adata)
    assert_allclose([False, False, True, False], adata.var[f"{VAR.PCOVER_M}"])
    assert_allclose([0, 0, 1, 0], adata.var[f"{VAR.PCOVER_O}"])
    assert_allclose(adata.uns[f"{UNS.PCOVER}"]['n_elements_remaining_per_iter_'], [0])
