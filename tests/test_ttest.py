import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData
from hydra.utils import instantiate
from omegaconf import OmegaConf

from grinch import UNS

from ._utils import to_view

X = np.array([
    [1, 5, 4, 45, 62],
    [5, 2, 4, 44, 75],
    [5, 2, 5, 44, 75],
    [75, 62, 4, 4, 2],
    [45, 64, 5, 6, 4],
    [45, 64, 4, 6, 4],
], dtype=np.float32)

label = [0, 0, 0, 1, 1, 1]

X_mods = [X, sp.csr_matrix(X), to_view(X)]


@pytest.mark.parametrize("X", X_mods)
def test_ttest(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.TTest.Config",
            "group_key": "obs.label",
        }
    )
    cfg = instantiate(cfg)
    ttest = cfg.initialize()
    adata = AnnData(X)
    adata.obs['label'] = label

    ttest(adata)
    pvals = adata.uns[UNS.TTEST]['0'][:, 0]
    log2fc = adata.uns[UNS.TTEST]['0'][:, 2]

    assert pvals[0] < 0.05
    assert pvals[1] < 0.05
    assert pvals[2] > 0.5
    assert pvals[3] < 0.05
    assert pvals[4] < 0.05

    assert log2fc[0] < 2
    assert log2fc[1] < 2
    assert log2fc[2] < 2
    assert log2fc[3] > 2
    assert log2fc[4] > 2
