import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData
from hydra.utils import instantiate
from omegaconf import OmegaConf

from grinch.aliases import OBS

from ._utils import to_view

X = np.array([
    [1.0, 4, 4, 2],
    [0, 1, 2, 0],
    [3, 2, 1, 7.0],
], dtype=np.float32)


X_mods = [X, sp.csr_matrix(X), sp.csc_matrix(X), to_view(X)]


@pytest.mark.parametrize("X", X_mods)
def test_indexer_obs(X):
    fcfg = OmegaConf.create(
        {
            "_target_": "src.grinch.Filter",
            "key": "var.pick",
        }
    )
    kmeanscfg = OmegaConf.create(
        {
            "_target_": "src.grinch.KMeans.Config",
            "n_clusters": 2,
            "x_key": "X",
        }
    )
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.IndexProcessor.Config",
            "filter_by": [fcfg],
            "processor": kmeanscfg,
            "axis": "var",
        }
    )
    cfg = instantiate(cfg, _convert_='all')
    index = cfg.create()
    adata = AnnData(X)
    adata.var['pick'] = [0, 1, 1, 0]
    index(adata)
    preds = adata.obs[OBS.KMEANS].to_numpy()
    assert preds[0] != preds[1]
    assert preds[1] == preds[2]
