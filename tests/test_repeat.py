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
    [3, 2, 1, 8.0],
], dtype=np.float32)


X_mods = [X, sp.csr_matrix(X), to_view(X)]


@pytest.mark.parametrize("X", X_mods)
def test_repeat(X):
    processor = OmegaConf.create(
        {
            "_target_": "src.grinch.KMeans.Config",
            "x_key": "X",
        }
    )
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.RepeatProcessor.Config",
            "processor_cfg": processor,
            "repeat_var": "n_clusters",
            "repeat_vals": (reps := [2, 3, 4]),
        }
    )
    cfg = instantiate(cfg, _convert_='all')
    rep = cfg.create()
    adata = AnnData(X)
    adata.var['pick'] = [0, 1, 1, 0]
    rep(adata)

    for rep in reps:
        assert len(np.unique(adata.obs[f"r-n_clusters-{rep}-{OBS.KMEANS}"].to_numpy())) == rep
