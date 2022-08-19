import numpy as np
import pytest
from anndata import AnnData
from hydra.utils import instantiate
from numpy.testing import assert_equal
from omegaconf import OmegaConf

from grinch import OBS

X = np.array([
    [5, 4, 6, 0, 1, 0],
    [5, 5, 4, 1, 1, 0],
    [0, 0, 1, 7, 8, 9],
    [1, 1, 0, 7, 7, 8],
    [9, 9, 9, 8, 7, 9],
    [8, 7, 9, 8, 7, 8]
]).astype(np.float32)

adata = AnnData(X)
adata.obs['mem'] = [0, 0, 1, 1, 2, 2]


@pytest.mark.parametrize("adata", [adata])
def test_groupby(adata):
    cfg = OmegaConf.create({
        "_target_": "src.grinch.KMeans.Config",
        "x_key": "X",
        "n_clusters": 2,
    })
    # cfg = instantiate(cfg)
    gcfg = OmegaConf.create({
        "_target_": "src.grinch.GroupProcess.Config",
        "processor": cfg,
        "group_key": "obs.mem",
    })
    # gcfg = instantiate(gcfg, _convert_='partial')
    gcfg = instantiate(gcfg)
    groupprocess = gcfg.initialize()

    obs_names = adata.obs_names.to_numpy().copy().astype(str)
    groupprocess(adata)

    for x in range(0, 6, 2):
        assert adata.obs[OBS.KMEANS][x] != adata.obs[OBS.KMEANS][x + 1]

    assert_equal(adata.obs_names.to_numpy().astype(str), obs_names)
