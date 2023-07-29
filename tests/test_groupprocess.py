import numpy as np
import pytest
from anndata import AnnData
from hydra.utils import instantiate
from omegaconf import OmegaConf

from grinch import OBS, UNS

X = np.array([
    [5, 4, 6, 0, 1, 0],
    [5, 5, 4, 1, 1, 0],
    [0, 0, 1, 7, 8, 9],
    [1, 1, 0, 7, 7, 8],
    [9, 9, 9, 8, 7, 9],
    [8, 7, 9, 8, 7, 8]
]).astype(np.float32)


@pytest.mark.parametrize("X", [X])
def test_groupby(X):
    cfg = OmegaConf.create({
        "_target_": "src.grinch.KMeans.Config",
        "x_key": "X",
        "labels_key": "uns.kmeans",
        "n_clusters": 2,
    })
    # cfg = instantiate(cfg)
    gcfg = OmegaConf.create({
        "_target_": "src.grinch.GroupProcess.Config",
        "processor": cfg,
        "group_key": "obs.mem",
    })
    adata = AnnData(X)
    adata.obs['mem'] = [0, 0, 0, 1, 1, 1]
    adata.obs['ct'] = ["p1", "p2", "p2", "p2", "p1", "p1"]
    # gcfg = instantiate(gcfg, _convert_='partial')
    gcfg = instantiate(gcfg)
    groupprocess = gcfg.create()
    groupprocess(adata)
    gr0 = adata.uns['g-mem']['0']
    gr1 = adata.uns['g-mem']['1']
    assert gr0[OBS.KMEANS][0] == gr0[OBS.KMEANS][1]
    assert gr0[OBS.KMEANS][0] != gr0[OBS.KMEANS][2]
    assert gr1[OBS.KMEANS][0] != gr1[OBS.KMEANS][1]
    assert gr1[OBS.KMEANS][1] == gr1[OBS.KMEANS][2]


@pytest.mark.parametrize("X", [X])
def test_groupby_ttest(X):
    cfg = OmegaConf.create({
        "_target_": "src.grinch.TTest.Config",
        "group_key": "obs.ct",
    })
    # cfg = instantiate(cfg)
    gcfg = OmegaConf.create({
        "_target_": "src.grinch.GroupProcess.Config",
        "processor": cfg,
        "group_key": "obs.mem",
    })
    adata = AnnData(X)
    adata.obs['mem'] = [0, 0, 0, 1, 1, 1]
    adata.obs['ct'] = ["p1", "p2", "p2", "p2", "p1", "p1"]
    # gcfg = instantiate(gcfg, _convert_='partial')
    gcfg = instantiate(gcfg)
    groupprocess = gcfg.create()
    groupprocess(adata)
    gr0 = adata.uns['g-mem']['0']
    gr1 = adata.uns['g-mem']['1']
    pvals = gr0[UNS.TTEST]['ct-p1']['pvals'].to_numpy()
    assert pvals.min() > 0.2
    pvals = gr1[UNS.TTEST]['ct-p1']['pvals'].to_numpy()
    assert pvals[:3].max() < 0.1


@pytest.mark.parametrize("X", [X])
def test_nested_groupby(X):
    cfg = OmegaConf.create({
        "_target_": "src.grinch.KMeans.Config",
        "x_key": "X",
        "labels_key": "uns.kmeans",
        "n_clusters": 1,
    })
    ginnercfg = OmegaConf.create({
        "_target_": "src.grinch.GroupProcess.Config",
        "processor": cfg,
        "group_key": "obs.ct",
    })
    # cfg = instantiate(cfg)
    gcfg = OmegaConf.create({
        "_target_": "src.grinch.GroupProcess.Config",
        "processor": ginnercfg,
        "group_key": "obs.mem",
    })
    adata = AnnData(X)
    adata.obs['mem'] = [0, 0, 0, 1, 1, 1]
    adata.obs['ct'] = ["p1", "p2", "p2", "p2", "p1", "p1"]
    # gcfg = instantiate(gcfg, _convert_='partial')
    gcfg = instantiate(gcfg)
    groupprocess = gcfg.create()
    groupprocess(adata)
    gr0 = adata.uns['g-mem']['0']
    gr1 = adata.uns['g-mem']['1']
    assert len(gr0['g-ct']['p1'][OBS.KMEANS]) == 1
    assert len(gr0['g-ct']['p2'][OBS.KMEANS]) == 2
    assert gr0['g-ct']['p1'][OBS.KMEANS][0] == 0
    assert gr0['g-ct']['p2'][OBS.KMEANS][0] == 0
    assert gr0['g-ct']['p2'][OBS.KMEANS][1] == 0
    assert len(gr1['g-ct']['p1'][OBS.KMEANS]) == 2
    assert len(gr1['g-ct']['p2'][OBS.KMEANS]) == 1
    assert gr1['g-ct']['p1'][OBS.KMEANS][0] == 0
    assert gr1['g-ct']['p1'][OBS.KMEANS][1] == 0
    assert gr1['g-ct']['p2'][OBS.KMEANS][0] == 0
