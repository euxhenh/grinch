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
    [5, 2, 5, 44, 75],
    [75, 62, 4, 4, 2],
    [45, 64, 5, 6, 4],
    [45, 64, 4, 6, 4],
], dtype=np.float32)

label = [0, 0, 0, 1, 1, 1]

X_mods = [X, sp.csr_matrix(X), to_view(X)]
tests = [("TTest", UNS.TTEST),
         ("KSTest", UNS.KSTEST),
         ("RankSum", UNS.RANK_SUM)]


# Test all combinations
@pytest.mark.parametrize("X", X_mods)
@pytest.mark.parametrize("test,key", tests)
def test_tests(X, test, key):
    cfg = OmegaConf.create(
        {
            "_target_": f"src.grinch.{test}.Config",
            "group_key": "obs.label",
        }
    )
    cfg = instantiate(cfg)
    test = cfg.create()
    adata = AnnData(X)
    adata.obs['label'] = label

    test(adata)
    pvals = adata.uns[key]['label-0']['pvals'].to_numpy()
    log2fc = adata.uns[key]['label-0']['log2fc'].to_numpy()
    dd = adata.uns[key]['label-0']
    assert_allclose(dd.pvals, pvals)
    assert_allclose(dd.log2fc, log2fc)

    assert pvals[0] <= 0.1
    assert pvals[1] <= 0.1
    assert pvals[2] >= 0.5
    assert pvals[3] <= 0.1
    assert pvals[4] <= 0.1

    assert log2fc[0] < 2
    assert log2fc[1] < 2
    assert log2fc[2] < 2
    assert log2fc[3] > 2
    assert log2fc[4] > 2

    pvals = adata.uns[key]['label-1']['pvals'].to_numpy()
    log2fc = adata.uns[key]['label-1']['log2fc'].to_numpy()
    dd = adata.uns[key]['label-1']
    assert_allclose(dd.pvals, pvals)
    assert_allclose(dd.log2fc, log2fc)

    assert pvals[0] <= 0.1
    assert pvals[1] <= 0.1
    assert pvals[2] >= 0.5
    assert pvals[3] <= 0.1
    assert pvals[4] <= 0.1

    assert log2fc[0] > 2
    assert log2fc[1] > 2
    assert log2fc[2] < 2
    assert log2fc[3] < 2
    assert log2fc[4] < 2


@pytest.mark.parametrize("test,key", tests)
def test_small_group(test, key):
    X = np.array([
        [1, 2, 3, 100, 150],
        [60, 46, 34, 0, 0],
        [50, 49, 34, 0, 0],
        [60, 46, 38, 0, 0],
    ])
    cfg = OmegaConf.create(
            {
                "_target_": f"src.grinch.{test}.Config",
                "min_points_per_group": 2,
                "group_key": "obs.label",
            }
        )
    cfg = instantiate(cfg)
    test = cfg.create()
    adata = AnnData(X)
    adata.obs['label'] = [0, 1, 1, 1]
    test(adata)
    # empty dataframe
    assert len(adata.uns[key]['label-0']) == 0


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
def test_unimodality(X):
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.UnimodalityTest.Config",
        }
    )
    cfg = instantiate(cfg)
    unimodal = cfg.create()
    adata = AnnData(X)

    unimodal(adata)
    is_sig = adata.uns[UNS.BIMODALTEST].qvals <= 0.05

    assert_allclose([True, True, False, True, True], is_sig)
