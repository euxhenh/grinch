import numpy as np
import pytest
from anndata import AnnData
from hydra.utils import instantiate
from omegaconf import OmegaConf

from grinch import OBS, OBSM, DataSplitter

from ._utils import to_view, assert_allclose

X = np.array([
    [2, 2, 0, 0, 0],
    [3, 7, 0, 0, 0],
    [0, 1, 3, 3, 3],
    [0, 3, 9, 9, 9],
    [0, 0, 1, 0, 0],
    [0, 1, 5, 3, 1],
], dtype=np.float32)

X_mods = [X, to_view(X)]


@pytest.mark.parametrize("X", X_mods)
def test_pipeline_end_to_end_single_dataset(X):
    processor_configs = []
    processor_configs.append(OmegaConf.create({
        "_target_": "src.grinch.FilterCells.Config",
        "min_counts": 2,
    }))
    processor_configs.append(OmegaConf.create({
        "_target_": "src.grinch.NormalizeTotal.Config",
        "total_counts": 10,
    }))
    processor_configs.append(OmegaConf.create({
        "_target_": "src.grinch.PCA.Config",
        "n_components": 3,
    }))
    processor_configs.append(OmegaConf.create({
        "_target_": "src.grinch.Splitter.Config",
        "val_fraction": 0.5,
    }))
    processor_configs.append(OmegaConf.create({
        "_target_": "src.grinch.KMeans.Config",
        "n_clusters": 2,
    }))
    processor_configs.append(OmegaConf.create({
        "_target_": "src.grinch.LogisticRegression.Config",
        "y_key": f"obs.{OBS.KMEANS}",
    }))

    cfg = OmegaConf.create({
        "_target_": "src.grinch.GRPipeline.Config",
        "processors": processor_configs,
        "seed": 42,
    })
    cfg = instantiate(cfg)
    pipeline = cfg.initialize()
    adata = AnnData(X)

    ds: DataSplitter = pipeline(adata)

    train = ds.TRAIN_SPLIT
    val = ds.VAL_SPLIT

    assert train.shape == (2, 5)
    assert val.shape == (3, 5)
    assert train.obsm[OBSM.X_PCA].shape == (2, 3)
    assert val.obsm[OBSM.X_PCA].shape == (3, 3)
    assert (train.X.sum(axis=1) == 10).all()
    assert (val.X.sum(axis=1) == 10).all()

    assert_allclose(train.obs[OBS.KMEANS], [0, 1])
    assert_allclose(train.obs[OBS.LOG_REG], [0, 1])
    assert_allclose(val.obs[OBS.LOG_REG], [0, 1, 1])
