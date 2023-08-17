import numpy as np
import pytest
from anndata import AnnData
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.grinch import BaseConfigurable

adata = AnnData(np.random.random((100, 80)))


def test_save_key_zero_len():
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.PCA.Config",
            "write_key": "uns.",
        }
    )
    cfg = instantiate(cfg)
    pca = cfg.create()

    with pytest.raises(ValueError, match=r"zero-length"):
        pca(adata)


def test_column_error():
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.PCA.Config",
            "write_key": "foo.pca",
        }
    )
    cfg = instantiate(cfg)
    pca = cfg.create()

    with pytest.raises(ValueError, match='class not in allowed list'):
        pca(adata)


def test_conf_errors():
    with pytest.raises(TypeError, match='immediate parent class'):
        class T:
            ...

        class BadParent(T, BaseConfigurable):
            ...

    with pytest.raises(NotImplementedError, match='does not implement a nested Config'):
        class NoConfig(BaseConfigurable):
            ...

    with pytest.raises(TypeError, match='should be of explicit type'):
        class NoTyping(BaseConfigurable):
            class Config(BaseConfigurable.Config):
                ...

            def __init__(self, cfg, /):
                ...

    with pytest.raises(ValueError, match='should be positional only'):
        class NoPositional(BaseConfigurable):
            class Config(BaseConfigurable.Config):
                ...

            def __init__(self, cfg: Config):
                ...
