import abc
from typing import Optional

import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from pydantic import validate_arguments
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_array, check_non_negative

from .conf import BaseConfigurable


class BaseTransform(BaseConfigurable):

    class Config(BaseConfigurable.Config):
        inplace: bool = True

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __call__(self, adata: AnnData) -> Optional[AnnData]:
        adata.X = check_array(
            adata.X,
            accept_sparse='csr',
            ensure_2d=True,
            ensure_min_features=2,
            ensure_min_samples=2,
        )

        if not self.cfg.inplace:
            adata = adata.copy()
        elif adata.is_view:
            adata._init_as_actual(adata.copy())

        original_shape = adata.shape
        self._transform(adata)
        # Transforms should not change the shape
        assert original_shape == adata.shape

        return adata if not self.cfg.inplace else None

    @abc.abstractmethod
    def _transform(self, adata: AnnData) -> None:
        raise NotImplementedError


class NormalizeTotal(BaseTransform):
    """Normalizes each cell so that total counts are equal."""

    class Config(BaseTransform.Config):
        total_counts: Optional[float] = None

    cfg: Config

    def _transform(self, adata: AnnData) -> None:
        check_non_negative(adata.X, f'{self.__class__.__name__}')
        normalize(adata.X, norm='l1', copy=False)
        adata.X *= self.cfg.total_counts


class Log1P(BaseTransform):
    """Normalizes each cell so that total counts are equal."""

    class Config(BaseTransform.Config):
        ...

    cfg: Config

    def _transform(self, adata: AnnData) -> None:
        check_non_negative(adata.X, f'{self.__class__.__name__}')

        if sp.issparse(adata.X):
            to_log = adata.X.data
        else:
            to_log = adata.X

        np.log1p(to_log, out=to_log)
