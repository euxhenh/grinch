import abc
from typing import Optional

import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from pydantic import Field, validate_arguments
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_array, check_non_negative

from .conf import BaseConfigurable


class BaseNormalizer(BaseConfigurable):
    """An abstract class for normalizers of adata.X.
    These normalizers cannot change the shape of the data, but can only
    modify the values of X.
    """

    class Config(BaseConfigurable.Config):
        inplace: bool = True

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __call__(self, adata: AnnData) -> Optional[AnnData]:
        if not self.cfg.inplace:
            adata = adata.copy()

        X = check_array(
            adata.X,
            accept_sparse='csr',
            ensure_2d=True,
            ensure_min_features=2,
            ensure_min_samples=2,
        )

        adata.X = X
        if adata.is_view:
            adata._init_as_actual(adata.copy())

        original_shape = adata.shape
        self._normalize(adata)
        # Transforms should not change the shape
        assert original_shape == adata.shape

        return adata if not self.cfg.inplace else None

    @abc.abstractmethod
    def _normalize(self, adata: AnnData) -> None:
        raise NotImplementedError


class NormalizeTotal(BaseNormalizer):
    """Normalizes each cell so that total counts are equal."""

    class Config(BaseNormalizer.Config):
        total_counts: Optional[float] = Field(None, gt=0)

    cfg: Config

    def _normalize(self, adata: AnnData) -> None:
        # Make sure values are non-negative for l1 norm to work as expected
        check_non_negative(adata.X, f'{self.__class__.__name__}')
        if self.cfg.total_counts is None:
            # Use median of nonzero total counts as scaling factor
            counts_per_cell = np.ravel(adata.X.sum(axis=1))
            scaling_factor = np.median(counts_per_cell[counts_per_cell > 0])
        else:
            scaling_factor = self.cfg.total_counts

        normalize(adata.X, norm='l1', copy=False)
        to_scale = adata.X.data if sp.issparse(adata.X) else adata.X
        np.multiply(to_scale, scaling_factor, out=to_scale)


class Log1P(BaseNormalizer):
    """Log(X+1) transforms the data. Uses natural logarithm."""

    class Config(BaseNormalizer.Config):
        ...

    cfg: Config

    def _normalize(self, adata: AnnData) -> None:
        check_non_negative(adata.X, f'{self.__class__.__name__}')
        to_log = adata.X.data if sp.issparse(adata.X) else adata.X
        np.log1p(to_log, out=to_log)
