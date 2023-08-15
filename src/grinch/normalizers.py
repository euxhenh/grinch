import abc
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from pydantic import Field, field_validator, validate_call
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_array, check_non_negative

from .conf import BaseConfigurable
from .external.combat import combat  # type: ignore
from .processors import BaseProcessor
from .utils.stats import mean_var


class BaseNormalizer(BaseConfigurable):
    """An abstract class for normalizers of adata.X.
    These normalizers cannot change the shape of the data, but can only
    modify the values of X.

    Parameters
    __________
    save_input : bool
        If True, will save the raw input into adata.layers.

    input_layer_name : str
        Ignored if save_input is False. Will store the raw input into
        adata.layers[input_layer_name]. If None, the name will be
        automatically set to 'pre_{self.__class__.__name__}'.
    """

    class Config(BaseConfigurable.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'BaseNormalizer']

        inplace: bool = True
        save_input: bool = True
        input_layer_name: str = Field(None)

        @field_validator('input_layer_name', mode='before')
        def resolve_input_layer_name(cls, value):
            return f'pre_{cls._init_cls.__name__}'.lower() if value is None else value

    cfg: Config

    @validate_call(config=dict(arbitrary_types_allowed=True))
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
        if self.cfg.save_input:
            adata.layers[self.cfg.input_layer_name] = adata.X.copy()

        self._normalize(adata)
        # Transforms should not change the shape
        assert original_shape == adata.shape

        return adata if not self.cfg.inplace else None

    @abc.abstractmethod
    def _normalize(self, adata: AnnData) -> None:
        raise NotImplementedError


class Combat(BaseNormalizer):
    """Performs batch correction using Combat
    Source:
    https://academic.oup.com/biostatistics/article/8/1/118/252073?login=false

    Uses code from https://github.com/brentp/combat.py/tree/master
    """
    class Config(BaseNormalizer.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'Combat']

        batch_key: str

    cfg: Config

    def _normalize(self, adata: AnnData) -> None:
        batch: pd.Series = BaseProcessor.read(adata, self.cfg.batch_key)
        if not isinstance(batch, pd.Series):
            raise ValueError("Batch should be a pandas series")

        if sp.issparse(adata.X):
            if adata.is_view:
                adata._init_as_actual(adata.copy())
            adata.X = adata.X.toarray()

        data = pd.DataFrame(adata.X)
        data.index = batch.index
        corrected_data = combat(data.T, batch).T
        adata.X = corrected_data.to_numpy()


class NormalizeTotal(BaseNormalizer):
    """Normalizes each cell so that total counts are equal."""

    class Config(BaseNormalizer.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'NormalizeTotal']

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

        if TYPE_CHECKING:
            create: Callable[..., 'Log1P']

    cfg: Config

    def _normalize(self, adata: AnnData) -> None:
        check_non_negative(adata.X, f'{self.__class__.__name__}')
        to_log = adata.X.data if sp.issparse(adata.X) else adata.X
        np.log1p(to_log, out=to_log)


class Scale(BaseNormalizer):
    """Scales to standard deviation and optionally zero mean.
    """

    class Config(BaseNormalizer.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'Scale']

        max_value: float | None = Field(None, gt=0)
        with_mean: bool = True
        with_std: bool = True

    cfg: Config

    def _normalize(self, adata: AnnData) -> None:
        X = adata.X
        # Run before densifying: faster computation if sparse
        mean, var = mean_var(X, axis=0)
        var[var == 0] = 1
        std = var ** (1/2)

        if self.cfg.with_mean and sp.issparse(adata.X):
            X = X.toarray()
        if self.cfg.with_mean:
            X -= mean
        X /= std

        if self.cfg.max_value is not None:
            to_clip = X.data if sp.issparse(X) else X
            np.clip(to_clip, None, self.cfg.max_value, out=to_clip)

        adata.X = X
