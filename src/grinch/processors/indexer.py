import abc
from typing import Dict

from anndata import AnnData
from pydantic import Field, validator

from ..custom_types import NP1D_bool
from ..filter_condition import FilterCondition, StackedFilterCondition
from ..utils.validation import validate_axis
from .base_processor import BaseProcessor


class BaseIndexer(BaseProcessor, abc.ABC):
    """A base class for indexing operations."""

    class Config(BaseProcessor.Config):
        filter_by: Dict[str, FilterCondition]
        # Can be 0, 1 or 'obs', 'var'
        axis: int | str = Field(0, ge=0, le=1, regex='^(obs|var)$')

        @validator('axis')
        def ensure_correct_axis(cls, axis):
            return validate_axis(axis)

        @validator('filter_by')
        def at_least_one_filter(cls, val):
            if len(val) < 1:
                raise ValueError("At least one filter should be provided.")
            return val

    cfg: Config

    def _process(self, adata: AnnData) -> None:
        sfc = StackedFilterCondition(*self.cfg.filter_by.values())
        mask: NP1D_bool = sfc(adata, as_mask=True)
        return self._process_mask(adata, mask)

    @abc.abstractmethod
    def _process_mask(self, adata: AnnData, mask: NP1D_bool) -> None:
        raise NotImplementedError


class InplaceIndexer(BaseIndexer):
    """Inplace indexes adata over obs or var axis using a mask stored in
    obs/var or an uns list of indices.
    """

    class Config(BaseIndexer.Config):
        ...

    cfg: Config

    def _process_mask(self, adata: AnnData, mask: NP1D_bool) -> None:
        if self.cfg.axis == 0:
            adata._inplace_subset_obs(mask)
        else:
            adata._inplace_subset_var(mask)
