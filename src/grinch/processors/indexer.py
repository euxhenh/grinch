import abc
import gc
from typing import List

from anndata import AnnData
from pydantic import Field, validator

from ..cond_filter import Filter, StackedFilter
from ..custom_types import NP1D_bool
from ..utils.validation import validate_axis
from .base_processor import BaseProcessor


class BaseIndexer(BaseProcessor, abc.ABC):
    """A base class for indexing operations."""

    class Config(BaseProcessor.Config):
        filter_by: List[Filter]
        # Can be 0, 1 or 'obs', 'var'
        axis: int | str = Field(0, ge=0, le=1, pattern='^(obs|var)$')

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
        sfc = StackedFilter(*self.cfg.filter_by)
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

        gc.collect()


class IndexProcessor(BaseIndexer):
    """Runs a processor on a subset of adata. Makes sure that no writes are
    applied to the view of adata, but to the full adata.
    """

    class Config(BaseIndexer.Config):
        processor: BaseProcessor.Config

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

        self.cfg.processor.inplace = True
        self.processor = self.cfg.processor.initialize()

    def _process_mask(self, adata: AnnData, mask: NP1D_bool) -> None:
        # dont let the processor do the adata modifications, since we are
        # passing a view
        key = ['obs_indices', 'var_indices'][int(self.cfg.axis)]
        kwargs = {key: mask}
        self.processor(adata, **kwargs)
