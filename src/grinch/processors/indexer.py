import abc
import gc
import logging
from typing import TYPE_CHECKING, Callable, List, Literal

from anndata import AnnData
from pydantic import Field, field_validator

from ..cond_filter import Filter, StackedFilter
from ..custom_types import Binary, NP1D_bool
from ..utils.validation import validate_axis
from .base_processor import BaseProcessor

logger = logging.getLogger(__name__)


class BaseIndexer(BaseProcessor, abc.ABC):
    """A base class for indexing operations."""

    class Config(BaseProcessor.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'BaseIndexer']

        filter_by: List[Filter] = Field(min_length=1)
        axis: Binary | Literal['obs', 'var'] = 0

        @field_validator('axis')
        def ensure_correct_axis(cls, axis):
            return validate_axis(axis)

        @field_validator('filter_by', mode='before')
        def ensure_filter_list(cls, val):
            return [val] if isinstance(val, Filter) else val

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

        if TYPE_CHECKING:
            create: Callable[..., 'InplaceIndexer']

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

        if TYPE_CHECKING:
            create: Callable[..., 'IndexProcessor']

        processor_cfg: BaseProcessor.Config

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

        self.processor = self.cfg.processor_cfg.create()

    def _process_mask(self, adata: AnnData, mask: NP1D_bool) -> None:
        # TODO improve naming and add indexing stats
        # dont let the processor do the adata modifications, since we are
        # passing a view
        key = ['obs_indices', 'var_indices'][int(self.cfg.axis)]
        kwargs = {key: mask}
        logger.info(f"Running '{self.processor.__class__.__name__}'.")
        self.processor(adata, **kwargs)
