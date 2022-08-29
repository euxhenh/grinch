from typing import Dict

from anndata import AnnData
from pydantic import validator

from ..filter_condition import FilterCondition, StackedFilterCondition
from ..utils.validation import validate_axis
from .base_processor import BaseProcessor


class InplaceIndexer(BaseProcessor):
    """Inexes adata over obs or var axis using a mask stored in obs/var or
    an uns list of indices.
    """

    class Config(BaseProcessor.Config):
        filter_by: Dict[str, FilterCondition]
        # Can be 0, 1 or 'obs', 'var'
        axis: int | str = 0

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
        mask = sfc(adata, as_mask=True)

        if self.cfg.axis == 0:
            adata._inplace_subset_obs(mask)
        else:
            adata._inplace_subset_var(mask)
