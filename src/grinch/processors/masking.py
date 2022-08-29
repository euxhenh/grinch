from typing import Dict

from anndata import AnnData

from ..filter_condition import FilterCondition, StackedFilterCondition
from ..custom_types import NP1D_bool
from .base_processor import BaseProcessor


class StoreAsMask(BaseProcessor):
    # Simple class that stores a mask of filtered conditions

    class Config(BaseProcessor.Config):
        filter_by: Dict[str, FilterCondition]
        save_key: str

    cfg: Config

    def _process(self, adata: AnnData) -> None:
        sfc = StackedFilterCondition(*self.cfg.filter_by.values())
        mask: NP1D_bool = sfc(adata, as_mask=True)
        self.set_repr(adata, self.cfg.save_key, mask)
