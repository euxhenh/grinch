from numbers import Number
from typing import Dict, Optional

import numpy as np
from anndata import AnnData
from pydantic import Field

from ..custom_types import NP1D_bool
from ..filter_condition import FilterCondition, StackedFilterCondition
from .base_processor import BaseProcessor


class StoreAsMask(BaseProcessor):
    # Simple class that stores a mask of FilterConditions

    class Config(BaseProcessor.Config):
        filter_by: Dict[str, FilterCondition]
        save_key: str

    cfg: Config

    def _process(self, adata: AnnData) -> None:
        sfc = StackedFilterCondition(*self.cfg.filter_by.values())
        mask: NP1D_bool = sfc(adata, as_mask=True)
        self.set_repr(adata, self.cfg.save_key, mask)


class ReplaceNAN(BaseProcessor):

    class Config(BaseProcessor.Config):
        read_key: str
        save_key: Optional[str]
        replace_value: Number | str = Field(0.0, regex='median')

    cfg: Config

    def _process(self, adata: AnnData) -> None:
        x = self.get_repr(adata, self.cfg.read_key)
        mask = np.isnan(x)

        if self.cfg.save_key is not None:
            x = x.copy()

        val = np.median(x[~mask]) if self.cfg.replace_value == 'median' else self.cfg.replace_value
        x[mask] = val

        if self.cfg.save_key is not None:
            self.set_repr(adata, self.cfg.save_key, x)
        else:
            self.set_repr(adata, self.cfg.read_key, x)
