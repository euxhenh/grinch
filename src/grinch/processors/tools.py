import logging
from numbers import Number
from typing import Dict, Optional

import numpy as np
from anndata import AnnData
from pydantic import Field
from pyensembl import EnsemblRelease

from ..aliases import UNS, VAR
from ..custom_types import NP1D_bool
from ..filter_condition import FilterCondition, StackedFilterCondition
from .base_processor import BaseProcessor

logger = logging.getLogger(__name__)


class GeneIdToName(BaseProcessor):
    """Used to convert gene naming convention.
    """
    class Config(BaseProcessor.Config):
        read_key: str = "var_names"
        save_key: str = f"var.{VAR.FEATURE_NAME}"
        stats_key: str = f"uns.{UNS.N_GENE_ID_TO_NAME_FAILED}"
        ensembl_release: int = 77

    cfg: Config

    def __init__(self, cfg: Config, /) -> None:
        super().__init__(cfg)

        self.data = EnsemblRelease(self.cfg.ensembl_release)

    def _process(self, adata: AnnData) -> None:
        gene_names = []
        not_found = 0

        gene_ids = self.get_repr(adata, self.cfg.read_key, to_numpy=True)
        for gene_id in gene_ids:
            try:
                gene_names.append(self.data.gene_by_id(gene_id).gene_name)
            except ValueError:
                gene_names.append(gene_id)
                not_found += 1

        self.set_repr(adata, self.cfg.save_key, np.asarray(gene_names))
        self.store_item(self.cfg.stats_key, not_found)


class StoreAsMask(BaseProcessor):
    # Simple class that stores a mask of FilterConditions

    class Config(BaseProcessor.Config):
        filter_by: Dict[str, FilterCondition]
        save_key: str

    cfg: Config

    def _process(self, adata: AnnData) -> None:
        sfc = StackedFilterCondition(*self.cfg.filter_by.values())
        mask: NP1D_bool = sfc(adata, as_mask=True)
        self.store_item(self.cfg.save_key, mask)


class ReplaceNaN(BaseProcessor):

    class Config(BaseProcessor.Config):
        read_key: str
        save_key: Optional[str]
        replace_value: Number | str = Field(0.0, regex='^median$')

    cfg: Config

    def _process(self, adata: AnnData) -> None:
        x = np.asarray(self.get_repr(adata, self.cfg.read_key), dtype=float)
        mask = np.isnan(x)
        logger.info(f"Replacing {mask.sum()} nan values.")

        if self.cfg.save_key is not None:
            x = x.copy()

        val = (
            np.median(x[~mask]) if self.cfg.replace_value == 'median'
            else self.cfg.replace_value
        )
        x[mask] = val

        if self.cfg.save_key is not None:
            self.store_item(self.cfg.save_key, x)
        else:
            self.store_item(self.cfg.read_key, x)


class FilterNaN(BaseProcessor):

    class Config(BaseProcessor.Config):
        read_key: str

    cfg: Config

    def _process(self, adata: AnnData) -> None:
        x = np.asarray(self.get_repr(adata, self.cfg.read_key), dtype=float)
        mask = np.isnan(x)
        if mask.sum() > 0:
            logger.info(f"Removing {mask.sum()} rows with NaNs.")
            adata._inplace_subset_obs(~mask)
