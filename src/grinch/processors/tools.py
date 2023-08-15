import logging
from numbers import Number
from typing import TYPE_CHECKING, Callable, Dict, Literal

import numpy as np
from anndata import AnnData
from pydantic import Field
from pyensembl import EnsemblRelease

from ..aliases import UNS, VAR
from ..cond_filter import Filter, StackedFilter
from ..custom_types import NP1D_bool
from .base_processor import BaseProcessor, ReadKey, WriteKey

logger = logging.getLogger(__name__)


class GeneIdToName(BaseProcessor):
    """Used to convert gene naming convention.
    """
    class Config(BaseProcessor.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'GeneIdToName']

        read_key: ReadKey = "var_names"
        save_key: WriteKey = f"var.{VAR.FEATURE_NAME}"
        stats_key: WriteKey = f"uns.{UNS.N_GENE_ID_TO_NAME_FAILED}"
        ensembl_release: int = 77

    cfg: Config

    def __init__(self, cfg: Config, /) -> None:
        super().__init__(cfg)

        self.data = EnsemblRelease(self.cfg.ensembl_release)

    def _process(self, adata: AnnData) -> None:
        gene_names = []
        not_found = 0

        gene_ids = self.read(adata, self.cfg.read_key, to_numpy=True)
        for gene_id in gene_ids:
            try:
                gene_names.append(self.data.gene_by_id(gene_id).gene_name)
            except ValueError:
                gene_names.append(gene_id)
                not_found += 1

        logger.info(f"Could not convert {not_found} gene IDs.")
        self.store_item(self.cfg.save_key, np.asarray(gene_names))
        self.store_item(self.cfg.stats_key, not_found)


class StoreAsMask(BaseProcessor):
    # Simple class that stores a mask of Filters

    class Config(BaseProcessor.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'StoreAsMask']

        filter_by: Dict[str, Filter]
        save_key: WriteKey

    cfg: Config

    def _process(self, adata: AnnData) -> None:
        sfc = StackedFilter(*self.cfg.filter_by.values())
        mask: NP1D_bool = sfc(adata, as_mask=True)
        self.store_item(self.cfg.save_key, mask)


class ReplaceNaN(BaseProcessor):

    class Config(BaseProcessor.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'ReplaceNaN']

        read_key: ReadKey
        save_key: WriteKey | None
        replace_value: Number | Literal['median'] = Field(0.0)

    cfg: Config

    def _process(self, adata: AnnData) -> None:
        x = np.asarray(self.read(adata, self.cfg.read_key), dtype=float)
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

        if TYPE_CHECKING:
            create: Callable[..., 'FilterNaN']

        read_key: ReadKey

    cfg: Config

    def _process(self, adata: AnnData) -> None:
        x = np.asarray(self.read(adata, self.cfg.read_key), dtype=float)
        mask = np.isnan(x)
        if mask.sum() > 0:
            logger.info(f"Removing {mask.sum()} rows with NaNs.")
            adata._inplace_subset_obs(~mask)


class ApplyOp(BaseProcessor):
    """Applies a numpy operation to a column and stores
    it somewhere else.

    Parameters
    __________
    read_key: str
    save_key: str
        If save_key is None, will store at read_key.
    op: str
        Numpy operation to apply. Can also be an attribute,
        but in that case as_attr must be set to True.
    as_attr: bool
        If False, will call np.op(x), otherwise call x.op().
    """

    class Config(BaseProcessor.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'ApplyOp']

        read_key: ReadKey
        save_key: WriteKey | None = None
        op: str
        as_attr: bool = False

    cfg: Config

    def _process(self, adata: AnnData) -> None:
        x = self.read(adata, self.cfg.read_key)
        x = (
            getattr(np, self.cfg.op)(x) if not self.cfg.as_attr
            else getattr(x, self.cfg.op)()
        )
        save_key = self.cfg.save_key or self.cfg.read_key
        self.store_item(save_key, x)
