import abc
from typing import Optional

import numpy as np
from anndata import AnnData
from pydantic import validate_arguments
from sklearn.utils.validation import (
    check_array,
    check_non_negative,
)

from .aliases import OBS, VAR
from .conf import BaseConfigurable
from .utils import any_not_None, true_inside


class BaseFilter(BaseConfigurable):
    """A base class for filters."""

    class Config(BaseConfigurable.Config):
        inplace: bool = True

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __call__(self, adata: AnnData) -> Optional[AnnData]:
        check_array(
            adata.X,
            accept_sparse=True,
            ensure_2d=True,
            ensure_min_features=2,
            ensure_min_samples=2,
        )
        # Make sure there are no negative counts
        check_non_negative(adata.X, f'{self.__class__.__name__}')

        if not self.cfg.inplace:
            adata = adata.copy()

        self._filter(adata)

        return adata if not self.cfg.inplace else None

    @abc.abstractmethod
    def _filter(self, adata: AnnData) -> None:
        raise NotImplementedError


class FilterCells(BaseFilter):
    """Filters cells based on counts and number of expressed genes."""

    class Config(BaseFilter.Config):
        min_counts: Optional[float] = None
        max_counts: Optional[float] = None
        min_genes: Optional[int] = None
        max_genes: Optional[int] = None

    cfg: Config

    def _filter(self, adata: AnnData) -> None:
        # Keep all cells by default
        to_keep = np.ones(adata.shape[0], dtype=bool)

        counts_per_cell = np.ravel(adata.X.sum(axis=1))
        if any_not_None(self.cfg.min_counts, self.cfg.max_counts):
            to_keep &= true_inside(
                counts_per_cell,
                self.cfg.min_counts,
                self.cfg.max_counts,
            )

        # Values are ensured to be non-negative
        genes_per_cell = np.ravel((adata.X > 0).sum(axis=1))
        if any_not_None(self.cfg.min_genes, self.cfg.max_genes):
            to_keep &= true_inside(
                genes_per_cell,
                self.cfg.min_genes,
                self.cfg.max_genes,
            )

        if to_keep.sum() <= 1:
            raise ValueError(
                "Filtering options are too stringent. "
                "Less than 2 cells remained."
            )

        # Set these after the exception above
        adata.obs[OBS.N_COUNTS] = counts_per_cell
        adata.obs[OBS.N_GENES] = genes_per_cell

        adata._inplace_subset_obs(to_keep)

        self.log(
            f"Keeping {adata.shape[0]}/{len(to_keep)} cells.",
            shape=adata.shape,
        )


class FilterGenes(BaseFilter):
    """Filters cells based on counts and number of expressed genes."""

    class Config(BaseFilter.Config):
        min_counts: Optional[float] = None
        max_counts: Optional[float] = None
        min_cells: Optional[int] = None
        max_cells: Optional[int] = None

    cfg: Config

    def _filter(self, adata: AnnData) -> None:
        # Keep all genes by default
        to_keep = np.ones(adata.shape[1], dtype=bool)

        counts_per_gene = np.ravel(adata.X.sum(axis=0))
        if any_not_None(self.cfg.min_counts, self.cfg.max_counts):
            to_keep &= true_inside(
                counts_per_gene,
                self.cfg.min_counts,
                self.cfg.max_counts,
            )

        # Values are ensured to be non-negative
        cells_per_gene = np.ravel((adata.X > 0).sum(axis=0))
        if any_not_None(self.cfg.min_cells, self.cfg.max_cells):
            to_keep &= true_inside(
                cells_per_gene,
                self.cfg.min_cells,
                self.cfg.max_cells,
            )

        if to_keep.sum() < 1:
            raise ValueError(
                "Filtering options are too stringent. "
                "Less than 1 gene remained."
            )

        # Set these after the exception above
        adata.var[VAR.N_COUNTS] = counts_per_gene
        adata.var[VAR.N_CELLS] = cells_per_gene

        adata._inplace_subset_var(to_keep)

        self.log(
            f"Keeping {adata.shape[1]}/{len(to_keep)} genes.",
            shape=adata.shape,
        )
