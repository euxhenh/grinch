import abc
import logging
from typing import TYPE_CHECKING, Callable

import numpy as np
from anndata import AnnData
from pydantic import NonNegativeFloat, NonNegativeInt, validate_call
from sklearn.utils.validation import check_non_negative

from .aliases import OBS, VAR
from .conf import BaseConfigurable
from .utils import any_not_None, true_inside
from .utils.decorators import plt_interactive
from .utils.plotting import plot1d
from .utils.stats import _var

logger = logging.getLogger(__name__)


class BaseFilter(BaseConfigurable):
    r"""A base class for sample and feature filters.

    Args:
        inplace (bool): If ``True`` will copy and return the processed adata.
    """

    class Config(BaseConfigurable.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'BaseFilter']

        inplace: bool = True

    cfg: Config

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __call__(self, adata: AnnData) -> AnnData | None:
        if not self.cfg.inplace:
            adata = adata.copy()

        self._filter(adata)
        return adata if not self.cfg.inplace else None

    @abc.abstractmethod
    def _filter(self, adata: AnnData) -> None:
        raise NotImplementedError


class FilterCells(BaseFilter):
    r"""Filters cells based on counts and number of expressed genes.
    All filters will be applied to the original data matrix, and not
    consecutively.

    Args:
        min_counts (float >= 0): Minimum number of counts for each sample.
        max_counts (float >= 0): Maximum number of counts for each sample.
        min_genes (int >= 0): Minimum number of genes for each sample.
        min_genes (int >= 0): Maximum number of genes for each sample.
    """

    class Config(BaseFilter.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'FilterCells']

        min_counts: NonNegativeFloat | None = None
        max_counts: NonNegativeFloat | None = None
        min_genes: NonNegativeInt | None = None
        max_genes: NonNegativeInt | None = None

    cfg: Config

    def _filter(self, adata: AnnData) -> None:
        # Make sure there are no negative counts
        check_non_negative(adata.X, f'{self.__class__.__name__}')
        # Keep all cells by default
        to_keep = np.full(adata.shape[0], True)

        counts_per_cell = np.ravel(adata.X.sum(axis=1))

        if self.cfg.interactive:
            with plt_interactive(self.logs_path / 'counts_per_cell.png'):
                plot1d(counts_per_cell, 'nbinom', title='Counts per Cell')
                self.cfg.min_counts = eval(input("Enter min_counts="))
                self.cfg.max_counts = eval(input("Enter max_counts="))

        if any_not_None(self.cfg.min_counts, self.cfg.max_counts):
            to_keep &= true_inside(
                counts_per_cell,
                self.cfg.min_counts,
                self.cfg.max_counts,
            )

        # Values are ensured to be non-negative
        genes_per_cell = np.ravel((adata.X > 0).sum(axis=1))

        if self.cfg.interactive:
            with plt_interactive(self.logs_path / 'genes_per_cell.png'):
                plot1d(genes_per_cell, 'nbinom', title='Genes per Cell')
                self.cfg.min_genes = eval(input("Enter min_genes="))
                self.cfg.max_genes = eval(input("Enter max_genes="))

        if any_not_None(self.cfg.min_genes, self.cfg.max_genes):
            to_keep &= true_inside(
                genes_per_cell,
                self.cfg.min_genes,
                self.cfg.max_genes,
            )

        if to_keep.sum() <= 1:
            raise ValueError("Filtering options are too stringent. "
                             "Less than 2 cells remained.")

        # Set these after the exception above
        adata.obs[OBS.N_COUNTS] = counts_per_cell.astype(np.float32)
        adata.obs[OBS.N_GENES] = genes_per_cell.astype(np.float32)

        adata._inplace_subset_obs(to_keep)

        logger.info(f"Keeping {adata.shape[0]}/{len(to_keep)} cells.")


class FilterGenes(BaseFilter):
    """Filters cells based on counts and number of expressed genes."""

    class Config(BaseFilter.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'FilterGenes']

        min_counts: NonNegativeFloat | None = None
        max_counts: NonNegativeFloat | None = None
        min_cells: NonNegativeInt | None = None
        max_cells: NonNegativeInt | None = None

    cfg: Config

    def _filter(self, adata: AnnData) -> None:
        check_non_negative(adata.X, f'{self.__class__.__name__}')
        # Keep all genes by default
        to_keep = np.full(adata.shape[1], True)

        counts_per_gene = np.ravel(adata.X.sum(axis=0))

        if self.cfg.interactive:
            with plt_interactive(self.logs_path / 'counts_per_gene.png'):
                plot1d(counts_per_gene, 'halfnorm', title='Counts per Gene')
                self.cfg.min_counts = eval(input("Enter min_counts="))
                self.cfg.max_counts = eval(input("Enter max_counts="))

        if any_not_None(self.cfg.min_counts, self.cfg.max_counts):
            to_keep &= true_inside(
                counts_per_gene,
                self.cfg.min_counts,
                self.cfg.max_counts,
            )

        cells_per_gene = np.ravel((adata.X > 0).sum(axis=0))

        if self.cfg.interactive:
            with plt_interactive(self.logs_path / 'cells_per_gene.png'):
                plot1d(cells_per_gene, 'nbinom', title='Cells per Gene')
                self.cfg.min_cells = eval(input("Enter min_cells="))
                self.cfg.max_cells = eval(input("Enter max_cells="))

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

        adata.var[VAR.N_COUNTS] = counts_per_gene.astype(np.float32)
        adata.var[VAR.N_CELLS] = cells_per_gene.astype(np.float32)

        adata._inplace_subset_var(to_keep)

        logger.info(f"Keeping {adata.shape[1]}/{len(to_keep)} genes.")


class VarianceFilter(BaseFilter):
    """Filter features by variance."""

    class Config(BaseFilter.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'VarianceFilter']

        min_var: NonNegativeFloat | None = None
        max_var: NonNegativeFloat | None = None
        ddof: NonNegativeInt = 1

    cfg: Config

    def _filter(self, adata: AnnData) -> None:
        to_keep = np.full(adata.shape[1], True)
        gene_var = _var(adata.X, axis=0, ddof=self.cfg.ddof)

        if self.cfg.interactive:
            with plt_interactive(self.logs_path / 'gene_var.png'):
                plot1d(gene_var, 'halfnorm', title='Gene Variance')
                self.cfg.min_var = eval(input("Enter min_var="))
                self.cfg.max_var = eval(input("Enter max_var="))

        if any_not_None(self.cfg.min_var, self.cfg.max_var):
            to_keep &= true_inside(gene_var, self.cfg.min_var, self.cfg.max_var)

        if to_keep.sum() < 1:
            raise ValueError(
                "Filtering options are too stringent. "
                "Less than 1 gene remained."
            )

        adata.var[VAR.VARIANCE] = gene_var.astype(np.float32)
        adata._inplace_subset_var(to_keep)

        logger.info(f"Keeping {adata.shape[1]}/{len(to_keep)} genes.")
