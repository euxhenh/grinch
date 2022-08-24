import logging
from typing import Any, Dict, List

import gseapy as gp
import numpy as np
import pandas as pd
from anndata import AnnData
from pydantic import validator
from sklearn.utils.validation import column_or_1d

from ..aliases import UNS
from ..custom_types import NP1D_int, NP1D_str
from ..de_test_summary import DETestSummary, FilterCondition
from ..utils.validation import pop_args
from .base_processor import BaseProcessor

logger = logging.getLogger(__name__)


class GSEA(BaseProcessor):
    """Performs gene set enrichment analysis. Will parse a dict of
    dataframes or a single dataframe and select the top k genes to perform
    GSEA based on user defined criteria.

    Parameters
    __________
    read_key: str
        Must point to a dict of dataframes in anndata, or to a single
        dataframe. Each of these dataframes will be converted to a
        DETestSummary.
    save_key: str
        Will point to a dict of dataframes if read_key also points to a
        dict, or to a single dataframe otherwise.
    gene_sets: str or list of str
        Names of gene sets to use for GSEA.
    filter_by: dict of FilterCondition
        These will be used to filter genes for GSEA. Dict keys are ignored.
    """

    class Config(BaseProcessor.Config):
        read_key: str = f"uns.{UNS.TTEST}"
        save_key: str = f"uns.{UNS.GSEA}"
        gene_sets: List[str] | str = "HuBMAP_ASCTplusB_augmented_2022"
        # Dict of keys to use for filtering DE genes
        filter_by: Dict[str, FilterCondition]
        gene_names_key: str = "var_names"
        kwargs: Dict[str, Any] = {}

        @validator('kwargs')
        def remove_explicit_args(cls, val):
            return pop_args(['gene_list', 'gene_sets', 'no_plot'], val)

    cfg: Config

    def _gsea(self, gene_list: List[str] | NP1D_str) -> pd.DataFrame:
        """Wrapper around gp.enrichr."""
        if hasattr(gene_list, 'tolist') and not isinstance(gene_list, list):  # 2nd if for mypy
            gene_list = gene_list.tolist()

        return gp.enrichr(
            gene_list=gene_list,
            gene_sets=self.cfg.gene_sets,
            no_plot=True,
            **self.cfg.kwargs,
        ).results

    def _process_test(
        self,
        test: pd.DataFrame | DETestSummary,
        gene_list_all: NP1D_str
    ) -> pd.DataFrame:
        """Process a single DataFrame or DETestSummary object."""
        if isinstance(test, pd.DataFrame):
            test = DETestSummary.from_df(test)
        elif not isinstance(test, DETestSummary):
            raise TypeError(
                f"Expected a DataFrame or DETestSummary object but found {type(test)}.")

        if len(gene_list_all) != len(test):
            raise ValueError(
                "Expected gene_list to be of same length as DETestSummary, but "
                f"found gene_list of length {len(gene_list_all)} and DETestSummary "
                f"of length {len(test)}."
            )

        # Apply all filters
        gene_idx: NP1D_int = test.where(*self.cfg.filter_by.values(), as_mask=False)
        gene_list = gene_list_all[gene_idx]
        return self._gsea(gene_list)

    def _process(self, adata: AnnData) -> None:
        tests = self.get_repr(adata, self.cfg.read_key)
        gene_list_all = self.get_repr(adata, self.cfg.gene_names_key)
        gene_list_all = column_or_1d(gene_list_all).astype(str)
        gene_list_all = np.char.upper(gene_list_all)

        if len(gene_list_all) != adata.shape[1]:
            logger.warn(
                "Gene list has a different dimension than AnnData's column dimension. "
                "Please make sure 'read_key' is what you intended to use."
            )

        if isinstance(tests, dict):  # Dict of tests
            for label, test in tests.items():
                gsea_test_summary = self._process_test(test, gene_list_all)
                save_key = f'{self.cfg.save_key}.{label}'
                self.set_repr(adata, save_key, gsea_test_summary)
        else:  # Single test
            gsea_test_summary = self._process_test(tests, gene_list_all)
            self.set_repr(adata, self.cfg.save_key, gsea_test_summary)
