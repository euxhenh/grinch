import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List, Optional

import gseapy as gp
import numpy as np
import pandas as pd
from anndata import AnnData
from pydantic import Field, validator
from sklearn.utils.validation import column_or_1d

from ..aliases import UNS
from ..custom_types import NP1D_int, NP1D_str
from ..de_test_summary import DETestSummary, FilterCondition
from ..utils.validation import pop_args
from .base_processor import BaseProcessor

logger = logging.getLogger(__name__)


DEFAULT_FILTERS: Dict[str, FilterCondition] = {
    'filter-by-qval': FilterCondition(key='qvals', cutoff=0.05, greater_is_better=False),
    'filter-by-log2fc': FilterCondition(key='log2fc', cutoff=1, greater_is_better=True),
}


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
    gene_names_key: str
        Key to use for parsing gene symbols (names). Can be 'var_names' or
        any other key 'var.*'.
    kwargs: dict
        These will be passed to gp.enrichr.
    max_workers: int
        Number of threads to launch for enrichment analysis. If None, will
        set to number of CPU's on the machine. Max allowed numer of workers
        is 2 * number of CPU's.
    """

    class Config(BaseProcessor.Config):
        read_key: str = f"uns.{UNS.TTEST}"
        save_key: str = f"uns.{UNS.GSEA}"

        gene_sets: List[str] | str = "HuBMAP_ASCTplusB_augmented_2022"
        # Dict of keys to use for filtering DE genes; keys are ignored
        filter_by: Dict[str, FilterCondition] = DEFAULT_FILTERS
        gene_names_key: str = "var_names"
        kwargs: Dict[str, Any] = {}

        # Set max to 4; don't want to DDOS
        max_workers: Optional[int] = Field(None, ge=1, le=4, exclude=True)

        @validator('max_workers')
        def init_max_workers(cls, val):
            return 4 if val is None else val

        @validator('kwargs')
        def remove_explicit_args(cls, val):
            return pop_args(['gene_list', 'gene_sets', 'no_plot'], val)

    cfg: Config

    @staticmethod
    def _gsea(
        gene_list: List[str] | NP1D_str,
        gene_sets: List[str] | str = "HuBMAP_ASCTplusB_augmented_2022",
        **kwargs
    ) -> pd.DataFrame:
        """Wrapper around gp.enrichr."""
        if hasattr(gene_list, 'tolist') and not isinstance(gene_list, list):  # 2nd if for mypy
            gene_list = gene_list.tolist()
        return gp.enrichr(gene_list=gene_list, gene_sets=gene_sets, no_plot=True, **kwargs).results

    @staticmethod
    def _process_de_test(
        test: pd.DataFrame | DETestSummary,
        gene_list_all: NP1D_str,
        filter_by: Dict[str, FilterCondition]
    ) -> pd.DataFrame:
        """Process a single DataFrame or DETestSummary object.

        Parameters
        __________
        test: pd.DataFrame or DETestSummary
            Must contain keys specified in all FilterCondition's passed.
        gene_list_all: ndarray of str
            List of all genes to select from. Must have the same length as
            test.
        filter_by: dict of FilterCondition
            Determine which genes to pick from gene_list_all based on
            results of test.

        Returns
        _______
        A pandas DataFrame with test results.
        """
        if isinstance(test, pd.DataFrame):
            test = DETestSummary.from_df(test)
        elif not isinstance(test, DETestSummary):
            raise TypeError(f"Expected DataFrame or DETestSummary but found {type(test)}.")

        if len(gene_list_all) != len(test):
            raise ValueError(
                "Expected gene_list to be of same length as DETestSummary, but "
                f"found gene_list of length {len(gene_list_all)} and DETestSummary "
                f"of length {len(test)}."
            )

        # Apply all filters
        gene_idx: NP1D_int = test.where(*filter_by.values(), as_mask=False)
        gene_list = gene_list_all[gene_idx]
        if len(gene_list) == 0:  # empty list
            logger.warning('Encountered empty gene list.')
            # Empty dataframe
            return pd.DataFrame(columns=[
                'Gene_set', 'Term', 'Overlap', 'P-value', 'Adjusted P-value',
                'Old P-value', 'Old Adjusted P-value', 'Odds Ratio', 'Combined Score',
                'Genes'])
        return GSEA._gsea(gene_list)

    def _process(self, adata: AnnData) -> None:
        # Get list of all gene names
        gene_list_all = self.get_repr(adata, self.cfg.gene_names_key)
        gene_list_all = column_or_1d(gene_list_all).astype(str)
        gene_list_all = np.char.upper(gene_list_all)

        if len(gene_list_all) != adata.shape[1]:
            logger.warning(
                "Gene list has a different dimension than AnnData's column dimension. "
                "Please make sure 'read_key' is what you intended to use."
            )

        tests = self.get_repr(adata, self.cfg.read_key)
        if isinstance(tests, dict):  # Dict of tests
            _gsea_f = partial(
                GSEA._process_de_test,
                gene_list_all=gene_list_all,
                filter_by=self.cfg.filter_by
            )
            # We multithread this since gseapy makes http requests
            with ThreadPoolExecutor(max_workers=self.cfg.max_workers) as executor:
                gsea_test_summaries = executor.map(_gsea_f, tests.values())

            for label, gsea_test_summary in zip(tests, gsea_test_summaries):
                save_key = f'{self.cfg.save_key}.{label}'
                self.set_repr(adata, save_key, gsea_test_summary)
        else:  # Single test
            gsea_test_summary = GSEA._process_de_test(
                tests,
                gene_list_all=gene_list_all,
                filter_by=self.cfg.filter_by)
            self.set_repr(adata, self.cfg.save_key, gsea_test_summary)
