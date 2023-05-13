import logging
from functools import partial
from typing import Any, Callable, Dict, List, Type

import gseapy as gp
import numpy as np
import pandas as pd
from anndata import AnnData
from pydantic import validator
from sklearn.utils.validation import column_or_1d

from ..aliases import UNS
from ..custom_types import NP1D_int, NP1D_str
from ..de_test_summary import DETestSummary, FilterCondition, TestSummary
from ..utils.validation import pop_args
from .base_processor import BaseProcessor

logger = logging.getLogger(__name__)


DEFAULT_FILTERS: List[FilterCondition] = [
    FilterCondition(key='qvals', cutoff=0.05, greater_is_True=False, dtype='float'),
    FilterCondition(key='log2fc', cutoff=1, greater_is_True=True, dtype='float'),
]

DEFAULT_GENE_SET = "HuBMAP_ASCTplusB_augmented_2022"

EMPTY_TEST = pd.DataFrame(columns=[
    'Gene_set', 'Term', 'Overlap', 'P-value', 'Adjusted P-value',
    'Old P-value', 'Old Adjusted P-value', 'Odds Ratio', 'Combined Score',
    'Genes'
])


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
    filter_by: list of FilterCondition
        These will be used to filter genes for GSEA. Dict keys are ignored.
    gene_names_key: str
        Key to use for parsing gene symbols (names). Can be 'var_names' or
        any other key 'var.*'.
    kwargs: dict
        These will be passed to gp.enrichr.
    """

    class Config(BaseProcessor.Config):
        read_key: str = f"uns.{UNS.TTEST}"
        save_key: str = f"uns.{UNS.GSEA}"

        gene_sets: List[str] | str = DEFAULT_GENE_SET
        # Dict of keys to use for filtering DE genes; keys are ignored
        filter_by: List[FilterCondition] = DEFAULT_FILTERS
        gene_names_key: str = "var_names"
        kwargs: Dict[str, Any] = {}
        # not to be used in a config
        test_type: Type[TestSummary] = DETestSummary

        @validator('kwargs')
        def remove_explicit_args(cls, val):
            return pop_args(['gene_list', 'gene_sets', 'no_plot'], val)

        @validator('filter_by')
        def at_least_one_filter(cls, val):
            if len(val) < 1:
                raise ValueError("At least one filter should be provided.")
            return val

    cfg: Config

    @staticmethod
    def _gsea(
        gene_list: List[str] | NP1D_str,
        gene_sets: List[str] | str = DEFAULT_GENE_SET,
        **kwargs
    ) -> pd.DataFrame:
        """Wrapper around gp.enrichr."""
        if hasattr(gene_list, 'tolist') and not isinstance(gene_list, list):
            gene_list = gene_list.tolist()

        logger.info(f"Using {len(gene_list)} genes.")
        try:
            results = gp.enrichr(
                gene_list=gene_list,
                gene_sets=gene_sets,
                no_plot=True,
                **kwargs,
            ).results
        except ValueError as ve:
            # Occurs when no gene set has a hit
            logger.warning(f"No hits found. {str(ve)}")
            return EMPTY_TEST
        return results

    @staticmethod
    def _process_de_test(
        test: pd.DataFrame | TestSummary,
        gene_list_all: NP1D_str,
        *,
        filter_by: List[FilterCondition],
        gene_sets: List[str] | str = DEFAULT_GENE_SET,
        test_type: Type[TestSummary] = DETestSummary,
    ) -> pd.DataFrame:
        """Process a single DataFrame or TestSummary object.

        Parameters
        __________
        test: pd.DataFrame or TestSummary
            Must contain keys specified in all FilterCondition's passed.
        gene_list_all: ndarray of str
            List of all genes to select from. Must have the same length as
            test.
        filter_by: List of FilterCondition
            Determine which genes to pick from gene_list_all based on
            results of test.
        test_type: type
            Should point to the type of test to convert test to.

        Returns
        _______
        A pandas DataFrame with test results.
        """
        if isinstance(test, pd.DataFrame):
            test = test_type.from_df(test)
        elif not isinstance(test, test_type):
            raise TypeError(
                f"Expected DataFrame or {test_type.__name__} but found {type(test)}."
            )

        if len(gene_list_all) != len(test):
            raise ValueError(
                f"Expected gene_list to be of same length as {test_type.__name__}, but "
                f"found gene_list of length {len(gene_list_all)} and {test_type.__name__} "
                f"of length {len(test)}."
            )

        # Apply all filters
        gene_idx: NP1D_int = test.where(*filter_by, as_mask=False)
        gene_list = gene_list_all[gene_idx]
        if len(gene_list) == 0:  # empty list
            logger.warning('Encountered empty gene list.')
            # Empty dataframe
            return EMPTY_TEST
        return GSEA._gsea(gene_list, gene_sets=gene_sets)

    def _process_dict(
        self,
        test_dict: Dict[str, Dict | pd.DataFrame | TestSummary],
        prefix: str,
        func: Callable
    ) -> None:
        """Recursively processes a dict."""
        for k, v in test_dict.items():
            if isinstance(v, dict):
                self._process_dict(v, prefix=f'{k}.', func=func)
            else:
                logger.info(f"Running GSEA for '{k}'.")
                gsea_test_summary: pd.DataFrame = func(v)
                save_key = f'{self.cfg.save_key}.{prefix}{k}'
                self.store_item(save_key, gsea_test_summary)

    def _process(self, adata: AnnData) -> None:
        # Get list of all gene names
        gene_list_all = self.get_repr(adata, self.cfg.gene_names_key)
        gene_list_all = np.char.upper(column_or_1d(gene_list_all).astype(str))

        if len(gene_list_all) != adata.shape[1]:
            logger.warning(
                "Gene list has a different dimension than AnnData's column dimension. "
                "Please make sure 'read_key' is what you intended to use."
            )

        _gsea_f = partial(
            GSEA._process_de_test,
            gene_list_all=gene_list_all,
            gene_sets=self.cfg.gene_sets,
            filter_by=self.cfg.filter_by,
            test_type=self.cfg.test_type,
        )
        tests = self.get_repr(adata, self.cfg.read_key)
        if isinstance(tests, dict):  # Dict of tests
            self._process_dict(tests, prefix='', func=_gsea_f)
        else:  # Single test
            gsea_test_summary = _gsea_f(tests)
            self.store_item(self.cfg.save_key, gsea_test_summary)
