import abc
import logging
from functools import partial
from typing import Any, Callable, Dict, List, Type

import gseapy as gp
import numpy as np
import pandas as pd
from anndata import AnnData
from pydantic import validator
from sklearn.utils.validation import column_or_1d

from ..aliases import UNS, VAR
from ..cond_filter import Filter, StackedFilter
from ..custom_types import NP1D_int, NP1D_str
from ..de_test_summary import DETestSummary, TestSummary
from ..shortcuts import FDRqVal_Filter_05, log2fc_Filter_1, qVal_Filter_05
from ..utils.decorators import retry
from ..utils.validation import pop_args
from .base_processor import BaseProcessor

logger = logging.getLogger(__name__)


DEFAULT_ENRICH_FILTERS: List[Filter] = [qVal_Filter_05(), log2fc_Filter_1()]

DEFAULT_PRERANK_FILTERS: List[Filter] = [qVal_Filter_05()]

DEFAULT_LEAD_GENE_FILTERS: List[Filter] = [FDRqVal_Filter_05()]

DEFAULT_GENE_SET = "HuBMAP_ASCTplusB_augmented_2022"

EMPTY_ENRICH_TEST = pd.DataFrame(columns=[
    'Gene_set', 'Term', 'Overlap', 'P-value', 'Adjusted P-value',
    'Old P-value', 'Old Adjusted P-value', 'Odds Ratio', 'Combined Score',
    'Genes', 'N_Genes_Tested',
])

EMPTY_PRERANK_TEST = pd.DataFrame(columns=[
    'Name', 'Term', 'ES', 'NES', 'NOM p-val', 'FDR q-val', 'FWER p-val',
    'Tag %', 'Gene %', 'Lead_genes'
])


class GSEA(BaseProcessor, abc.ABC):
    """Will parse a dict of dataframes or a single dataframe and
    select the top k genes to perform GSEA based on user defined criteria.

    To get a list of all gene sets, run
    ```
    import gseapy
    gseapy.get_library_name()
    ```

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
    filter_by: list of Filter
        These will be used to filter genes for GSEA. Dict keys are ignored.
    gene_names_key: str
        Key to use for parsing gene symbols (names). Can be 'var_names' or
        any other key 'var.*'.
    kwargs: dict
        These will be passed to gp.enrichr.
    """

    class Config(BaseProcessor.Config):
        read_key: str = f"uns.{UNS.TTEST}"
        save_key: str

        gene_sets: List[str] | str = DEFAULT_GENE_SET
        # Dict of keys to use for filtering DE genes; keys are ignored
        filter_by: List[Filter]
        gene_names_key: str = "var_names"
        # not to be used in a config
        test_type: Type[TestSummary] = DETestSummary
        kwargs: Dict[str, Any] = {}

        @validator('filter_by', pre=True, always=True)
        def ensure_filter_list(cls, val):
            return [val] if isinstance(val, Filter) else val

    cfg: Config

    @staticmethod
    @abc.abstractmethod
    def _gsea(test: DETestSummary, gene_sets: List[str] | str, **kwargs):
        raise NotImplementedError

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
            type(self)._process_de_test,
            gene_list_all=gene_list_all,
            gene_sets=self.cfg.gene_sets,
            filter_by=self.cfg.filter_by,
            gsea_func=type(self)._gsea,
            test_type=self.cfg.test_type,
            seed=self.cfg.seed,
            **self.cfg.kwargs,
        )
        tests = self.get_repr(adata, self.cfg.read_key)
        if isinstance(tests, dict):  # Dict of tests
            self._process_dict(tests, prefix='', func=_gsea_f)
        else:  # Single test
            gsea_test_summary = _gsea_f(tests)
            self.store_item(self.cfg.save_key, gsea_test_summary)

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

    @staticmethod
    def _process_de_test(
        test: pd.DataFrame | TestSummary,
        gene_list_all: NP1D_str,
        *,
        filter_by: List[Filter],
        gsea_func: Callable,
        gene_sets: List[str] | str = DEFAULT_GENE_SET,
        test_type: Type[TestSummary] = DETestSummary,
        **kwargs,
    ) -> pd.DataFrame:
        """Process a single DataFrame or TestSummary object.

        Parameters
        __________
        test: pd.DataFrame or TestSummary
            Must contain keys specified in all Filter's passed.
        gene_list_all: ndarray of str
            List of all genes to select from. Must have the same length as
            test.
        filter_by: List of Filter
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
                f"Expected DataFrame or {test_type.__name__}"
                f" but found {type(test)}."
            )
        if len(gene_list_all) != len(test):
            raise ValueError(
                "Expected gene_list to be of same length "
                f"as {test_type.__name__}, but "
                f"found gene_list of length {len(gene_list_all)} "
                f"and {test_type.__name__} "
                f"of length {len(test)}."
            )
        test.name = gene_list_all
        # Apply all filters
        gene_mask: NP1D_int = test.where(*filter_by, as_mask=True)
        test = test[gene_mask]

        if len(test) == 0:  # empty list
            logger.warning('Encountered empty gene list.')
            # Empty dataframe
            return pd.DataFrame()
        return gsea_func(test, gene_sets=gene_sets, **kwargs)


class GSEAEnrich(GSEA):
    """Performs gene set enrichment analysis enrichment.
    """

    class Config(GSEA.Config):
        save_key: str = f"uns.{UNS.GSEA_ENRICH}"
        filter_by: List[Filter] = DEFAULT_ENRICH_FILTERS

        @validator('kwargs')
        def remove_explicit_args(cls, val):
            return pop_args(['gene_list', 'gene_sets', 'no_plot'], val)

    cfg: Config

    @staticmethod
    @retry(5, msg="Error sending gene list", logger=logger, sleep=1)
    def _gsea(
        test: DETestSummary,
        gene_sets: List[str] | str = DEFAULT_GENE_SET,
        **kwargs
    ) -> pd.DataFrame:
        """Wrapper around gp.enrichr."""
        gene_list = test.name.tolist()  # type: ignore
        logger.info(f"Using {len(gene_list)} genes.")
        _ = kwargs.pop("seed", None)
        try:
            results = gp.enrichr(gene_list=gene_list, gene_sets=gene_sets,
                                 no_plot=True, **kwargs).results
            results['N_Genes_Tested'] = len(gene_list)
        except ValueError as ve:
            # Occurs when no gene set has a hit
            logger.warning(f"No hits found. {str(ve)}")
            return EMPTY_ENRICH_TEST
        return results


class GSEAPrerank(GSEA):
    """Runs the prerank GSEA module.
    """

    class Config(GSEA.Config):
        save_key: str = f"uns.{UNS.GSEA_PRERANK}"
        filter_by: List[Filter] = DEFAULT_PRERANK_FILTERS
        seed: int = 123  # Prerank doesn't accept null seeds

        @validator('kwargs')
        def remove_explicit_args(cls, val):
            return pop_args(['rnk', 'gene_sets', 'outdir', 'seed'], val)

    cfg: Config

    @staticmethod
    @retry(5, msg="Error sending gene list", logger=logger, sleep=1)
    def _gsea(
        test: DETestSummary,
        gene_sets: List[str] | str = DEFAULT_GENE_SET,
        **kwargs
    ) -> pd.DataFrame:
        """Wrapper around gp.prerank."""
        rnk = pd.DataFrame(data=test.log2fc, index=test.name)
        logger.info(f"Using {len(rnk)} genes.")
        try:
            results = gp.prerank(rnk=rnk, gene_sets=gene_sets,
                                 outdir=None, **kwargs).res2d
        except ValueError as ve:
            # Occurs when no gene set has a hit
            logger.warning(f"No hits found. {str(ve)}")
            return EMPTY_PRERANK_TEST
        to_numeric_cols = ['ES', 'NES', 'NOM p-val', 'FDR q-val', 'FWER p-val']
        for col in to_numeric_cols:
            results[col] = pd.to_numeric(results[col])
        return results


class FindLeadGenes(BaseProcessor):
    """Compute a mask of lead genes as determined by the significant
    GSEA Prerank processes.
    """

    class Config(BaseProcessor.Config):
        read_key: str = f"uns.{UNS.GSEA_PRERANK}"
        all_leads_save_key: str = f"var.{VAR.IS_LEAD}"
        lead_group_save_key: str = f"var.{VAR.LEAD_GROUP}"
        filter_by: List[Filter] = DEFAULT_LEAD_GENE_FILTERS
        gene_names_key: str = "var_names"

        @validator('filter_by', pre=True, always=True)
        def ensure_filter_list(cls, val):
            return [val] if isinstance(val, Filter) else val

    cfg: Config

    def _process(self, adata: AnnData) -> None:
        gene_list_all = self.get_repr(adata, self.cfg.gene_names_key)
        gene_list_all = np.char.upper(column_or_1d(gene_list_all).astype(str))

        gsea_prerank_dict = self.get_repr(adata, self.cfg.read_key)
        if not isinstance(gsea_prerank_dict, dict):
            raise ValueError("Expected a dictionary of GSEA Prerank tests.")
        filter_by = StackedFilter(*self.cfg.filter_by)

        all_leads = np.full(adata.shape[1], False)  # [i]=True if i is lead somewhere
        lead_group = np.full(adata.shape[1], "", dtype=object)  # [i]=Group if i is lead in Group

        for group, df in gsea_prerank_dict.items():
            # First find all the lead genes for all significant processes.
            # TODO make sure this comforms with GSEA prerank results
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Expected a test results in DataFrame format.")
            if 'Lead_genes' not in df:
                raise ValueError("Expected a `Lead_genes` column in the DataFrame.")

            sig_processes_mask = filter_by(df)  # type: ignore
            sig_processes_df = df.iloc[sig_processes_mask]
            # GSEAPy returns these genes as a single string sep by ;
            lead_genes = sig_processes_df['Lead_genes'].str.cat(sep=';')
            lead_genes = np.unique(lead_genes.split(';'))

            lead_genes_mask = np.in1d(gene_list_all, lead_genes)
            all_leads[lead_genes_mask] = True
            lead_group[lead_genes_mask] += f"{group};"

        lead_group = lead_group.astype(str)
        lead_group = np.char.rstrip(lead_group, ";")  # remove trailing ;
        self.store_item(self.cfg.all_leads_save_key, all_leads)
        self.store_item(self.cfg.lead_group_save_key, lead_group)
