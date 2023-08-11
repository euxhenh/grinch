import abc
import logging
import re
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List

import gseapy as gp
import numpy as np
import pandas as pd
from anndata import AnnData
from pydantic import field_validator, validate_call
from sklearn.utils.validation import check_consistent_length, column_or_1d

from ..aliases import UNS, VAR
from ..cond_filter import Filter, StackedFilter
from ..custom_types import NP1D_str
from ..shortcuts import FWERpVal_Filter_05, log2fc_Filter_1, qVal_Filter_05
from ..utils.decorators import retry
from .base_processor import BaseProcessor, ReadKey, WriteKey

logger = logging.getLogger(__name__)

DEFAULT_GENE_SET_ENRICH: str | List[str] = "HuBMAP_ASCTplusB_augmented_2022"
DEFAULT_ENRICH_FILTERS: List[Filter] = [qVal_Filter_05(), log2fc_Filter_1()]
DEFAULT_GENE_SET_PRERANK: str | List[str] = "GO_Biological_Process_2023"
DEFAULT_FILTERS_LEAD_GENES: List[Filter] = [FWERpVal_Filter_05()]


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
        These will be passed to GSEA.
    """

    class Config(BaseProcessor.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'GSEA']

        read_key: ReadKey = f"uns.{UNS.TTEST}"
        save_key: WriteKey

        gene_sets: str | List[str]
        # Dict of keys to use for filtering DE genes; keys are ignored
        filter_by: List[Filter]
        gene_names_key: str = "var_names"
        # not to be used in a config
        kwargs: Dict[str, Any] = {}

        @field_validator('filter_by', mode='before')
        def ensure_filter_list(cls, val):
            return [val] if isinstance(val, Filter) else val

    cfg: Config

    @staticmethod
    @abc.abstractmethod
    def _gsea(test: pd.DataFrame, gene_sets: str | List[str], **kwargs):
        raise NotImplementedError

    def get_gene_list_all(self, adata):
        # Get list of all gene names
        gene_list_all = self.get_repr(adata, self.cfg.gene_names_key)
        gene_list_all = np.char.upper(column_or_1d(gene_list_all).astype(str))
        return gene_list_all

    def _process(self, adata: AnnData) -> None:
        gene_list_all = self.get_gene_list_all(adata)

        if len(gene_list_all) != adata.shape[1]:
            logger.warning(
                "Gene list has a different dimension than AnnData's "
                "column dimension. Please make sure 'read_key' is what "
                "you intended to use."
            )

        _gsea_f = partial(
            self._process_de_test,
            gene_list_all=gene_list_all,
            gene_sets=self.cfg.gene_sets,
            filter_by=self.cfg.filter_by,
            gsea_func=type(self)._gsea,
            seed=self.cfg.seed,
            **self.cfg.kwargs,
        )
        tests = self.get_repr(adata, self.cfg.read_key)
        if isinstance(tests, dict):  # Dict of tests
            self._process_dict(tests, prefix='', func=_gsea_f)
        else:  # Single test
            self.store_item(self.cfg.save_key, _gsea_f(tests))

    def _process_dict(
        self,
        test_dict: Dict[str, Dict | pd.DataFrame],
        prefix: str,
        func: Callable,
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
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def _process_de_test(
        test: pd.DataFrame,
        gene_list_all: NP1D_str,
        *,
        filter_by: List[Filter],
        gsea_func: Callable,
        gene_sets: str | List[str],
        **kwargs,
    ) -> pd.DataFrame:
        """Process a single DataFrame or TestSummary object.

        Parameters
        __________
        test: pd.DataFrame
            Must contain keys specified in all Filter's passed.
        gene_list_all: ndarray of str
            List of all genes to select from. Must have the same length as
            test.
        filter_by: List of Filter
            Determine which genes to pick from gene_list_all based on
            results of test.

        Returns
        _______
        A pandas DataFrame with test results.
        """
        check_consistent_length(gene_list_all, test)
        test.index = gene_list_all
        # Apply all filters
        if len(filter_by) > 0:
            sf = StackedFilter(*filter_by)
            gene_mask = sf(test, as_mask=True)
            test = test.iloc[gene_mask]

        if len(test) == 0:  # empty list
            logger.warning('Encountered empty gene list.')
            return pd.DataFrame()  # Empty dataframe

        return gsea_func(test, gene_sets=gene_sets, **kwargs)


class GSEAEnrich(GSEA):
    """Performs gene set enrichment analysis enrichment.
    """

    class Config(GSEA.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'GSEAEnrich']

        save_key: WriteKey = f"uns.{UNS.GSEA_ENRICH}"
        gene_sets: str | List[str] = DEFAULT_GENE_SET_ENRICH
        filter_by: List[Filter] = DEFAULT_ENRICH_FILTERS

    cfg: Config

    @staticmethod
    @retry(5, msg="Error sending gene list", logger=logger, sleep=1)
    def _gsea(
        test: pd.DataFrame,
        *,
        gene_sets: str | List[str] = DEFAULT_GENE_SET_ENRICH,
        **kwargs,
    ) -> pd.DataFrame:
        """Wrapper around gp.enrichr."""
        gene_list = test.index.tolist()  # type: ignore
        logger.info(f"Using {len(gene_list)} genes.")
        _ = kwargs.pop("seed", None)

        try:
            results = gp.enrichr(gene_list=gene_list, gene_sets=gene_sets,
                                 no_plot=True, **kwargs).results
            results['N_Genes_Tested'] = len(gene_list)
        except ValueError as ve:
            # Occurs when no gene set has a hit
            logger.warning(f"No hits found. {str(ve)}")
            results = GSEAEnrich.empty_test()

        to_numeric_cols = ['P-value', 'Adjusted P-value', 'Old P-value',
                           'Old Adjusted P-value', 'Odds Ratio',
                           'Combined Score', 'N_Genes_Tested']
        for col in to_numeric_cols:
            if col in results:  # Custom sets have no old p values
                results[col] = pd.to_numeric(results[col])
        results.sort_values(by=['Adjusted P-value'], inplace=True)
        return results

    @staticmethod
    def empty_test() -> pd.DataFrame:
        return pd.DataFrame(columns=[
            'Gene_set', 'Term', 'Overlap', 'P-value', 'Adjusted P-value',
            'Old P-value', 'Old Adjusted P-value', 'Odds Ratio',
            'Combined Score', 'Genes', 'N_Genes_Tested',
        ])


class GSEAPrerank(GSEA):
    """Runs the prerank GSEA module.

    Parameters:
    qval_scaling: bool
        If True, will instead rank by -log10(qval) * log2fc.
    """

    class Config(GSEA.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'GSEAPrerank']

        save_key: WriteKey = f"uns.{UNS.GSEA_PRERANK}"
        gene_sets: str | List[str] = DEFAULT_GENE_SET_PRERANK
        # By default all genes are inputted into prerank. DE tests are
        # still needed in order to scale log2fc by the q-values before
        # ranking.
        filter_by: List[Filter] = []

        qval_scaling: bool = True
        seed: int = 123  # Prerank doesn't accept null seeds

    cfg: Config

    @staticmethod
    @retry(5, msg="Error sending gene list", logger=logger, sleep=1)
    def _gsea(
        test: pd.DataFrame,
        *,
        gene_sets: str | List[str] = DEFAULT_GENE_SET_PRERANK,
        qval_scaling: bool = True,  # pass inside cfg.kwargs
        **kwargs
    ) -> pd.DataFrame:
        """Wrapper around gp.prerank."""
        data = test.log2fc
        if qval_scaling:
            if test.qvals.min() < 1e-50:
                logger.warning("Some q-values are <1e-50.")
                qvals = np.clip(test.qvals, 1e-50, None)
            else:
                qvals = test.qvals
            data = data * (-np.log10(qvals))
        rnk = pd.DataFrame(data=data, index=test.index)

        logger.info(f"Using {len(rnk)} genes.")
        try:
            results = gp.prerank(rnk=rnk, gene_sets=gene_sets,
                                 outdir=None, **kwargs).res2d
        except KeyError as ke:
            logger.warning(f"Possibly no overlap found. {str(ke)}")
            results = GSEAPrerank.empty_test()
        except ValueError as ve:
            # Occurs when no gene set has a hit
            logger.warning(f"No hits found. {str(ve)}")
            results = GSEAPrerank.empty_test()

        to_numeric_cols = ['ES', 'NES', 'NOM p-val', 'FDR q-val', 'FWER p-val']
        for col in to_numeric_cols:
            results[col] = pd.to_numeric(results[col])
        results.sort_values(by=['FWER p-val'], inplace=True)
        return results

    @staticmethod
    def empty_test() -> pd.DataFrame:
        # Return a new dataframe to avoid referencing
        return pd.DataFrame(columns=[
            'Name', 'Term', 'ES', 'NES', 'NOM p-val', 'FDR q-val',
            'FWER p-val', 'Tag %', 'Gene %', 'Lead_genes'
        ])


class FindLeadGenes(BaseProcessor):
    """Compute a mask of lead genes as determined by the significant
    GSEA Prerank processes.
    """

    class Config(BaseProcessor.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'FindLeadGenes']

        read_key: ReadKey = f"uns.{UNS.GSEA_PRERANK}"
        all_leads_save_key: WriteKey = f"var.{VAR.IS_LEAD}"
        lead_group_save_key: WriteKey = f"var.{VAR.LEAD_GROUP}"
        filter_by: List[Filter] = DEFAULT_FILTERS_LEAD_GENES
        gene_names_key: ReadKey = "var_names"

        @field_validator('filter_by', mode='before')
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
            if len(df) == 0:  # No results for this df
                continue
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


class FindLeadGenesForProcess(BaseProcessor):
    """Finds all lead genes associated with a list of processes and
    overlaps them with genes in adata.
    """

    class Config(BaseProcessor.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'FindLeadGenesForProcess']

        gene_set: str = 'GO_Biological_Process_2023'
        organism: str = 'Human'
        terms: str | List[str] = '.*'  # by default take all
        save_key: WriteKey = f'var.{VAR.CUSTOM_LEAD_GENES}'
        regex: bool = False
        all_leads_save_key: WriteKey = f'uns.{UNS.ALL_CUSTOM_LEAD_GENES}'
        gene_names_key: ReadKey = "var_names"

        @field_validator('terms')
        def to_list(cls, val):
            if isinstance(val, str):
                return [val]
            return val

    cfg: Config

    def _process(self, adata: AnnData) -> None:
        genes = []
        lib = gp.get_library(name=self.cfg.gene_set, organism=self.cfg.organism)
        if self.cfg.regex:
            all_terms = list(lib)
            matched = 0
            for term in self.cfg.terms:
                r = re.compile(term)
                try:
                    selected_terms = list(filter(r.match, all_terms))
                    matched += len(selected_terms)
                    for selected_term in selected_terms:
                        genes.extend(lib[selected_term])
                except Exception as e:
                    logger.warning(f"Could not match term '{term}'. Skipping...")
                    print(e)
            logging.info(f"Matched a total of {matched} terms.")
        else:
            for term in self.cfg.terms:
                genes.extend(lib[term])

        lead_genes = np.unique(genes)
        gene_list_all = self.get_repr(adata, self.cfg.gene_names_key)
        gene_list_all = np.char.upper(column_or_1d(gene_list_all).astype(str))
        is_lead = np.in1d(gene_list_all, lead_genes)
        logger.info(f"Found {is_lead.sum()}/{len(lead_genes)} custom lead genes.")
        self.store_item(self.cfg.save_key, is_lead)
        self.store_item(self.cfg.all_leads_save_key, lead_genes)
