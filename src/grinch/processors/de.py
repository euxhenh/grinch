import abc
import logging
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from operator import attrgetter
from typing import Iterable, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from diptest import diptest
from pydantic import Field, validator
from scipy.stats import ks_2samp
from sklearn.utils import check_array, column_or_1d, indexable
from tqdm.auto import tqdm

from ..aliases import UNS
from ..custom_types import NP1D_Any, NP1D_float
from ..de_test_summary import BimodalTestSummary, DETestSummary, KSTestSummary
from ..utils.stats import (
    PartMeanVar,
    _compute_log2fc,
    _correct,
    group_indices,
    ttest_from_mean_var,
)
from .base_processor import BaseProcessor

logger = logging.getLogger(__name__)


class PairwiseDETest(BaseProcessor, abc.ABC):
    """A base class for differential expression testing that
    compares two distributions.

    Parameters
    __________
    x_key: str
        Points to the data matrix that will be used to run t-tests. The
        first (0) axis should consist of observations and the second axis
        should consist of the genes.
    save_key: str
        Points to a location where the test results will be saved. This
        should start with 'uns' as we are storing a dictionary of
        dataframes.
    group_key: str
        The column to look for group labels. Must be 1D.
    is_logged: bool
        Will only affect the computation of log2 fold-change.
    base: str or float
        Will only affect the computation of log2 fold-change. Is ignored if
        data is not logged. Should point to the base that was used to take
        the log of the data in order to convert to base 2 for fold-change.
    correction: str
        P-value correction to use.
    show_progress_bar: bool
        Whether to draw a tqdm progress bar.
    replace_nan: bool
        If True, will not allow nan's in the TTest Summary dataframes.
        These will be replaced with appropriate values (1 for p-values).
    """

    class Config(BaseProcessor.Config):
        x_key: str = "X"
        save_key: str
        group_key: str

        is_logged: bool = True
        # If the data is logged, this should point to the base of the
        # logarithm used. Can be 'e' or a positive float.
        base: Optional[float | str] = Field('e', gt=0, regex='^e$')
        correction: str = 'fdr_bh'
        replace_nan: bool = True

        show_progress_bar: bool = Field(True, exclude=True)

        @validator('save_key')
        def _starts_with_uns(cls, save_key):
            if save_key.split('.')[0] != 'uns':
                raise ValueError("Anndata column for DE Test should be 'uns'.")
            return save_key

        @validator('base')
        def _remove_base_if_not_logged(cls, base, values):
            return None if not values['is_logged'] else base

        def get_label_key(self, label: str) -> str:
            key = (
                f"{self.save_key}."
                f"{self.group_key.rsplit('.')[-1]}-"
                f"{label}"
            )
            return key

    cfg: Config

    def get_pqvals(self, pvals):
        """Performs basic processing on p and computes qvals."""
        not_none_mask = ~np.isnan(pvals)

        qvals = np.full_like(pvals, 1.0 if self.cfg.replace_nan else np.nan)
        # only correct not nan's.
        qvals[not_none_mask] = _correct(pvals[not_none_mask],
                                        method=self.cfg.correction)[1]
        if self.cfg.replace_nan:
            pvals[~not_none_mask] = 1.0
        return pvals, qvals

    def get_log2fc(self, m1, m2):
        log2fc = _compute_log2fc(m1, m2, self.cfg.base, self.cfg.is_logged)
        return log2fc

    def _process(self, adata: AnnData) -> None:
        group_labels: NP1D_Any = column_or_1d(
            self.get_repr(adata, self.cfg.group_key)
        )
        unq_labels = np.unique(group_labels)
        if len(unq_labels) <= 1:
            logger.warning(
                "Found only one unique value "
                f"under key '{self.cfg.group_key}'."
            )
            return

        x = self.get_repr(adata, self.cfg.x_key)
        x = check_array(
            x,
            accept_sparse='csr',
            ensure_2d=True,
            ensure_min_features=2,
            ensure_min_samples=2,
        )
        self._test(x, group_labels)

    @abc.abstractmethod
    def _test(self, x, group_labels) -> None:
        raise NotImplementedError


class TTest(PairwiseDETest):
    """A class for performing differential expression analysis by using a
    t-Test to determine if a gene is differentially expressed in one group
    vs the other. An efficient implementation of mean and var computation
    is used to quickly run all one-vs-all tests.
    """

    class Config(PairwiseDETest.Config):
        save_key: str = f"uns.{UNS.TTEST}"

    cfg: Config

    def _test(self, x, group_labels: NP1D_Any) -> None:
        # efficient mean and variance computation
        pmv = PartMeanVar(x, group_labels, self.cfg.show_progress_bar)
        unq_labels = np.unique(group_labels)

        to_iter = (
            tqdm(unq_labels, desc=f"Running {self.__class__.__name__}")
            if self.cfg.show_progress_bar
            else unq_labels
        )
        for label in to_iter:
            ts: DETestSummary = self._single_test(pmv, label)
            key: str = self.cfg.get_label_key(label)
            self.store_item(key, ts.df())

    def _single_test(self, pmv: PartMeanVar, label) -> DETestSummary:
        """Perform a single ttest."""
        n1, m1, v1 = pmv.compute([label], ddof=1)  # take label
        n2, m2, v2 = pmv.compute([label], ddof=1, exclude=True)  # all but label

        pvals = ttest_from_mean_var(n1, m1, v1, n2, m2, v2)[1]
        pvals, qvals = self.get_pqvals(pvals)
        log2fc = self.get_log2fc(m1, m2)
        return DETestSummary(pvals=pvals, qvals=qvals,
                             mean1=m1, mean2=m2, log2fc=log2fc)


class KSTest(PairwiseDETest):
    """A class for comparing two distributions based on the
    Kolmogorov-Smirnov Test.
    https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    """

    class Config(PairwiseDETest.Config):
        save_key: str = f"uns.{UNS.KSTEST}"
        method: str = 'auto'
        alternative: str = 'two-sided'
        max_workers: Optional[int] = Field(None, ge=1, le=2 * mp.cpu_count(),
                                           exclude=True)

    cfg: Config

    def _test(self, x, group_labels: NP1D_Any) -> None:
        pmv = PartMeanVar(x, group_labels, self.cfg.show_progress_bar)
        unq_labels, groups = group_indices(group_labels, as_mask=True)

        to_iter = (
            tqdm(unq_labels, desc=f"Running {self.__class__.__name__}")
            if self.cfg.show_progress_bar
            else unq_labels
        )

        if sp.issparse(x):
            logger.warning((
                "KS Test cannot work with sparse matrices. "
                "Densifying..."
            ))
            x = x.toarray()

        x, = indexable(x)
        for label, group in zip(to_iter, groups):
            ts: KSTestSummary = self._single_test(
                pmv, label,
                x=x[group], y=x[~group],
            )
            key: str = self.cfg.get_label_key(label)
            self.store_item(key, ts.df())

    def _single_test(self, pmv: PartMeanVar, label, *, x, y) -> KSTestSummary:
        """Perform a single ks test"""
        part_ks_2samp = partial(
            ks_2samp,
            alternative=self.cfg.alternative,
            method=self.cfg.method
        )

        with ThreadPoolExecutor(max_workers=self.cfg.max_workers) as executor:
            test_results: Iterable = executor.map(part_ks_2samp, x.T, y.T)

        stat_getter = attrgetter('pvalue', 'statistic', 'statistic_sign')
        stats = np.asarray([stat_getter(res) for res in test_results])
        pvals, stats, statistic_sign = stats.T

        pvals, qvals = self.get_pqvals(pvals)
        _, m1, _ = pmv.compute([label], ddof=1)  # take label
        _, m2, _ = pmv.compute([label], ddof=1, exclude=True)  # all but labl
        log2fc = self.get_log2fc(m1, m2)
        return KSTestSummary(pvals=pvals, qvals=qvals, mean1=m1, mean2=m2,
                             statistic=stats, statistic_sign=statistic_sign,
                             log2fc=log2fc)


class BimodalTest(BaseProcessor):

    class Config(BaseProcessor.Config):
        x_key: str = "X"
        save_key: str = f"uns.{UNS.BIMODALTEST}"
        correction: str = 'fdr_bh'
        skip_zeros: bool = False

        max_workers: Optional[int] = Field(None, ge=1, le=2 * mp.cpu_count(),
                                           exclude=True)

        @validator('max_workers')
        def init_max_workers(cls, val):
            return 2 * mp.cpu_count() if val is None else val

        @validator('save_key')
        def _starts_with_uns(cls, save_key):
            if save_key.split('.')[0] != 'uns':
                raise ValueError(
                    "Anndata column for bimodaltest should be 'uns'."
                )
            return save_key

    cfg: Config

    def _process(self, adata: AnnData) -> None:
        x = self.get_repr(adata, self.cfg.x_key)
        x = check_array(
            x,
            accept_sparse='csr',
            ensure_2d=True,
            ensure_min_features=2,
            ensure_min_samples=2,
        )

        def _diptest_sp_wrapper(_x):
            # slow to densify each column separately, but is memory efficient
            arr = np.ravel(_x.toarray()) if sp.issparse(_x) else _x
            if self.cfg.skip_zeros:
                arr = arr[arr != 0]
            if len(arr) <= 3:  # diptest is not defined for n <= 3
                return (np.nan, 1)
            return diptest(arr)

        with ThreadPoolExecutor(max_workers=self.cfg.max_workers) as executor:
            test_results: Iterable[Tuple[float, float]] = executor.map(
                _diptest_sp_wrapper, x.T)
        test_results = np.asarray(list(test_results))

        # first dimension is the dip statistic, the second is the pvalue
        stats, pvals = test_results[:, 0], test_results[:, 1]
        qvals: NP1D_float = _correct(pvals, method=self.cfg.correction)[1]

        bts = BimodalTestSummary(pvals=pvals, qvals=qvals, statistic=stats)
        self.store_item(self.cfg.save_key, bts.df())
