import abc
import logging
import multiprocessing as mp
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from operator import attrgetter
from typing import TYPE_CHECKING, Callable, Iterable, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from diptest import diptest
from pydantic import Field, PositiveFloat, field_validator
from scipy.stats import ks_2samp
from sklearn.utils import (
    check_array,
    check_consistent_length,
    column_or_1d,
    indexable,
)
from tqdm.auto import tqdm

from ..aliases import UNS
from ..custom_types import NP_SP, NP1D_Any, NP1D_float
from ..utils.stats import (
    PartMeanVar,
    _compute_log2fc,
    _correct,
    group_indices,
    mean_var,
    ttest_from_mean_var,
)
from ..utils.validation import all_None, any_not_None
from .base_processor import BaseProcessor, ReadKey, WriteKey

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
    group_key: str
        The column to look for group labels. Must be 1D.
    save_key: str
        Points to a location where the test results will be saved. This
        should start with 'uns' as we are storing a dictionary of
        dataframes.
    is_logged: bool
        Will only affect the computation of log2 fold-change.
    base: str or float
        Will only affect the computation of log2 fold-change. Is ignored if
        data is not logged. Should point to the base that was used to take
        the log of the data in order to convert to base 2 for fold-change.
        Can be 'e' or a positive float.
    correction: str
        P-value correction to use. Any correction method supported by
        statsmodels can be used. See
        https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
    replace_nan: bool
        If True, will not allow nan's in the TTest Summary dataframes.
        These will be replaced with appropriate values (1 for p-values).
    control_label: str
        The label to use in a 'one_vs_one' test type. Must be present in
        the array specified by `group_key`.
    control_key: str
        Alternatively, the control data matrix can be obtained from a
        separate key. This is useful when, e.g., you wish to cluster and
        work with only a subset of the data, but comparison for ttest
        should be done against control.
    show_progress_bar: bool
        Whether to draw a tqdm progress bar.
    """

    class Config(BaseProcessor.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'PairwiseDETest']

        x_key: ReadKey = "X"
        group_key: ReadKey
        save_key: WriteKey

        is_logged: bool = True
        base: PositiveFloat | Literal['e'] | None = Field('e')
        correction: str = 'fdr_bh'
        replace_nan: bool = True

        # If any of the following is not None, will perform a one_vs_one
        # test. E.g., if control samples are given in `control_key`, then for
        # each group G determined by `group_key`, will run the test G vs
        # control.
        control_label: str | None = None
        control_key: ReadKey | None = None

        show_progress_bar: bool = Field(True, exclude=True)

        @field_validator('base')
        def _remove_base_if_not_logged(cls, base, info):
            """Set base to None if not logged.
            """
            return None if not info.data['is_logged'] else base

        @property
        def is_one_vs_one(self) -> bool:
            """Checks if test is one vs one.

            Returns: bool
                True if any of the control keys is not None.
            """
            return any_not_None(self.control_key, self.control_label)

        def get_label_key(self, label: str) -> str:
            """Get the WriteKey to store a given test in.
            """
            key = (
                f"{self.save_key}"
                f".{self.group_key.rsplit('.')[-1]}"
                f"-{label}"
            )
            return key

    cfg: Config

    def get_pqvals(self, pvals: NP1D_float) -> Tuple[NP1D_float, NP1D_float]:
        """Performs basic processing on p and computes qvals.

        Parameters
        ----------
        pvals: 1D array
            The computed p-values.

        Returns
        -------
        Tuple[1D array, 1D array]
            The p-values and the computed adjusted p-values.
        """
        not_nan_mask = ~np.isnan(pvals)

        qvals = np.full_like(pvals, 1.0 if self.cfg.replace_nan else np.nan)
        # only correct not nan's.
        qvals[not_nan_mask] = _correct(pvals[not_nan_mask], method=self.cfg.correction)[1]
        if self.cfg.replace_nan:
            pvals[~not_nan_mask] = 1.0
        return pvals, qvals

    def get_log2fc(self, m1: NP1D_float, m2: NP1D_float) -> NP1D_float:
        """Compute the log2 fold-change between the given mean vectors.

        Parameters
        ----------
        m1, m2: 1D arrays

        Returns
        -------
        1D array
            The computed log2 fold-change.
        """
        return _compute_log2fc(m1, m2, self.cfg.base, self.cfg.is_logged)

    def _process(self, adata: AnnData) -> None:
        # Read data matrix and labels
        x = check_array(self.read(adata, self.cfg.x_key), accept_sparse='csr')
        group_labels: NP1D_Any = column_or_1d(self.read(adata, self.cfg.group_key))
        check_consistent_length(x, group_labels)

        if np.unique(group_labels).size <= 1 and not self.cfg.is_one_vs_one:
            logger.warning("Cannot run test with only one group.")
            return

        def get_x_control(adata, key, label) -> NP_SP | None:
            """Return the x representation of a condition if set.
            """
            nonlocal x
            if all_None(key, label):
                return None

            if key is not None:  # Found in a separate key
                x_cond = check_array(self.read(adata, key), accept_sparse='csr')
                if key.startswith('var'):
                    x_cond = x_cond.T  # Transpose to (samples, features)
                # Ensure same number of features
                check_consistent_length(x.T, x_cond.T)
            else:  # We search x
                if label not in group_labels:
                    raise ValueError(f"Could not find {label=} in group.")
                x, = indexable(x)
                x_cond = x[group_labels == label]
            return x_cond

        # Read x_control from either key or label
        x_control = get_x_control(adata, self.cfg.control_key, self.cfg.control_label)
        # Run test
        self._test(x, group_labels, x_control)

    @abc.abstractmethod
    def _test(self, x, group_labels, x_control=None) -> None:
        raise NotImplementedError


_Statistics = namedtuple('_Statistics', ['n', 'mean', 'var'])


class TTest(PairwiseDETest):
    """A class for performing differential expression analysis by using a
    t-Test to determine if a gene is differentially expressed in one group
    vs the other. An efficient implementation of mean and var computation
    is used to quickly run all one-vs-all tests.
    """

    class Config(PairwiseDETest.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'TTest']

        save_key: WriteKey = f"uns.{UNS.TTEST}"

    cfg: Config

    def _test(self, x: NP_SP, group_labels: NP1D_Any, x_control: NP_SP | None = None):
        """Runs t-Test.
        """
        pmv = PartMeanVar(x, group_labels, self.cfg.show_progress_bar)
        unq_labels = np.unique(group_labels)

        def get_x_stats(x_cond) -> _Statistics | None:
            if x_cond is None:
                return None
            mean, var = mean_var(x_cond, axis=0, ddof=1)
            return _Statistics(n=len(x_cond), mean=mean, var=var)

        control_stats = get_x_stats(x_control)

        to_iter = (
            tqdm(unq_labels, desc=f"Running {self.__class__.__name__}")
            if self.cfg.show_progress_bar
            else unq_labels
        )
        # Skip if label is the same as control label
        for label in filter(lambda x: x != self.cfg.control_label, to_iter):
            test = self._single_test(pmv, label, control_stats)
            self.store_item(self.cfg.get_label_key(label), test)

    def _single_test(
        self,
        pmv: PartMeanVar,
        label,
        control_stats: _Statistics | None = None,
    ) -> pd.DataFrame:
        """Perform a single t-Test.
        """
        n1, m1, v1 = pmv.compute([label], ddof=1)  # Stats for label
        # If no control group, will compute from all but label (one-vs-all)
        n2, m2, v2 = control_stats or pmv.compute([label], ddof=1, exclude=True)

        pvals = ttest_from_mean_var(n1, m1, v1, n2, m2, v2)[1]
        pvals, qvals = self.get_pqvals(pvals)
        log2fc = self.get_log2fc(m1, m2)

        return pd.DataFrame(
            data=dict(pvals=pvals, qvals=qvals, mean1=m1, mean2=m2, log2fc=log2fc)
        )


class KSTest(PairwiseDETest):
    """A class for comparing two distributions based on the
    Kolmogorov-Smirnov Test.
    https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    """

    class Config(PairwiseDETest.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'KSTest']

        save_key: WriteKey = f"uns.{UNS.KSTEST}"
        method: str = 'auto'
        alternative: str = 'two-sided'
        max_workers: Optional[int] = Field(None, ge=1, le=2 * mp.cpu_count(),
                                           exclude=True)

    cfg: Config

    def _test(self, x: NP_SP, group_labels: NP1D_Any, x_control: NP_SP | None = None):
        """Runs KS test.
        """
        pmv = PartMeanVar(x, group_labels, self.cfg.show_progress_bar)

        def densify(x_cond):
            if x_cond is None:
                return x_cond
            if sp.issparse(x_cond):
                logger.warning("KS Test cannot work with sparse matrices. Densifying...")
                x_cond = x_cond.toarray()
            return x_cond

        x, = indexable(densify(x))  # This will be split into groups
        x_control = densify(x_control)
        # Cache mean
        m2 = None if x_control is None else x_control.mean(axis=0).ravel()

        unq_labels, groups = group_indices(group_labels, as_mask=True)
        to_iter = (
            tqdm(unq_labels, desc="Running Kolmogorov-Smirnov tests.")
            if self.cfg.show_progress_bar
            else unq_labels
        )

        for label, group in zip(to_iter, groups):
            if label == self.cfg.control_label:
                continue
            y = x_control if x_control is not None else x[~group]
            test = self._single_test(pmv, label, x=x[group], y=y, m2=m2)
            self.store_item(self.cfg.get_label_key(label), test)

    def _single_test(self, pmv: PartMeanVar, label, *, x, y, m2) -> pd.DataFrame:
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
        m1 = pmv.compute([label], ddof=1)[1]  # take label
        m2 = m2 or pmv.compute([label], ddof=1, exclude=True)[1]  # all but label
        log2fc = self.get_log2fc(m1, m2)

        return pd.DataFrame(data=dict(
            pvals=pvals, qvals=qvals, mean1=m1, mean2=m2,
            statistic=stats, statistic_sign=statistic_sign, log2fc=log2fc
        ))


class BimodalTest(BaseProcessor):

    class Config(BaseProcessor.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'BimodalTest']

        x_key: ReadKey = "X"
        save_key: WriteKey = f"uns.{UNS.BIMODALTEST}"
        correction: str = 'fdr_bh'
        skip_zeros: bool = False

        max_workers: int | None = Field(None, ge=1, le=2 * mp.cpu_count(),
                                        exclude=True)

        @field_validator('max_workers')
        def init_max_workers(cls, val):
            return 2 * mp.cpu_count() if val is None else val

        @field_validator('save_key')
        def _starts_with_uns(cls, save_key):
            if save_key.split('.')[0] != 'uns':
                raise ValueError(
                    "Anndata column for bimodaltest should be 'uns'."
                )
            return save_key

    cfg: Config

    def _process(self, adata: AnnData) -> None:
        x = self.read(adata, self.cfg.x_key)
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

        bts = pd.DataFrame(data=dict(pvals=pvals, qvals=qvals, statistic=stats))
        self.store_item(self.cfg.save_key, bts)
