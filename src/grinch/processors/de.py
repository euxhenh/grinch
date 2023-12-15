import abc
import logging
import multiprocessing as mp
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from operator import attrgetter
from typing import TYPE_CHECKING, Callable, Iterable, Literal, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData
from diptest import diptest
from pydantic import Field, PositiveFloat, PositiveInt, field_validator
from scipy.stats import ks_2samp, ranksums
from sklearn.utils import (
    check_array,
    check_consistent_length,
    column_or_1d,
    indexable,
)
from tqdm.auto import tqdm

from ..aliases import UNS
from ..custom_types import NP_SP, NP1D_Any, NP1D_float
from ..utils.ops import densify
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
    """

    class Config(BaseProcessor.Config):
        r"""PairwiseDETest.Config

        Parameters
        __________
        x_key : str, default='X'
            Points to the data matrix that will be used to run t-tests. The
            first (0) axis should consist of observations and the second
            axis should consist of the genes.

        group_key : str
            The column to look for group labels. Must be 1D.

        write_key : str
            Points to a location where the test results will be saved. This
            should start with 'uns' as we are storing a dictionary of
            dataframes.

        is_logged : bool, default=True
            Will only affect the computation of log2 fold-change.

        base : str or float, default='e'
            Will only affect the computation of log2 fold-change. Is
            ignored if data is not logged. Should point to the base that
            was used to take the log of the data in order to convert to
            base 2 for fold-change. Can be 'e' or a positive float.

        correction : str, default='fdr_bh'
            P-value correction to use. Any correction method supported by
            statsmodels can be used. See
            https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html

        replace_nan : bool, default=True
            If True, will not allow nan's in the TTest Summary dataframes.
            These will be replaced with appropriate values (1 for
            p-values).

        min_points_per_group : int, default=None
            If not None, will skip computing values for groups with fewer
            than this many points.

        control_group : str, default=None
            The label to use in a 'one_vs_one' test type. Must be present
            in the array specified by `group_key`.

        control_key : str, default=None
            Alternatively, the control data matrix can be obtained from a
            separate key. This is useful when, e.g., you wish to cluster
            and work with only a subset of the data, but comparison for
            ttest should be done against control.

        show_progress_bar : bool, default=True
            Whether to draw a tqdm progress bar.
        """

        if TYPE_CHECKING:
            create: Callable[..., 'PairwiseDETest']

        x_key: ReadKey = "X"
        group_key: ReadKey
        write_key: WriteKey

        is_logged: bool = True
        base: PositiveFloat | Literal['e'] | None = Field('e')
        correction: str = 'fdr_bh'
        replace_nan: bool = True
        min_points_per_group: PositiveInt | None = None

        # If any of the following is not None, will perform a one_vs_one
        # test. E.g., if control samples are given in `control_key`, then for
        # each group G determined by `group_key`, will run the test G vs
        # control.
        control_group: str | None = None
        control_key: ReadKey | None = None

        show_progress_bar: bool = Field(True, exclude=True)

        @field_validator('base')
        def _remove_base_if_not_logged(cls, base, info):
            """Set base to None if not logged."""
            return None if not info.data['is_logged'] else base

        @property
        def is_one_vs_one(self) -> bool:
            """Checks if test is one vs one.

            Returns
            -------
            is_ovo : bool
                True if any of the control keys is not None.
            """
            return any_not_None(self.control_key, self.control_group)

        def get_label_key(self, label: str) -> str:
            """Get the WriteKey to store a given test in."""
            key = (
                f"{self.write_key}"
                f".{self.group_key.rsplit('.')[-1]}"
                f"-{label}"
            )
            return key

    cfg: Config

    def get_pqvals(self, pvals: NP1D_float) -> Tuple[NP1D_float, NP1D_float]:
        """Performs basic processing on p and computes qvals.

        Parameters
        ----------
        pvals : 1D array
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
        m1, m2 : 1D arrays

        Returns
        -------
        1D array
            The computed log2 fold-change.
        """
        return _compute_log2fc(m1, m2, self.cfg.base, self.cfg.is_logged)

    def _to_iter(self, labels) -> Iterable:
        """Determine if should use tqdm"""
        if self.cfg.show_progress_bar:
            return tqdm(labels, desc=f"Running {self.__class__.__name__}")
        return labels

    def _get_x_control(self, adata, x, group_labels) -> NP_SP | None:
        """Return the x representation of a condition if set.
        """
        if all_None(self.cfg.control_key, self.cfg.control_group):
            return None

        if self.cfg.control_key is not None:  # Found in a separate key
            x_cond = self.read(adata, self.cfg.control_key)
            x_cond = check_array(x_cond, accept_sparse='csr')
            if self.cfg.control_key.startswith('var'):
                x_cond = x_cond.T  # Transpose to (samples, features)
            # Ensure same number of features
            check_consistent_length(x.T, x_cond.T)
        else:  # Index x by label
            if self.cfg.control_group not in group_labels:
                raise ValueError(f"Could not find label='{self.cfg.control_group}'.")
            x_cond = x[group_labels == self.cfg.control_group]
        return x_cond

    def _process(self, adata: AnnData) -> None:
        # Read data matrix and labels
        x = check_array(self.read(adata, self.cfg.x_key), accept_sparse='csr')
        x, = indexable(x)
        group_labels: NP1D_Any = column_or_1d(self.read(adata, self.cfg.group_key))
        check_consistent_length(x, group_labels)

        if np.unique(group_labels).size <= 1 and not self.cfg.is_one_vs_one:
            logger.warning("Cannot run test with only one group.")
            return

        pmv = PartMeanVar(x, group_labels, self.cfg.show_progress_bar)
        # Read x_control and run test
        self._test(pmv, x, group_labels, self._get_x_control(adata, x, group_labels))

    @abc.abstractmethod
    def _test(self, pmv: PartMeanVar, x, group_labels, x_control=None) -> None:
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

        write_key: WriteKey = f"uns.{UNS.TTEST}"

    cfg: Config

    def _test(
        self,
        pmv: PartMeanVar,
        x: NP_SP,
        group_labels: NP1D_Any,
        x_control: NP_SP | None = None,
    ):
        """Runs t-Test."""
        unq_labels = np.unique(group_labels)

        def get_x_stats(x_cond) -> _Statistics | None:
            if x_cond is None:
                return None
            mean, var = mean_var(x_cond, axis=0, ddof=1)
            return _Statistics(n=len(x_cond), mean=mean, var=var)

        # Skip if label is the same as control label
        for label in filter(lambda x: x != self.cfg.control_group, self._to_iter(unq_labels)):
            test = self._single_test(pmv, label, get_x_stats(x_control))
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
        if self.cfg.min_points_per_group is not None:
            if n1 < self.cfg.min_points_per_group:
                return pd.DataFrame()  # Empty
        # If no control group, will compute from all but label (one-vs-all)
        n2, m2, v2 = control_stats or pmv.compute([label], ddof=1, exclude=True)

        pvals = ttest_from_mean_var(n1, m1, v1, n2, m2, v2)[1]
        pvals, qvals = self.get_pqvals(pvals)
        log2fc = self.get_log2fc(m1, m2)

        return pd.DataFrame(
            data=dict(pvals=pvals, qvals=qvals, mean1=m1, mean2=m2, log2fc=log2fc)
        )


class RankSum(PairwiseDETest):
    """A class for performing differential expression analysis by using
    Wilcoxon rank-sum statistic to determine if a gene is differentially
    expressed in one group vs the other.
    """

    class Config(PairwiseDETest.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'RankSum']

        write_key: WriteKey = f"uns.{UNS.RANK_SUM}"
        alternative: str = 'two-sided'

    cfg: Config

    def _test(
        self,
        pmv: PartMeanVar,
        x: NP_SP,
        group_labels: NP1D_Any,
        x_control: NP_SP | None = None,
    ):
        """Runs rank sum."""
        x = densify(x, ensure_2d=True)
        if x_control is not None:
            x_control = densify(x, ensure_2d=True)
            m2 = x_control.mean(axis=0).ravel()
        else:
            m2 = None

        unq_labels, groups = group_indices(group_labels, as_mask=True)
        for label, group in zip(self._to_iter(unq_labels), groups):
            if label == self.cfg.control_group:
                continue
            y = x_control if x_control is not None else x[~group]
            test = self._single_test(pmv, label, x=x[group], y=y, m2=m2)
            self.store_item(self.cfg.get_label_key(label), test)

    def _single_test(self, pmv: PartMeanVar, label, *, x, y, m2=None) -> pd.DataFrame:
        """Perform a single rank sum test."""
        if self.cfg.min_points_per_group is not None:
            if x.shape[0] < self.cfg.min_points_per_group:
                return pd.DataFrame()

        statistic, pvals = ranksums(x, y, alternative=self.cfg.alternative)
        pvals, qvals = self.get_pqvals(pvals)
        m1 = pmv.compute([label], ddof=1)[1]  # take label
        m2 = m2 if m2 is not None else pmv.compute([label], ddof=1, exclude=True)[1]
        log2fc = self.get_log2fc(m1, m2)

        return pd.DataFrame(
            data=dict(pvals=pvals, qvals=qvals, statistic=statistic,
                      mean1=m1, mean2=m2, log2fc=log2fc)
        )


class KSTest(PairwiseDETest):
    """A class for comparing two distributions based on the
    Kolmogorov-Smirnov Test.
    https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    """

    class Config(PairwiseDETest.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'KSTest']

        write_key: WriteKey = f"uns.{UNS.KSTEST}"
        method: str = 'auto'
        alternative: str = 'two-sided'
        max_workers: int | None = Field(None, ge=1, le=2 * mp.cpu_count(), exclude=True)

    cfg: Config

    def _test(
        self,
        pmv: PartMeanVar,
        x: NP_SP,
        group_labels: NP1D_Any,
        x_control: NP_SP | None = None,
    ):
        """Runs KS test.
        """
        x, = indexable(densify(x, ensure_2d=True, warn=True))

        if x_control is not None:
            # Obtain the control samples and cache their mean
            x_control = densify(x_control, ensure_2d=True, warn=True)
            m2 = x_control.mean(axis=0).ravel()
        else:
            m2 = None

        unq_labels, groups = group_indices(group_labels, as_mask=True)
        for label, group in zip(self._to_iter(unq_labels), groups):
            if label == self.cfg.control_group:
                continue
            y = x_control if x_control is not None else x[~group]
            test = self._single_test(pmv, label, x=x[group], y=y, m2=m2)
            self.store_item(self.cfg.get_label_key(label), test)

    def _single_test(self, pmv: PartMeanVar, label, *, x, y, m2) -> pd.DataFrame:
        """Perform a single ks test"""
        if self.cfg.min_points_per_group is not None:
            if x.shape[0] < self.cfg.min_points_per_group:
                return pd.DataFrame()

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
        m2 = m2 if m2 is not None else pmv.compute([label], ddof=1, exclude=True)[1]  # all - label
        log2fc = self.get_log2fc(m1, m2)

        return pd.DataFrame(data=dict(
            pvals=pvals, qvals=qvals, mean1=m1, mean2=m2,
            statistic=stats, statistic_sign=statistic_sign, log2fc=log2fc
        ))


class UnimodalityTest(BaseProcessor):
    """A class for determining whether a distribution is unimodal.
    """

    class Config(BaseProcessor.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'UnimodalityTest']

        x_key: ReadKey = "X"
        write_key: WriteKey = f"uns.{UNS.BIMODALTEST}"
        correction: str = 'fdr_bh'
        skip_zeros: bool = False

        max_workers: int | None = Field(None, ge=1, le=2 * mp.cpu_count(), exclude=True)

        @field_validator('max_workers')
        def init_max_workers(cls, val):
            return 2 * mp.cpu_count() if val is None else val

    cfg: Config

    def _process(self, adata: AnnData) -> None:
        x = check_array(self.read(adata, self.cfg.x_key), accept_sparse='csr')

        def diptest_sp_wrapper(_x):
            arr = densify(_x).ravel()  # slow, but memory efficient
            if self.cfg.skip_zeros:
                arr = arr[arr != 0]
            # diptest is not defined for n <= 3
            return diptest(arr) if len(arr) > 3 else (np.nan, 1)

        with ThreadPoolExecutor(max_workers=self.cfg.max_workers) as executor:
            test_results: Iterable[Tuple[float, float]] = executor.map(diptest_sp_wrapper, x.T)
        test_results = np.asarray(list(test_results))

        # first dimension is the dip statistic, the second is the pvalue
        stats, pvals = test_results[:, 0], test_results[:, 1]
        qvals: NP1D_float = _correct(pvals, method=self.cfg.correction)[1]

        uts = pd.DataFrame(data=dict(pvals=pvals, qvals=qvals, statistic=stats))
        self.store_item(self.cfg.write_key, uts)
