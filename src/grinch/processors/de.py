import abc
import logging
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from operator import attrgetter
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from diptest import diptest
from pydantic import Field, field_validator
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
    test_type: 'one_vs_all' or 'one_vs_one'.
        If 'one_vs_all' will run the ttest by comparing the mean of one
        cluster vs the mean of all other clusters combined. If 'one_vs_one'
        will compare against a single cluster specified by `control_label`.
    control_label: str
        The label to use in a 'one_vs_one' test type. Must be present in
        the array specified by `group_key`.
    control_key: str
        Alternatively, the control data matrix can be obtained from a
        separate key. This is useful when, e.g., you wish to cluster and
        work with only a subset of the data, but comparison for ttest
        should be done against control.
    """

    class Config(BaseProcessor.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'PairwiseDETest']

        x_key: ReadKey = "X"
        group_key: ReadKey
        save_key: WriteKey

        is_logged: bool = True
        # If the data is logged, this should point to the base of the
        # logarithm used. Can be 'e' or a positive float.
        base: float | str | None = Field('e')
        correction: str = 'fdr_bh'
        replace_nan: bool = True

        # If any of the following is not None, will perform a one_vs_one
        # test. E.g., if control samples are given in `control_key`, then for
        # each group G determined by `group_key`, will run the test G vs
        # control. If both `experimental` and `control` samples are given,
        # will perform a single one_vs_one test between those two.
        experimental_label: str | None = None
        experimental_key: ReadKey | None = None
        control_label: str | None = None
        control_key: ReadKey | None = None

        show_progress_bar: bool = Field(True, exclude=True)

        @field_validator('save_key')
        def _starts_with_uns(cls, save_key):
            if save_key.split('.')[0] != 'uns':
                raise ValueError("Anndata column for DE Test should be 'uns'.")
            return save_key

        @field_validator('base')
        def _remove_base_if_not_logged(cls, base, info):
            return None if not info.data['is_logged'] else base

        @property
        def is_ovo(self) -> bool:
            return any_not_None(self.control_key, self.control_label,
                                self.experimental_key, self.experimental_label)

        def get_label_key(self, label: str) -> str:
            key = (
                f"{self.save_key}"
                f".{self.group_key.rsplit('.')[-1]}"
                f"-{label}"
            )
            return key

    cfg: Config

    def get_pqvals(self, pvals):
        """Performs basic processing on p and computes qvals."""
        not_nan_mask = ~np.isnan(pvals)

        qvals = np.full_like(pvals, 1.0 if self.cfg.replace_nan else np.nan)
        # only correct not nan's.
        qvals[not_nan_mask] = _correct(pvals[not_nan_mask],
                                       method=self.cfg.correction)[1]
        if self.cfg.replace_nan:
            pvals[~not_nan_mask] = 1.0
        return pvals, qvals

    def get_log2fc(self, m1, m2):
        log2fc = _compute_log2fc(m1, m2, self.cfg.base, self.cfg.is_logged)
        return log2fc

    def _process(self, adata: AnnData) -> None:

        def extract_ovo_repr(adata, key):
            """Used if using a experimental or control data stored
            in a different key.
            """
            if key is None:
                return None
            x = self.get_repr(adata, key)
            x = check_array(x, accept_sparse='csr')
            if key.startswith('var'):
                x = x.T  # Transpose to (samples, features)
            if x.shape[1] != adata.shape[1]:
                raise ValueError(
                    "Found differing number of features for `x` and "
                    f"`adata`: {x.shape[1]} vs {adata.shape[1]}."
                )
            return x

        group_labels: NP1D_Any = column_or_1d(
            self.get_repr(adata, self.cfg.group_key)
        )
        unq_labels = np.unique(group_labels)
        x = self.get_repr(adata, self.cfg.x_key)
        x = check_array(x, accept_sparse='csr')

        x_experimental = None
        x_control = None
        if self.cfg.is_ovo:
            x_experimental = extract_ovo_repr(adata, self.cfg.experimental_key)
            x_control = extract_ovo_repr(adata, self.cfg.control_key)

        if len(unq_labels) <= 1 and all_None(x_control, x_experimental):
            logger.warning(
                "Found only one unique value "
                f"under key '{self.cfg.group_key}'."
            )
            return

        self._test(x, group_labels, x_experimental=x_experimental, x_control=x_control)

    @abc.abstractmethod
    def _test(self, x, group_labels, x_experimental, x_control) -> None:
        raise NotImplementedError


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

    def _test(self, x, group_labels: NP1D_Any, x_experimental=None, x_control=None) -> None:
        # efficient mean and variance computation
        pmv = PartMeanVar(x, group_labels, self.cfg.show_progress_bar)
        unq_labels = np.unique(group_labels)

        if self.cfg.is_ovo:
            if x_control is not None:
                mean, var = mean_var(x_control, axis=0, ddof=1)
                # TODO switch to namedtuple
                control_stats = {'n': len(x_control), 'm': mean, 'v': var}
            elif self.cfg.control_label not in unq_labels:
                raise ValueError("'control_label' not found in `group_key`.")
            else:
                n, m, v = pmv.compute([self.cfg.control_label], ddof=1)
                control_stats = {'n': n, 'm': m, 'v': v}
        else:
            control_stats = None

        to_iter = (
            tqdm(unq_labels, desc=f"Running {self.__class__.__name__}")
            if self.cfg.show_progress_bar
            else unq_labels
        )
        for label in to_iter:  # type: ignore
            if self.cfg.is_ovo and label == self.cfg.control_label:
                continue
            ts: DETestSummary = self._single_test(pmv, label, control_stats)
            key: str = self.cfg.get_label_key(label)
            self.store_item(key, ts.df())

    def _single_test(self, pmv: PartMeanVar, label, control_stats=None) -> DETestSummary:
        """Perform a single ttest."""
        n1, m1, v1 = pmv.compute([label], ddof=1)  # take label
        if control_stats is not None:
            n2, m2, v2 = control_stats['n'], control_stats['m'], control_stats['v']
        else:
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

        if TYPE_CHECKING:
            create: Callable[..., 'KSTest']

        save_key: WriteKey = f"uns.{UNS.KSTEST}"
        method: str = 'auto'
        alternative: str = 'two-sided'
        max_workers: Optional[int] = Field(None, ge=1, le=2 * mp.cpu_count(),
                                           exclude=True)

    cfg: Config

    def _test(self, x, group_labels: NP1D_Any, x_experimental=None, x_control=None) -> None:
        pmv = PartMeanVar(x, group_labels, self.cfg.show_progress_bar)
        spmat_warning = "KS Test cannot work with sparse matrices. Densifying..."
        if sp.issparse(x):
            logger.warning(spmat_warning)
            x = x.toarray()
        if x_control is not None and sp.issparse(x_control):
            logger.warning(spmat_warning)
            x_control = x_control.toarray()

        unq_labels, groups = group_indices(group_labels, as_mask=True)
        to_iter = (
            tqdm(unq_labels, desc="Running Kolmogorov-Smirnov tests.")
            if self.cfg.show_progress_bar
            else unq_labels
        )

        x, = indexable(x)

        y, m2 = None, None
        if self.cfg.is_ovo:
            if x_control is not None:
                y = x_control
                m2 = np.ravel(y.mean(axis=0))
            else:
                _control_idx = unq_labels.tolist().index(self.cfg.control_label)
                _control_group = groups[_control_idx]
                _, m2, _ = pmv.compute([self.cfg.control_label], ddof=1)
                y = x[_control_group]

        for label, group in zip(to_iter, groups):  # type: ignore
            if self.cfg.is_ovo and self.cfg.control_label is not None:
                if self.cfg.control_label == label:
                    continue
            ts: KSTestSummary = self._single_test(
                pmv, label, x=x[group],
                y=y if y is not None else x[~group], m2=m2,
            )
            key: str = self.cfg.get_label_key(label)
            self.store_item(key, ts.df())

    def _single_test(
        self,
        pmv: PartMeanVar,
        label,
        *,
        x, y,
        m2: NP1D_float | None = None,
    ) -> KSTestSummary:
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
        if m2 is None:
            _, m2, _ = pmv.compute([label], ddof=1, exclude=True)  # all but label
        log2fc = self.get_log2fc(m1, m2)
        return KSTestSummary(pvals=pvals, qvals=qvals, mean1=m1, mean2=m2,
                             statistic=stats, statistic_sign=statistic_sign,
                             log2fc=log2fc)


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
