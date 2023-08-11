import logging
from dataclasses import dataclass
from functools import wraps
from typing import Any, Dict, Hashable, List, Tuple, overload

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import scipy.stats as scs
from rich.pretty import pretty_repr
from scipy.stats import rv_continuous
from scipy.stats._stats_py import (
    Ttest_indResult,
    _ttest_ind_from_stats,
    _unequal_var_ttest_denom,
)
from sklearn.utils import check_consistent_length, indexable
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm

from ..custom_types import NP1D_float, NP1D_int
from .ops import group_indices

logger = logging.getLogger(__name__)


@overload
def _var(
    x: npt.ArrayLike,
    axis: None = None,
    ddof: int = 0,
    mean: int | npt.ArrayLike | None = None,
) -> int:
    ...


@overload
def _var(
    x: npt.ArrayLike,
    axis: int,
    ddof: int = 0,
    mean: int | npt.ArrayLike | None = None,
) -> np.ndarray:
    ...


def _var(x, axis=None, ddof=0, mean=None):
    """Computes variance of a given array.

    Parameters
    __________
    x: array-like
        The data matrix
    axis: int
        Axis along which to compute variance.
    ddof: int
        Degrees of freedom.
    mean: int, array-like
        Precomputed mean.

    Examples
    ________
    >>> _var([1, 2, 1, 4])
    1.5
    >>> a = np.array([0, 0, 2, 0, 4])
    >>> asp = sp.csr_matrix(a)
    >>> np.abs(_var(asp) - 2.56) <= 1e-10
    True
    >>> astack = np.array([a, a])
    >>> astack.shape
    (2, 5)
    >>> aspstack = sp.csr_matrix(astack)
    >>> v = _var(aspstack, axis=1)
    >>> v.shape
    (2,)
    >>> np.abs(v[0] - 2.56) <= 1e-10
    True
    >>> np.abs(v[1] - 2.56) <= 1e-10
    True
    >>> vd1 = _var(aspstack, axis=1, ddof=1)
    >>> np.abs(vd1[1] - 3.2) <= 1e-10
    True
    >>> _var([1])
    0.0
    """
    if not isinstance(x, (np.ndarray, sp.spmatrix)):
        x = np.asarray(x)

    if isinstance(x, sp.spmatrix):
        Ex = np.ravel(x.mean(axis=axis)) if mean is None else np.ravel(mean)
        Ex2 = np.ravel(x.power(2).mean(axis=axis))
        num = Ex2 - np.square(Ex)
        if axis is None:
            num = num.item()
        n = x.shape[axis] if axis is not None else x.size
        var = num * np.divide(n, n - ddof)
    else:
        var = np.var(x, axis=axis, ddof=ddof)

    return var


@overload
def mean_var(
    x: npt.ArrayLike,
    axis: None = None,
    ddof: int = 0,
) -> Tuple[float, float]:
    ...


@overload
def mean_var(
    x: npt.ArrayLike,
    axis: int,
    ddof: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    ...


def mean_var(x, axis=None, ddof=0):
    """Returns both mean and variance.

    Parameters
    __________
    x: array-like
        The data matrix
    axis: int
        Axis along which to compute variance.
    ddof: int
        Degrees of freedom.

    Returns
    _______
    mean: array-like, int
        The mean of the data along axis, or total mean
        if axis is None.
    var: array-like, int
        The variance of the data along axis, or total mean
        if axis is None.
    """
    if not isinstance(x, (np.ndarray, sp.spmatrix)):
        x = np.asarray(x)
    mean = np.ravel(x.mean(axis=axis))
    return mean, _var(x, axis=axis, ddof=ddof, mean=mean)


def ttest(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    axis: int | None = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Performs a Welch's t-test (unequal sample sizes, unequal vars).
    Extends scipy's ttest_ind to support sparse matrices.

    Parameters
    __________
    a, b: array-like
        Must have the same shape, except possibly along `axis`.
    axis: int
        Axis along with to perform ttest.

    Examples
    ________
    >>> from numpy.testing import assert_almost_equal as aae
    >>> from scipy.stats import ttest_ind
    >>> a = np.random.randint(0, 3, (5, 8))
    >>> b = np.random.randint(0, 3, (3, 8))
    >>> tstat, pval = ttest_ind(a, b, equal_var=False)
    >>> tstat_this, pval_this = ttest(a, b)
    >>> aae(tstat, tstat_this)
    >>> aae(pval, pval_this)
    >>> asp = sp.csr_matrix(a)
    >>> bsp = sp.csr_matrix(b)
    >>> tstat_sp, pval_sp = ttest(asp, bsp)
    >>> aae(tstat, tstat_sp)
    >>> aae(pval, pval_sp)
    """
    if not isinstance(a, (np.ndarray, sp.spmatrix)):
        a = np.asarray(a)
    if not isinstance(b, (np.ndarray, sp.spmatrix)):
        b = np.asarray(b)

    n1 = a.shape[axis] if axis is not None else a.size
    n2 = b.shape[axis] if axis is not None else b.size
    if n1 == 0 or n2 == 0:
        raise ValueError("Found zero-length arrays.")

    m1, v1 = mean_var(a, axis=axis, ddof=1)
    m2, v2 = mean_var(b, axis=axis, ddof=1)
    if isinstance(m1, np.ndarray):
        assert m1.shape == v1.shape == m2.shape == v2.shape  # type: ignore

    return ttest_from_mean_var(n1, m1, v1, n2, m2, v2)


def ttest_from_mean_var(
    n1: int,
    m1: NP1D_float | float,
    v1: NP1D_float | float,
    n2: int,
    m2: NP1D_float | float,
    v2: NP1D_float | float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Performs a two sided t_test given means and variances.
    n1, m1, v1: number of points, mean, and variance respectively.
    """
    df, denom = _unequal_var_ttest_denom(v1, n1, v2, n2)
    res = _ttest_ind_from_stats(m1, m2, denom, df, alternative="two-sided")
    return Ttest_indResult(*res)


@wraps(multipletests)
def _correct(pvals, method='fdr_bh'):
    """Simple wrapper for multipletests."""
    return multipletests(
        pvals=pvals,
        alpha=0.05,
        method=method,
        is_sorted=False,
        returnsorted=False,
    )


def _compute_log2fc(mean1, mean2, base='e', is_logged=False):
    """Computes log2 fold change and converts base if data is already logged."""
    if is_logged:
        log2fc = mean1 - mean2
        # Convert base
        if base is not None and base != 2:
            base = np.e if base == 'e' else float(base)
            log2fc *= np.log2(base)
    else:
        log2fc = np.log2((mean1 + 1) / (mean2 + 1))
    return log2fc


def fit_nbinom(rvs) -> Tuple[float, float]:
    """Fit a negative binomial distribution to the data. Returns (n, p).
    """
    s = np.ones_like(rvs.astype(float))
    nb = NegativeBinomial(rvs, s).fit()

    mu = np.exp(nb.params[0])
    p = 1 / (1 + mu * nb.params[1])
    n = mu * p / (1-p)

    return n, p


def stats1d(
    rvs,
    dist: str,
    *,
    params: Any | None = None,
    alpha: float = 0.05,
    pprint: bool = False,
) -> dict[str, float]:
    """Computes main statistics for a 1D distribution assumed to follow
    `dist`. `dist` should be a string pointing to a scipy distribution.

    Parameters
    __________
    rvs: array-like
        Samples from the distribution.
    dist: str
        A continuous distribution from scipy.stats.
    params: any
        If None, will call sc_dist.fit.
    alpha: float
        The outlier threshold to use for computing inverse quantiles.
    pprint: bool
        If True, will print the dictionary of stats.
    """
    assert alpha > 0 and alpha < 1
    sc_dist: rv_continuous = getattr(scs, dist)
    if params is None:
        params = sc_dist.fit(rvs)
    mean, var = sc_dist.stats(*params)
    stats = {'dist': dist,
             'dist_mean': mean, 'dist_std': var ** (1/2),
             'dist_q05': sc_dist.ppf(alpha, *params),
             'dist_q95': sc_dist.ppf(1 - alpha, *params),
             'min': rvs.min(), 'max': rvs.max(),
             'data_mean': rvs.mean(), 'data_std': rvs.std(),
             'data_q05': np.quantile(rvs, 0.05),
             'data_q95': np.quantile(rvs, 0.95)}
    if pprint:
        logger.info(f"'{dist}' distribution statistics")
        logger.info(pretty_repr(stats))
    return stats


@dataclass
class _MeanVarVector:
    n: int  # number of samples accumulated so far
    sums: NP1D_float  # sums of samples
    sum_of_squares: NP1D_float  # sums of samples squared

    def __post_init__(self):
        if len(self.sums) != len(self.sum_of_squares):
            raise ValueError(
                "Dimensions of sums and sum_of_squares "
                f"have to be the same, but found dims {len(self.sums)} "
                f"and {len(self.sum_of_squares)}."
            )
        self.dim: int = len(self.sums)

    def _check_dims(self, other):
        if self.dim != other.dim:
            raise ValueError(
                "Can only add vectors of the same dimension, "
                f"but found dimensions {self.dim} and {other.dim}."
            )

    def __add__(self, other: '_MeanVarVector'):
        self._check_dims(other)
        n = self.n + other.n
        sums = self.sums + other.sums
        sum_of_squares = self.sum_of_squares + other.sum_of_squares
        return _MeanVarVector(n, sums, sum_of_squares)

    def __iadd__(self, other: '_MeanVarVector'):
        self._check_dims(other)
        self.n += other.n
        self.sums += other.sums
        self.sum_of_squares += other.sum_of_squares
        return self

    def get_mean(self) -> NP1D_float:
        return self.sums / self.n

    def get_var(self, mean: NP1D_float | None = None, ddof: int = 0) -> NP1D_float:
        if mean is None:
            mean = self.get_mean()
        Ex2 = self.sum_of_squares / self.n
        var = (Ex2 - np.square(mean)) * np.divide(self.n, self.n - ddof)
        return var


class PartMeanVar:
    """A class for efficiently computing means and variances of partitions
    of a data matrix and unions of partitions. E.g., given a matrix of
    shape (m, n) row-partitioned into three submatrices A, B, and C, we can
    efficiently compute mean of [A, B], or [A, C] by caching sums of
    entries over each partition. This can be useful for DE analysis
    performed in a one-vs-all fashion for multiple groups.
    """

    def __init__(self, X, y: NP1D_int, show_progress_bar: bool = True):
        X, = indexable(X)
        if X.ndim != 2:
            raise ValueError("PartMeanVar currently only supports 2D arrays.")
        check_consistent_length(X, y)

        unq_labels, groups = group_indices(y)
        if show_progress_bar:
            unq_labels = tqdm(unq_labels, desc='Fitting data')  # type: ignore

        # support sparse matrices as well
        def square_func(x): return x.power(2) if sp.issparse(X) else x**2
        # maps label to a StatVector
        self.sum_vectors: Dict[Hashable, _MeanVarVector] = {}

        for label, group in zip(unq_labels, groups):
            xg = X[group]
            self.sum_vectors[label] = _MeanVarVector(
                n=xg.shape[0],
                sums=np.ravel(xg.sum(axis=0)),
                sum_of_squares=np.ravel(square_func(xg).sum(axis=0)),
            )

    def compute(
        self,
        labels: List[Hashable],
        ddof: int = 0,
        exclude: bool = False
    ) -> Tuple[int, NP1D_float, NP1D_float]:
        """Computes mean and variance of the submatrix defined by labels.
        If exclude is True, will compute the mean and variance of the
        submatrix that is formed by removing labels.
        """
        if exclude:
            labels = list(set(self.sum_vectors).difference(labels))
        if len(labels) == 0:
            raise ValueError("No labels found.")
        if len(diff := set(labels).difference(self.sum_vectors)) != 0:
            raise ValueError(f"Found labels {diff} not in dictionary.")

        accumul = _MeanVarVector(
            n=0,
            sums=np.zeros_like(self.sum_vectors[labels[0]].sums, dtype=float),
            sum_of_squares=np.zeros_like(self.sum_vectors[labels[0]].sum_of_squares, dtype=float)
        )

        for label in labels:
            accumul += self.sum_vectors[label]

        if accumul.n != 1 and ddof >= accumul.n:
            raise ValueError(f"Degrees of freedom are greater than n={accumul.n}.")
        if accumul.n == 1:
            logger.warning(
                "Found group with only 1 datapoint. Test results may not "
                "be reliable. Setting ddof=0."
            )
            ddof = 0

        mean = accumul.get_mean()
        var = accumul.get_var(mean=mean, ddof=ddof)
        return accumul.n, mean, var
