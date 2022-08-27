from dataclasses import dataclass
from functools import wraps
from typing import Dict, Hashable, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.stats._stats_py import (
    Ttest_indResult,
    _ttest_ind_from_stats,
    _unequal_var_ttest_denom,
)
from sklearn.utils import check_consistent_length, indexable
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm

from ..custom_types import NP1D_float, NP1D_int
from .ops import group_indices


def _var(
    x: npt.ArrayLike,
    axis: Optional[int] = None,
    ddof: int = 0,
    mean: Optional[int | npt.ArrayLike] = None
) -> int | np.ndarray:
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


def mean_var(
    x: npt.ArrayLike,
    axis: Optional[int] = None,
    ddof: int = 0
) -> Tuple[int | np.ndarray, int | np.ndarray]:
    """Returns both mean and variance.

    Parameters
    __________
    x: array-like
        The data matrix
    axis: int
        Axis along which to compute variance.
    ddof: int
        Degrees of freedom.
    """
    if not isinstance(x, (np.ndarray, sp.spmatrix)):
        x = np.asarray(x)
    mean = np.ravel(x.mean(axis=axis))
    return mean, _var(x, axis=axis, ddof=ddof, mean=mean)


def ttest(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    axis: Optional[int] = 0
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


@dataclass
class _StatVector:
    n: int  # number of points accumulated so far
    sums: NP1D_float  # sum of vectors
    sum_of_squares: NP1D_float  # sum of vectors-squared

    def __add__(self, other: '_StatVector'):
        n = self.n + other.n
        sums = self.sums + other.sums
        sum_of_squares = self.sum_of_squares + other.sum_of_squares
        return _StatVector(n, sums, sum_of_squares)

    def __iadd__(self, other: '_StatVector'):
        self.n += other.n
        self.sums += other.sums
        self.sum_of_squares += other.sum_of_squares
        return self


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
            unq_labels = tqdm(unq_labels, desc='Fitting data')

        # support sparse matrices as well
        def square_func(x): return x.power(2) if sp.issparse(X) else x**2
        # maps label to a StatVector
        self.sum_vectors: Dict[Hashable, _StatVector] = {}

        for label, group in zip(unq_labels, groups):
            xg = X[group]
            self.sum_vectors[label] = _StatVector(
                n=xg.shape[0],
                sums=np.ravel(xg.sum(axis=0)),
                sum_of_squares=np.ravel(square_func(xg).sum(axis=0))
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

        accumul = _StatVector(
            n=0,
            sums=np.zeros_like(self.sum_vectors[labels[0]].sums, dtype=float),
            sum_of_squares=np.zeros_like(self.sum_vectors[labels[0]].sum_of_squares, dtype=float)
        )

        for label in labels:
            accumul += self.sum_vectors[label]

        if ddof >= accumul.n:
            raise ValueError(f"Degrees of freedom are greater than n={accumul.n}.")

        Ex = accumul.sums / accumul.n
        Ex2 = accumul.sum_of_squares / accumul.n
        var = (Ex2 - np.square(Ex)) * np.divide(accumul.n, accumul.n - ddof)

        return accumul.n, Ex, var
