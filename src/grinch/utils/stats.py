from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.stats._stats_py import (
    Ttest_indResult,
    _ttest_ind_from_stats,
    _unequal_var_ttest_denom,
)


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

    df, denom = _unequal_var_ttest_denom(v1, n1, v2, n2)
    res = _ttest_ind_from_stats(m1, m2, denom, df, alternative="two-sided")
    return Ttest_indResult(*res)
