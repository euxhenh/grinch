import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_almost_equal

from grinch.utils.stats import PartMeanVar, mean_var

from ._utils import assert_allclose, to_view

X = np.array([
    [4, 2, 0, 1, 0],
    [8, 2, 0, 6, 7],
    [6, 2, 5, 3, 0],
    [4, 2, 7, 3, 3],
    [0, 2, 8, 9, 7],
    [5, 2, 0, 5, 2],
    [6, 2, 0, 5, 8]
])

labels = [0, 0, 1, 1, 1, 2, 2]

X_mods = [X, sp.csr_matrix(X), to_view(X), X + 0.3]


@pytest.mark.parametrize("X", X_mods)
def test_partmeanvar(X):
    pmv = PartMeanVar(X, labels)
    _, mean, var = pmv.compute([0])
    m, v = mean_var(X[[0, 1]], axis=0)
    assert_allclose(mean, m)
    assert_allclose(var, v)

    _, mean, var = pmv.compute([1], ddof=1)
    m, v = mean_var(X[[2, 3, 4]], axis=0, ddof=1)
    assert_allclose(mean, m)
    assert_allclose(var, v)
    _, mean, var = pmv.compute([0, 2], ddof=1, exclude=True)
    assert_allclose(mean, m)
    assert_allclose(var, v)

    _, mean, var = pmv.compute([1, 2], ddof=1)
    m, v = mean_var(X[[2, 3, 4, 5, 6]], axis=0, ddof=1)
    assert_allclose(mean, m)
    assert_allclose(var, v)
    _, mean, var = pmv.compute([0], ddof=1, exclude=True)
    assert_allclose(mean, m)
    assert_allclose(var, v)

    _, mean, var = pmv.compute([0, 1, 2], ddof=1)
    m, v = mean_var(X, axis=0, ddof=1)
    assert_allclose(mean, m)
    assert_almost_equal(var, v)
