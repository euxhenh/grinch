import numpy as np
import pytest
import scipy.sparse as sp
from grinch.processors.wrappers import Leiden

from ._utils import assert_allclose


X = np.array([
    [0, 64, 1, 3],
    [47, 0, 0.5, 2],
    [1, 2, 0, 65],
    [0.5, 6, 67, 0]
], dtype=float)

G = sp.csr_matrix(X)


@pytest.mark.parametrize("X", [G])
def test_leiden_weighted(X):
    leiden = Leiden()
    pred = leiden.fit_predict(X)
    ans = [0, 0, 1, 1]
    assert_allclose(ans, pred)


@pytest.mark.parametrize("X", [G])
def test_leiden_unweighted(X):
    leiden = Leiden(weighted=False)
    pred = leiden.fit_predict(X)
    ans = [0, 0, 0, 0]
    assert_allclose(ans, pred)
