from itertools import chain, combinations

import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_allclose as _assert_allclose


def assert_allclose(x1, x2, **kwargs):
    """Check if assert_allclose(x1, x2) and convert to numpy if sparse."""
    if sp.issparse(x1):
        x1 = x1.toarray()
    if sp.issparse(x2):
        x2 = x2.toarray()
    _assert_allclose(x1, x2, **kwargs)


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def to_view(x):
    X_mod = np.insert(x, [1, 2], 42, axis=1)
    X_mod = np.insert(X_mod, [1, 2], 42, axis=0)
    original_shape = X_mod.shape
    X_mod = np.ravel(X_mod)
    X_mod = X_mod.reshape(original_shape)[
        np.delete(np.arange(original_shape[0]), [1, 3])
    ][:, np.delete(np.arange(original_shape[1]), [1, 3])]
    assert X_mod.base is not None
    return X_mod
