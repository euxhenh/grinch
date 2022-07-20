from functools import reduce, wraps
from itertools import chain, combinations, product
from typing import Callable

import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_allclose as _assert_allclose


def assert_allclose(x1, x2):
    """Check if assert_allclose(x1, x2) and convert to numpy if sparse."""
    if sp.issparse(x1):
        x1 = x1.toarray()
    if sp.issparse(x2):
        x2 = x2.toarray()
    _assert_allclose(x1, x2)


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def compose(functions) -> Callable:
    """Composes functions from right to left. I.e., functions on the right
    will run after those on the left.
    """
    identity = lambda x, *args, **kwargs: x  # noqa: E731
    if len(functions) == 0:
        return identity
    return reduce(lambda f, g: lambda x: g(f(x)), functions, identity)


def parametrize(param_list, to_apply=None):
    """Applies all combinations of params and functions in to_apply.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self):
            for params in param_list:
                if to_apply is None:
                    func(self, params)
                    return

                try:
                    apply_iter = list(iter(to_apply))
                except TypeError:
                    apply_iter = [to_apply]

                for params, function in product(param_list, apply_iter):
                    params = function(params)
                    with self.subTest(params):
                        func(self, params)

        return wrapper
    return decorator


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