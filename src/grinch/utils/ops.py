from functools import reduce
from typing import List, Literal, Tuple, overload

import numpy as np
from scipy.sparse import issparse, spmatrix
from sklearn.utils import column_or_1d

from ..custom_types import (
    NP1D_Any,
    NP1D_bool,
    NP1D_int,
    NP2D_float,
    NP2D_int,
    NP_bool,
)


def IDENTITY(x):
    """Identity function."""
    return x


def compose(*funcs):
    """Composes functions left to right, i.e., the first function on the
    list will be applied first.

    Examples
    ________
    >>> f = compose(sum, lambda x: x + 10)
    >>> f([1, 2, 3])
    16
    """
    composed = reduce(lambda f, g: lambda x: g(f(x)), funcs, IDENTITY)
    return composed


def safe_format(prefix, **kwargs):
    """List Str.format() but will not raise an error if it does not find
    the key in the string.

    Examples
    ________
    >>> safe_format("hello", somevar="someval")
    'hello'
    >>> safe_format("hello_{somevar}", somevar="someval")
    'hello_someval'
    """
    for k, v in kwargs.items():
        if f"{{{k}}}" in prefix:  # { is escaped by doubling it {{
            prefix = prefix.replace(f"{{{k}}}", str(v))
    return prefix


def true_inside(x, v1: float | None = None, v2: float | None = None) -> NP_bool:
    """Returns a boolean array a with a[i] = True if x[i] is between v1 and
    v2 (inclusive). If any of v1 or v2 is None, will cap at -np.inf and
    np.inf respectively.

    Parameters
    __________
    x: array-like
    v1, v2: float
        Lower and upper bounds. If None, will assume unbounded.

    Returns
    _______
    indices_to_keep: ndarray of shape (len(x),)
        A boolean array with its i-th element set to True if the i-th
        element of x falls in the range [v1, v2].

    Examples
    ________
    >>> true_inside([1, 2, 5, 3, 4, 3], 1, 3)
    array([ True,  True, False,  True, False,  True])
    >>> true_inside([2.5, 4, 3.2], None, 3.7)
    array([ True, False,  True])
    >>> true_inside([[2.5], [4], [3.2]], 2.8, None)
    array([False,  True,  True])
    >>> true_inside(np.array([1, 2, 3]), None, None)
    array([ True,  True,  True])
    """
    x = column_or_1d(x)
    if v1 is None:
        v1 = -np.inf
    if v2 is None:
        v2 = np.inf

    return (v1 <= x) & (x <= v2)


@overload
def group_indices(x, as_mask: Literal[False] = False) -> Tuple[NP1D_Any, List[NP1D_int]]: ...


@overload
def group_indices(x, as_mask: Literal[True]) -> Tuple[NP1D_Any, List[NP1D_bool]]: ...


def group_indices(x, as_mask=False):
    """Returns an index array pointing to unique elements in x.

    Parameters
    __________
    x: array-like
    as_mask: bool
        If True, will return masks where the indices point to the elements
        inside the group.

    Returns
    _______
    (unique_items, group_indices): (ndarray, List[ndarray])
        The first element contains an array of the unique elements and the
        second contains a list of arrays.

    Examples
    ________
    >>> group_indices([1, 4, 3, 2, 1, 2, 3, 3, 4])
    (array([1, 2, 3, 4]), [array([0, 4]), array([3, 5]), array([2, 6, 7]), array([1, 8])])
    >>> group_indices([1, 1, 1])
    (array([1]), [array([0, 1, 2])])
    >>> group_indices([1, 4, 2, 2, 1], as_mask=True)[1][0].astype(int)
    array([1, 0, 0, 0, 1])
    >>> group_indices([1, 4, 2, 2, 1], as_mask=True)[1][1].astype(int)
    array([0, 0, 1, 1, 0])
    >>> group_indices([1, 4, 2, 2, 1], as_mask=True)[1][2].astype(int)
    array([0, 1, 0, 0, 0])
    """
    x = column_or_1d(x)
    if x.size == 0:
        raise ValueError("Encountered 0-sized array.")
    argidx = np.argsort(x, kind='stable')
    sorted_x = x[argidx]
    unique_items, first_indices = np.unique(sorted_x, return_index=True)
    groups = np.split(argidx, first_indices[1:])
    assert len(unique_items) == len(groups)

    if as_mask:
        _groups = [np.zeros(len(x), dtype=bool) for _ in range(len(groups))]
        for group, _group in zip(groups, _groups):
            _group[group] = True
        groups = _groups

    return unique_items, groups


def order_by(x: NP1D_Any, y: NP1D_Any, unique_x: bool = False) -> NP1D_Any:
    """Orders the elements of 'x' so that they follow the order of the
    elements in y. It is assumed that x is a subset of y and that y has no
    duplicates.

    Parameters
    __________
    x: ndarray
        The array to order.
    y: ndarray
        The reference array. The relative order of the elements in x will
        follow the same relative order of elements in y.
    unique_x: bool
        If True, can speed up the calculations by assuming no duplicates in
        x.

    Examples
    ________
    >>> order_by(np.array([5, 6, 0, 8, 6]), np.array([0, 5, 3, 9, 8, 6]))
    array([0, 5, 8, 6, 6])
    >>> order_by(np.array([5, 6, 0, 8]), np.array([0, 5, 3, 9, 8, 6]), unique_x=True)
    array([0, 5, 8, 6])
    """
    # Fast common corner case
    if np.array_equal(x, y):
        return x

    restricted_y = y[np.in1d(y, x, assume_unique=unique_x)]
    if unique_x:
        return restricted_y

    unq_x, counts_x = np.unique(x, return_counts=True)
    unq_y, inv_y = np.unique(restricted_y, return_inverse=True)
    if not np.array_equal(unq_x, unq_y):
        raise ValueError("'x' is not a subset of 'y'.")

    return np.repeat(restricted_y, counts_x[inv_y])


def get_indices_dists_from_adj(adj: spmatrix) -> Tuple[NP2D_int, NP2D_float]:
    """Given a square adjacency matrix of shape (n_samples, n_samples),
    return two arrays of shape (n_samples, n_neighbors) corresponding
    to the indices and weights of those neighbors.

    Parameters
    __________
    adj: spmatrix
        Sparse square adjacency matrix.

    Examples
    ________
    >>> import scipy.sparse as sp
    >>> src = np.array([1, 1, 0, 0, 2, 2, 3, 3])
    >>> trg = np.array([0, 3, 1, 2, 0, 1, 2, 1])
    >>> data = np.array([0, 1.1, 5, 0, 3, 5, 3, 2])
    >>> adj = sp.csr_matrix((data, (src, trg)))
    >>> knn_indices, knn_dists = get_indices_dists_from_adj(adj)
    >>> knn_indices
    array([[1, 2],
           [0, 3],
           [0, 1],
           [1, 2]], dtype=int32)
    >>> knn_dists
    array([[5. , 0. ],
           [0. , 1.1],
           [3. , 5. ],
           [2. , 3. ]])
    """
    if not issparse(adj):
        raise ValueError("Adjacency matrix should be sparse.")
    adj = adj.tocoo()

    if adj.shape[0] != adj.shape[1]:
        raise ValueError("Expected square adjacency matrix.")

    n_neighbors = np.unique(np.unique(adj.row, return_counts=True)[1])
    if n_neighbors.size != 1:
        raise ValueError(
            "Found differing numbers of neighbors in the adjacency matrix."
        )
    n_neighbors = n_neighbors[0]
    n_samples = adj.shape[0]

    assert adj.row.size == n_samples * n_neighbors
    assert np.all(adj.row[:-1] <= adj.row[1:])  # make sure it's sorted

    knn_indices = adj.col.reshape((n_samples, n_neighbors))
    knn_dists = adj.data.reshape((n_samples, n_neighbors))

    return knn_indices, knn_dists
