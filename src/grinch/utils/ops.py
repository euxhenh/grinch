from typing import List

import numpy as np
from sklearn.utils import column_or_1d


def true_inside(x, v1, v2) -> np.ndarray:
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
    >>> true_inside(np.matrix([[1], [2], [3]]), None, 2)
    array([ True,  True, False])
    """
    x = column_or_1d(x)
    if v1 is None:
        v1 = -np.inf
    if v2 is None:
        v2 = np.inf

    return (v1 <= x) & (x <= v2)


def group_indices(x) -> List[np.ndarray]:
    """Returns an index array pointing to unique elements in x.

    Parameters
    __________
    x: array-like

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
    """
    x = column_or_1d(x)
    if len(x) == 0:
        raise ValueError("Encountered 0-sized array.")
    argidx = np.argsort(x)
    sorted_x = np.asarray(x)[argidx]
    # Make sure this is a copy and not a view of x
    if sorted_x.base is not None:
        sorted_x = sorted_x.copy()

    unique_items, first_indices = np.unique(sorted_x, return_index=True)
    groups = np.split(argidx, first_indices[1:])
    assert len(unique_items) == len(groups)
    return unique_items, groups
