from __future__ import annotations

import logging
from collections import UserList
from functools import cached_property
from typing import Any, Generic, List, Literal, TypeVar, overload

import numpy as np
from pydantic import BaseModel, NonNegativeInt, model_validator, validate_call
from sklearn.utils import column_or_1d

from .custom_types import NP1D_bool, NP1D_int, PercentFraction
from .utils.validation import any_not_None

logger = logging.getLogger(__name__)

T = TypeVar("T", int, float, bool, str)


class Filter(BaseModel, Generic[T]):
    """Selects and returns item indices based on criteria.

    If key is None, will assume the passed object to call is the array to
    filter itself.

    To take a mask of True or False, simply use gt=False or lt=True.

    Parameters
    ----------
    key : str
        If not None, will search in obj for a member named as `key`.

    ge, le, gt, lt : T
        Greater than or less than in either strict or non-strict mode.

    top_k, bot_k : int
        Top or bottom k items to pick.

    top_ratio, bot_ratio : float
        A percent fraction betwen 0 and 1. Will round up to the nearest
        item.

    absolute : bool
        If True, will consider the absolute value of `key`.

    Examples
    --------
    >>> f1 = Filter(gt=3)
    >>> f1([1, 2, 3, 4, 5, 6], as_mask=True)
    array([False, False, False,  True,  True,  True])
    >>> f1([1, 2, 3, 4, 5, 6], as_mask=False)
    array([3, 4, 5])

    >>> f2 = Filter(top_k=2)
    >>> f2([7, 1, 2, 5, 6, 8], as_mask=False)
    array([5, 0])

    >>> f3 = Filter(bot_ratio=0.4)
    >>> f3([1, 7, 5, 3, 4], as_mask=False)
    array([0, 3])

    >>> f = f1 & f2  # Take greater than 3, but no more than 2 elements
    >>> f([2, 4, 3, 5, 6, 0, 1, 7], as_mask=False)
    array([4, 7])

    >>> r = f & f3  # Can also stack StackedFilter and Filter
    >>> r([3, 4, 5, 6, 7], as_mask=True)
    array([False, False, False, False, False])

    >>> fabs = Filter(ge=2, absolute=True)
    >>> fabs([-5, -6, -1, 0, 1, 2], as_mask=False)
    array([0, 1, 5])
    """
    __conditions__ = ['ge', 'le', 'gt', 'lt',
                      'equal', 'not_equal',
                      'top_k', 'bot_k',
                      'top_ratio', 'bot_ratio']

    model_config = {
        'validate_assignment': True,
        'validate_default': True,
        'extra': 'forbid',
        'frozen': True,
    }

    key: str | None = None  # Set to None if passing a container

    ge: T | None = None  # greater than or equal
    le: T | None = None  # less than or equal
    gt: T | None = None  # greater than
    lt: T | None = None  # less than

    equal: T | None = None  # exactly equal to
    not_equal: T | None = None  # not equal to

    top_k: NonNegativeInt | None = None  # top k items after sorting
    bot_k: NonNegativeInt | None = None  # bottom k items after sorting
    # These will be rounded up to the nearest item
    top_ratio: PercentFraction | None = None  # top fraction of items
    bot_ratio: PercentFraction | None = None  # bottom fraction of items

    absolute: bool = False

    @model_validator(mode='before')
    def at_most_one_not_None(cls, data):
        """Ensure that at most one condition is set. If no conditions are
        set, will treat the input as a boolean.
        """
        if sum(data.get(key, None) is not None for key in cls.__conditions__) > 1:
            raise ValueError(
                "At most one filter key should not be None. If more than "
                "one key is desired, then stack multiple filters together."
            )
        return data

    def __and__(self, other) -> StackedFilter:
        """If f1 & f2, return a StackedFilter that performs element-wise &.
        """
        return StackedFilter(self, other)

    def __repr_args__(self):
        """Override method to exclude None fields.
        """
        for k, v in super().__repr_args__():
            if v is None:
                continue
            yield k, v

    @cached_property
    def is_top(self) -> bool:
        """Determine if we are taking top or bot elements.

        Returns: True if we are selecting greatest elements.
        """
        return any_not_None(self.ge, self.gt, self.top_k, self.top_ratio)

    @staticmethod
    def _take_k_functional(arr, k: NonNegativeInt, as_mask: bool, top: bool):
        r"""Functional variant of take top (bot) k items.

        If n := len(arr), then this runs in $O(n + k\log{k})$ time.

        Parameters
        ----------
        arr: 1D numpy array
            Must be sortable.
        k: int
            The number of elements to take.
        as_mask: bool
            If False, will return indices, otherwise a boolean mask.
        top: bool
            If True, will return `k` greatest elements, else smallest.
        """
        if k > (n := len(arr)):
            logger.warning(f"Requested {k} items but array has size {n}.")
            k = n

        idx = np.argpartition(arr, -k if top else k)  # linear time
        idx = idx[-k:] if top else idx[:k]
        idx = idx[np.argsort(arr[idx])]  # sort selected indices
        if top:
            idx = np.flip(idx)

        if not as_mask:
            return idx

        mask = np.full_like(arr, False, dtype=bool)
        mask[idx] = True
        return mask

    def _take_k(self, arr, as_mask: bool = True):
        """Take top or bot k items.
        """
        k = self.top_k if self.is_top else self.bot_k
        assert k is not None
        return self._take_k_functional(arr, k, as_mask, self.is_top)

    def _take_ratio(self, arr, as_mask: bool = True):
        """Take top or bot fraction of items.
        """
        ratio = self.top_ratio if self.is_top else self.bot_ratio
        assert ratio is not None
        k = int(np.ceil(ratio * len(arr)))  # round up
        return self._take_k_functional(arr, k, as_mask, self.is_top)

    def _take_equal(self, arr, as_mask: bool = True):
        """Take elements exactly equal to `self.cfg.equal`.
        """
        if self.equal is not None:
            mask = arr == self.equal
        elif self.not_equal is not None:
            mask = arr != self.not_equal
        return mask if as_mask else arr[mask]

    def _take_cutoff(self, arr, as_mask: bool = True):
        """Takes the elements which are greater than or less than cutoff.
        """
        strict = any_not_None(self.gt, self.lt)

        match self.is_top, strict:
            case True, True:
                mask = arr > self.gt
            case True, False:
                mask = arr >= self.ge
            case False, True:
                mask = arr < self.lt
            case False, False:
                mask = arr <= self.le
            case _:
                raise ValueError("Internal error. Found non bool values.")

        return mask if as_mask else np.argwhere(mask).ravel()

    @staticmethod
    def _get_member(obj: Any, key: str) -> Any:
        """Recursively get members. Check if they can be obtained as
        items or attributes.
        """
        for read_key in key.split('.'):
            if hasattr(obj, read_key):
                obj = getattr(obj, read_key)
            elif read_key in obj:
                obj = obj[read_key]
            else:
                raise KeyError(
                    f"Could not find '{read_key}' in object of type {type(obj)}."
                )
        return obj

    @overload
    def __call__(self, obj: Any, as_mask: Literal[True]) -> NP1D_bool: ...

    @overload
    def __call__(self, obj: Any, as_mask: Literal[False]) -> NP1D_int: ...

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __call__(self, obj, as_mask=True):
        """Applies filtering conditions and returns a mask or index array.

        Parameters
        ----------
        obj: If `self.key` is set, then this can be any object which has an
            attribute or member named `self.key`. Otherwise, needs to be
            array-like.
        as_mask: bool
            If True, will return a 1D boolean array where True means the
            item has been selected. Otherwise, will return the indices as a
            1D integer array.

        Returns
        -------
        mask: 1D bool or int numpy array.
        """
        if self.key is not None:
            obj = self._get_member(obj, self.key)

        arr: np.ndarray[T, Any] = column_or_1d(obj)
        if self.absolute:
            arr = np.abs(arr)

        if any_not_None(self.ge, self.gt, self.le, self.lt):
            return self._take_cutoff(arr, as_mask)
        if any_not_None(self.equal, self.not_equal):
            return self._take_equal(arr, as_mask)
        if any_not_None(self.top_k, self.bot_k):
            return self._take_k(arr, as_mask)
        if any_not_None(self.top_ratio, self.bot_ratio):
            return self._take_ratio(arr, as_mask)

        # Default to taking True
        return arr.astype(bool) if as_mask else np.argwhere(arr).ravel()


class StackedFilter(UserList):
    """A convenience class for stacking multiple Filter's together. Behaves
    like a Pythonic list.

    Will perform element-wise &. This supports different combinations of
    filters where `key` can be specified in some, but not all.

    Parameters
    ----------
    *filters: iterable
        An iterable of Filter's or StackedFilter's.
    """

    def __init__(self, *filters: Filter | StackedFilter):
        __filters__: List[Filter] = []

        for fc in filters:
            if isinstance(fc, Filter):
                __filters__.append(fc)
            elif isinstance(fc, StackedFilter):
                __filters__.extend(fc)
            else:
                raise TypeError(f"Cannot stack object of type {type(fc)}.")

        super().__init__(__filters__)

    def __and__(self, other) -> StackedFilter:
        return StackedFilter(self, other)

    @overload
    def __call__(self, obj: Any, as_mask: Literal[True]) -> NP1D_bool: ...

    @overload
    def __call__(self, obj: Any, as_mask: Literal[False]) -> NP1D_int: ...

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __call__(self, obj, as_mask=True):
        """Applies all filters and returns a mask or index array.

        Parameters
        ----------
        obj: If `filter.key` is set, then this can be any object which has an
            attribute or member named `self.key`. Otherwise, needs to be
            array-like.
        as_mask: bool
            If True, will return a 1D boolean array where True means the
            item has been selected. Otherwise, will return the indices as a
            1D integer array.

        Returns
        -------
        mask: 1D bool or int numpy array.
        """
        if len(self) == 0:
            raise ValueError("Called empty `StackedFilter`.")

        # Get first mask
        mask = self.data[0](obj, as_mask=True)
        # Get remaining masks
        for fc in self.data[1:]:
            mask &= fc(obj, as_mask=True)
        return mask if as_mask else np.argwhere(mask).ravel()
