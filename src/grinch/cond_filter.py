import logging
from typing import Any, Generic, Literal, TypeVar, overload

import numpy as np
from pydantic import BaseModel, NonNegativeInt, model_validator, validate_call
from sklearn.utils import column_or_1d

from .custom_types import NP1D_bool, NP1D_int, PercentFraction
from .utils.validation import any_not_None

logger = logging.getLogger(__name__)

T = TypeVar("T", int, float, bool, str)


class Filter(BaseModel, Generic[T]):
    """Selects and returns item indices based on criteria.

    Takes any object and looks for 'key' in its members. It then selects
    indices from 'key' based on the conditions defined in this class. If
    cutoff is not None, will take all values greater than or less than
    'cutoff'. If top_k is not None, will take the top k greatest (smallest)
    elements. If both are None, will assume key is a mask and convert it to
    a bool. The ordered key is useful if the returned indices can be in any
    order or if they should be ordered. 'greater_is_True' will reverse the
    selection criteria, except for the case when 'cutoff' and 'top_k' are
    both None, where it has no effect.

    If key is None, will assume the passed object to call is the array to
    filter itself.

    To take a mask of True or False, simply use gt=False or lt=True.

    Parameters
    ----------
    key: str
        If not None, will search in obj for a member named as `key`.
    ge, le, gt, lt: T
        Greater than or less than in either strict or non-strict mode.
    top_k, bot_k: int
        Top or bottom k items to pick.
    top_ratio, bot_ratio: float
        A percent fraction betwen 0 and 1. Will round up to the nearest
        item.

    Examples
    --------
    >>> f1 = Filter(gt=3)
    >>> f1([1, 2, 3, 4, 5, 6], as_mask=True)
    array([False, False, False,  True,  True,  True])
    >>> f1([5, 4, 6, 3, 2], as_mask=False)
    array([0, 1, 2])

    >>> f2 = Filter(top_k=2)
    >>> f2([7, 1, 2, 5, 6, 8], as_mask=False)
    array([5, 0])

    >>> f3 = Filter(bot_ratio=0.4)
    >>> f3([1, 7, 5, 3, 4], as_mask=False)
    array([0, 3])

    >>> f = f1 & f2  # Take greater than 3, but no more than 2 elements
    >>> f([2, 4, 3, 5, 6, 0, 1, 7], as_mask=False)
    array([4, 7])
    """
    model_config = {
        'validate_assignment': True,
        'validate_default': True,
        'extra': 'forbid',
    }

    key: str | None = None  # Set to None if passing a container

    ge: T | None = None  # greater than or equal
    le: T | None = None  # less than or equal
    gt: T | None = None  # greater than
    lt: T | None = None  # less than

    top_k: NonNegativeInt | None = None  # top k items after sorting
    bot_k: NonNegativeInt | None = None  # bottom k items after sorting
    # These will be rounded up to the nearest item
    top_ratio: PercentFraction | None = None  # top fraction of items
    bot_ratio: PercentFraction | None = None  # bottom fraction of items

    @model_validator(mode='before')
    def at_most_one_not_None(cls, data):
        to_check = ['ge', 'le', 'gt', 'lt', 'top_k', 'bot_k', 'top_ratio', 'bot_ratio']
        if sum(data.get(key, None) is not None for key in to_check) > 1:
            raise ValueError(
                "At most one filter key should not be None. If more than "
                "one key is desired, then stack multiple filters together."
            )
        return data

    def __and__(self, other) -> 'StackedFilter':
        return StackedFilter(self, other)

    @staticmethod
    def _take_k_functional(arr, k: NonNegativeInt, as_mask: bool, top: bool):
        if k > (n := len(arr)):
            logger.warning(f"Requested {k} items but array has size {n}.")
        argidx = np.argsort(arr)
        # Flip so that we start with greatest
        idx = np.flip(argidx[-k:]) if top else argidx[:k]

        if not as_mask:
            return idx

        mask = np.full_like(arr, False, dtype=bool)
        mask[idx] = True
        return mask

    def _take_k(self, arr, as_mask: bool = True):
        """Take top or bot k items.
        """
        top = self.top_k is not None
        k = self.top_k if top else self.bot_k
        assert k is not None
        return self._take_k_functional(arr, k, as_mask, top)

    def _take_ratio(self, arr, as_mask: bool = True):
        """Take top or bot fraction of items.
        """
        top = self.top_ratio is not None
        ratio = self.top_ratio if top else self.bot_ratio
        assert ratio is not None
        k = int(np.ceil(ratio * len(arr)))  # round up
        return self._take_k_functional(arr, k, as_mask, top)

    def _take_cutoff(self, arr, as_mask: bool = True):
        """Takes the elements which are greater than or less than cutoff.
        """
        assert any_not_None(self.gt, self.ge, self.lt, self.le)
        top = any_not_None(self.gt, self.ge)
        strict = any_not_None(self.gt, self.lt)

        match top, strict:
            case True, True:
                mask = arr > self.gt
            case True, False:
                mask = arr >= self.ge
            case False, True:
                mask = arr < self.lt
            case False, False:
                mask = arr <= self.le

        return mask if as_mask else np.argwhere(mask).ravel()

    @staticmethod
    def _get_repr(obj: Any, key: str) -> Any:
        """Recursively get members. Check if they can be obtained as
        items or attributes.
        """
        for read_key in key.split('.'):
            if hasattr(obj, read_key):
                obj = getattr(obj, read_key)
            elif read_key in obj:
                obj = obj[read_key]
            else:
                raise KeyError(f"Could not find '{read_key}' in object of type {type(obj)}.")
        return obj

    @overload
    def __call__(self, obj: Any, as_mask: Literal[True]) -> NP1D_bool: ...

    @overload
    def __call__(self, obj: Any, as_mask: Literal[False]) -> NP1D_int: ...

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __call__(self, obj, as_mask=True):
        """Applies filtering conditions and returns a mask or index array.
        """
        if self.key is not None:
            obj = self._get_repr(obj, self.key)

        arr: np.ndarray[T, Any] = column_or_1d(obj)

        if any_not_None(self.ge, self.gt, self.le, self.lt):
            return self._take_cutoff(arr, as_mask)
        if any_not_None(self.top_k, self.bot_k):
            return self._take_k(arr, as_mask)
        if any_not_None(self.top_ratio, self.bot_ratio):
            return self._take_ratio(arr, as_mask)

        # Default to taking True
        return arr.astype(bool) if as_mask else np.argwhere(arr).ravel()


class StackedFilter:
    """A convenience class for stacking multiple Filter's together.
    """

    def __init__(self, *fcs):
        self.fcs = []
        for fc in fcs:
            if isinstance(fc, Filter):
                self.fcs.append(fc)
            elif isinstance(fc, StackedFilter):
                self.fcs.extend(fc.fcs)
            else:
                raise TypeError(f"Cannot stack object of type {type(fc)}.")

    def __len__(self):
        return len(self.fcs)

    def __and__(self, other) -> 'StackedFilter':
        return StackedFilter(self, other)

    @overload
    def __call__(self, obj: Any, as_mask: Literal[True]) -> NP1D_bool: ...

    @overload
    def __call__(self, obj: Any, as_mask: Literal[False]) -> NP1D_int: ...

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __call__(self, obj, as_mask=True):
        # Get first mask
        mask = self.fcs[0](obj, as_mask=True)
        # Get any remaining masks
        for fc in self.fcs[1:]:
            mask &= fc(obj, as_mask=True)
        return mask if as_mask else np.argwhere(mask).ravel()
