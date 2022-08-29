import logging
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Extra, validate_arguments, validator
from sklearn.utils import column_or_1d

from .custom_types import NP1D_bool, NP1D_float, NP1D_int
from .utils.validation import all_not_None

logger = logging.getLogger(__name__)


class FilterCondition(BaseModel):
    """Takes any object and looks for 'key' in its members. It then selects
    indices from 'key' based on the conditions defined in this class. If
    cutoff is not None, will take all values greater than or less than
    'cutoff'. If top_k is not None, will take the top k greatest (smallest)
    elements. If both are None, will assume key is a mask and convert it to
    a bool. The ordered key is useful if the returned indices can be in any
    order or if they should be ordered. 'greater_is_better' will reverse
    the selection criteria, except for the case when 'cutoff' and 'top_k'
    are both None, where it has no effect.

    If key is None, will assume the passed object to call is the array to
    filter itself.
    """

    key: Optional[str]
    cutoff: Optional[float]
    top_k: Optional[int]
    greater_is_better: bool = False
    ordered: bool = False

    class Config:
        validate_assignment = True
        extra = Extra.forbid
        validate_all = True

    @validator('top_k')
    def _val_condition(cls, top_k, values):
        if all_not_None(top_k, values['cutoff']):
            raise ValueError("Only one or none of 'cutoff' or 'top_k' must be specified.")
        return top_k

    def __and__(self, other) -> 'StackedFilterCondition':
        return StackedFilterCondition(self, other)

    def _take_top_k(self, arr: NP1D_float, as_mask: bool = True) -> NP1D_int | NP1D_bool:
        """Takes the top k elements from arr and returns a mask or index
        array. If these elements need to be sorted, pass ordered=True.
        """
        if self.top_k is None:
            raise ValueError("Expected integer but 'top_k' is None.")
        if self.top_k > len(arr):
            raise ValueError(f"Requested {self.top_k} items but array has size {len(arr)}.")

        if self.greater_is_better:
            arr = -arr
        # argpartition is faster if we don't care about the order
        idx = np.argsort(arr) if self.ordered else np.argpartition(arr, self.top_k)
        idx = idx[:self.top_k]
        if not as_mask:
            return idx

        mask = np.full_like(arr, False, dtype=bool)
        mask[idx] = True
        return mask

    def _take_cutoff(self, arr: NP1D_float, as_mask: bool = True) -> NP1D_int | NP1D_bool:
        """Takes the elements which are greater than or less than cutoff
        depending on the value of greater_is_better.
        """
        if self.cutoff is None:
            raise ValueError("Expected float but 'cutoff' is None.")

        mask = arr >= self.cutoff if self.greater_is_better else arr <= self.cutoff
        if as_mask:
            return mask

        idx = np.argwhere(mask).ravel()
        if self.ordered:
            idx = idx[np.argsort(arr[idx])]  # Sort idx based on arr
        return np.flip(idx) if self.greater_is_better else idx

    def _take_mask(self, arr: NP1D_float | NP1D_bool, as_mask: bool = True) -> NP1D_int | NP1D_bool:
        """Assumes arr is a mask."""
        if not arr.dtype == bool:
            logger.warning("Array type is not boolean. Converting to bool...")
            arr = arr.astype(bool)
        return arr if as_mask else np.argwhere(arr).ravel()  # type: ignore

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

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __call__(self, obj: Any, as_mask: bool = True) -> NP1D_int | NP1D_bool:
        """Applies filtering conditions and returns a mask or index array.
        """
        if self.key is not None:
            obj = self._get_repr(obj, self.key)
        arr: NP1D_float | NP1D_bool = column_or_1d(obj)
        if self.cutoff is not None:
            return self._take_cutoff(arr.astype(float), as_mask=as_mask)
        elif self.top_k is not None:
            return self._take_top_k(arr.astype(float), as_mask=as_mask)
        # default to a mask
        return self._take_mask(arr, as_mask=as_mask)


class StackedFilterCondition:
    """A convenience class for stacking multiple FilterCondition's together.
    """

    def __init__(self, *fcs):
        self.fcs = []
        for fc in fcs:
            if isinstance(fc, FilterCondition):
                self.fcs.append(fc)
            elif isinstance(fc, StackedFilterCondition):
                self.fcs.extend(fc.fcs)
            else:
                raise TypeError(f"Cannot stack object of type {type(fc)}.")

    def __len__(self):
        return len(self.fcs)

    def __and__(self, other) -> 'StackedFilterCondition':
        return StackedFilterCondition(self, other)

    def __call__(self, obj: Any, as_mask: bool = True) -> NP1D_int | NP1D_bool:
        # Get first mask
        mask = self.fcs[0](obj, as_mask=True)
        # Get any remaining masks
        for fc in self.fcs[1:]:
            mask &= fc(obj, as_mask=True)
        return mask if as_mask else np.argwhere(mask).ravel()
