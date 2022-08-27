from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Extra, validate_arguments, validator
from sklearn.utils import check_consistent_length
from sklearn.utils.validation import column_or_1d

from .custom_types import NP1D_bool, NP1D_float, NP1D_int
from .utils.stats import _correct
from .utils.validation import only_one_not_None


class FilterCondition(BaseModel):

    class Config:
        validate_assignment = True
        extra = Extra.forbid
        validate_all = True

    key: str
    cutoff: Optional[float]
    top_k: Optional[int]
    greater_is_better: bool = False
    ordered: bool = False

    @validator('top_k')
    def _val_condition(cls, top_k, values):
        if not only_one_not_None(top_k, values['cutoff']):
            raise ValueError("Only one of 'cutoff' or 'top_k' must be specified.")
        return top_k

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

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __call__(self, obj: Any, as_mask: bool = True) -> NP1D_int | NP1D_bool:
        """Applies filtering conditions and returns a mask or index array.
        """
        arr: NP1D_float = column_or_1d(getattr(obj, self.key))
        if self.cutoff is not None:
            return self._take_cutoff(arr, as_mask=as_mask)
        return self._take_top_k(arr, as_mask=as_mask)


class DETestSummary(BaseModel):
    """A summary dataclas for DE test results.

    Parameters
    __________
    pvals: 1D array
        p-values of the test.
    qvals: 1D array
        corrected p-values of the test. If these are not passed explicitly,
        they will be automatically computed using 'fdr_bh' correction.
    mean1, mean2: 1D array
        Mean of genes in group 1 vs the rest (group 2).
    log2fc: 1D array
        Log fold change of base 2.
    """

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
        extra = Extra.ignore
        validate_all = True

    pvals: NP1D_float
    qvals: Optional[NP1D_float]
    # Group means
    mean1: Optional[NP1D_float]
    mean2: Optional[NP1D_float]
    log2fc: Optional[NP1D_float]

    @validator('*', pre=True)
    def _to_np(cls, v):
        # Convert to numpy before performing any validation
        return v if v is None else column_or_1d(v)

    @validator('qvals', pre=True)
    def _init_qvals(cls, qvals, values):
        """Make sure qvals are initialized."""
        return qvals if qvals is not None else _correct(values['pvals'])[1]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        not_none_arrs = self._tuple(exclude_none=True)
        check_consistent_length(*not_none_arrs)

    def __len__(self):
        return len(self.pvals)

    def _tuple(self, exclude_none: bool = False) -> Tuple[Optional[NP1D_float], ...]:
        """Converts self to tuple. To be used internally only."""
        data: Dict[str, NP1D_float] = self.dict(exclude_none=exclude_none)
        return tuple(data.values())

    def _get_key(self, key):
        arr = getattr(self, key, None)
        if arr is None:  # Includes the case when self.key has not been set.
            raise KeyError(f"Key '{key}' is None or not found in DETestSummary.")
        return key

    def df(self) -> pd.DataFrame:
        """Converts self to a pandas dataframe."""
        return pd.DataFrame(data=self.dict(exclude_none=True))

    def array(self, dtype=None) -> np.ndarray:
        """Stacks all numeric vectors in the dataclass and returns an array
        where rows are the tests performed. Any arrays that are None will
        be replaced with arrays filled with np.nan. We do this to maintain
        shape consistency of the returned array.
        """
        to_stack = [(i if i is not None else np.full_like(self.pvals, np.nan))
                    for i in self._tuple()]
        return np.vstack(to_stack).T.astype(dtype)  # type: ignore

    @classmethod
    def from_dict(cls, val: Dict) -> 'DETestSummary':
        """Constructs an instance of TestSummary given a dict. Extra fields
        are ignored.
        """
        return cls(**val)

    @classmethod
    def from_df(cls, val: pd.DataFrame) -> 'DETestSummary':
        """Constructs an instance of TestSummary given a df."""
        return cls.from_dict(val.to_dict('list'))

    def where(self, *conds: FilterCondition, as_mask: bool = False) -> NP1D_int | NP1D_bool:
        """Given a condition which conists of a key field, a threshold
        (cutoff or top_k), return a mask or list of indices which satisfy
        the conditions.
        """
        # Iterate over conditions and take logical-& of masks returned
        mask = np.full_like(self.pvals, True, dtype=bool)
        for cond in conds:
            mask &= cond(self, as_mask=True)
        return mask if as_mask else np.argwhere(mask).ravel()

    def argsort(self, by: str = 'qvals', reverse: bool = False) -> NP1D_int:
        argidx = np.argsort(getattr(self, by))
        return argidx if not reverse else np.flip(argidx)
