from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Extra, validator
from sklearn.utils import check_consistent_length
from sklearn.utils.validation import column_or_1d

from .custom_types import NP1D_bool, NP1D_float, NP1D_int
from .utils.stats import _correct


class FilterCondition(BaseModel):

    class Config:
        validate_assignment = True
        extra = Extra.forbid
        validate_all = True

    key: str
    cutoff: float
    greater_is_better: bool = False


class TestSummary(BaseModel):
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
    def to_np(cls, v):
        # Convert to numpy before performing any validation
        return v if v is None else column_or_1d(v)

    @validator('qvals', pre=True)
    def init_qvals(cls, qvals, values):
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
    def from_dict(cls, val: Dict) -> 'TestSummary':
        """Constructs an instance of TestSummary given a dict. Extra fields
        are ignored.
        """
        return cls(**val)

    @classmethod
    def from_df(cls, val: pd.DataFrame) -> 'TestSummary':
        """Constructs an instance of TestSummary given a df."""
        return cls.from_dict(val.to_dict('list'))

    def _where(self, cond: FilterCondition) -> NP1D_bool:
        """Helper function for self.where that takes a single condition."""
        arr = getattr(self, cond.key, None)
        if arr is None:  # Includes the case when self.key has not been set.
            raise KeyError(f"Key '{cond.key}' not found in TestSummary.")
        if cond.greater_is_better:
            return arr >= cond.cutoff
        return arr <= cond.cutoff

    def where(
        self,
        *conds: Tuple[FilterCondition, ...],
        as_mask: bool = False
    ) -> NP1D_int | NP1D_bool:
        """Given a condition which conists of a key field, a threshold,
        return a mask or list of indices which satisfy the conditions.
        """
        mask = np.ones_like(self.pvals).astype(bool)
        for cond in conds:
            mask &= self._where(cond)
        if as_mask:
            return mask
        return np.argwhere(mask).flatten()
