import abc
from typing import Dict, Literal, Tuple, overload

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from sklearn.utils import check_consistent_length
from sklearn.utils.validation import column_or_1d

from .cond_filter import Filter, StackedFilter
from .custom_types import NP1D_Any, NP1D_bool, NP1D_float, NP1D_int, NP1D_str
from .utils.stats import _correct


class TestSummary(BaseModel, abc.ABC):
    """A base class for Test summaries."""
    model_config = {
        'arbitrary_types_allowed': True,
        'validate_assignment': True,
        'extra': 'forbid',
        'validate_default': True,
    }

    name: NP1D_str | None = None

    @field_validator('*', mode='before')
    def _to_np(cls, v) -> NP1D_Any | None:
        # Convert to numpy before performing any validation
        return v if v is None else column_or_1d(v)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        not_none_arrs = self._tuple(exclude_none=True)
        if len(not_none_arrs) == 0:
            raise ValueError("No data is stored in this test summary.")
        check_consistent_length(*not_none_arrs)

    def __len__(self):
        return len(self._tuple(exclude_none=True)[0])  # type: ignore

    def __str__(self):
        s = f"{self.__class__.__name__} with fields "
        for field in self.model_fields:
            if getattr(self, field) is not None:
                s += f"'{field}', "
        s += f"of length {len(self)}."
        return s

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(\n"
        for field in self.model_fields:
            if (arr := getattr(self, field)) is not None:
                if arr.dtype == np.float_:
                    arr = np.round(arr, 3)
                arr = np.array_repr(arr)
                s += f"    {field}={arr},\n"
        s += ")"
        return s

    def __getitem__(self, val):
        return type(self)(**{
            field: (arr[val] if arr is not None else None)
            for field, arr in self.model_dump().items()
        })

    def _tuple(self, exclude_none: bool = False) -> Tuple[NP1D_float | None, ...]:
        """Converts self to tuple. To be used internally only."""
        data: Dict[str, NP1D_float] = self.model_dump(exclude_none=exclude_none)
        return tuple(data.values())

    def _get_key(self, key):
        arr = getattr(self, key, None)
        if arr is None:  # Includes the case when self.key has not been set.
            raise KeyError(f"Key '{key}' is None or not found in {self.__class__.__name__}.")
        return key

    def df(self) -> pd.DataFrame:
        """Converts self to a pandas dataframe."""
        return pd.DataFrame(data=self.model_dump(exclude_none=True))

    def array(self, dtype=None) -> np.ndarray:
        """Stacks all numeric vectors in the dataclass and returns an array
        where rows are the tests performed. Any arrays that are None will
        be replaced with arrays filled with np.nan. We do this to maintain
        shape consistency of the returned array.
        """
        cols = self._tuple()
        to_stack = [(i if i is not None else np.full_like(cols[0], np.nan)) for i in cols]
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
        _dict = {}
        for column in val.columns:
            _dict[column] = val[column].to_numpy()
        return cls.from_dict(_dict)

    @overload
    def where(self, *conds: Filter, as_mask: Literal[True]) -> NP1D_bool: ...

    @overload
    def where(self, *conds: Filter, as_mask: Literal[False]) -> NP1D_int: ...

    def where(self, *conds, as_mask=False) -> NP1D_int | NP1D_bool:
        """Given a condition which conists of a key field, a threshold
        (cutoff or top_k), return a mask or list of indices which satisfy
        the conditions.
        """
        # Iterate over conditions and take logical-& of masks returned
        if len(conds) == 1:
            return conds[0](self, as_mask=as_mask)
        return StackedFilter(*conds)(self, as_mask=as_mask)

    def argsort(self, by: str, reverse: bool = False) -> NP1D_int:
        argidx = np.argsort(getattr(self, by))
        return argidx if not reverse else np.flip(argidx)


class PvalTestSummary(TestSummary):
    """A base class for test summaries that take pvalues.

    Parameters
    __________
    pvals: 1D array
        p-values of the test.
    qvals: 1D array
        corrected p-values of the test. If these are not passed explicitly,
        they will be automatically computed using 'fdr_bh' correction.
    """
    pvals: NP1D_float
    qvals: NP1D_float = Field(None)

    @field_validator('qvals', mode='before')
    def _init_qvals(cls, qvals, info) -> NP1D_float:
        """Make sure qvals are initialized."""
        return qvals if qvals is not None else _correct(info.data['pvals'])[1]

    def __len__(self) -> int:
        return len(self.pvals)


class DETestSummary(PvalTestSummary):
    """A summary dataclas for DE test results.

    Parameters
    __________
    mean1, mean2: 1D array
        Mean of genes in group 1 vs the rest (group 2).
    log2fc: 1D array
        Log fold change of base 2.
    """

    # Group means
    mean1: NP1D_float | None = None
    mean2: NP1D_float | None = None
    log2fc: NP1D_float | None = None

    @property
    def abs_log2fc(self) -> NP1D_float | None:
        return np.abs(self.log2fc) if self.log2fc is not None else None


class KSTestSummary(DETestSummary):
    """A class for Kolmogorov-Smirnov test summary.
    """
    statistic: NP1D_float
    statistic_sign: NP1D_int


class BimodalTestSummary(PvalTestSummary):
    """A summary dataclass for bimodal test results.
    """

    statistic: NP1D_float | None = None
