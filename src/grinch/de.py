import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData
from pydantic import BaseModel, Extra, Field, validator
from sklearn.utils import check_consistent_length, indexable

from .aliases import UNS
from .base_processor import BaseProcessor
from .custom_types import NP1D_float
from .utils.ops import group_indices
from .utils.stats import _correct, ttest

logger = logging.getLogger(__name__)


def _compute_log2fc(mean1, mean2, base='e', is_logged=False):
    """Computes log2 fold change and converts base if data is already logged."""
    if is_logged:
        log2fc = mean1 - mean2
        # Convert base
        if base is not None and base != 2:
            base = np.e if base == 'e' else float(base)
            log2fc *= np.log2(base)
    else:
        log2fc = np.log2((mean1 + 1) / (mean2 + 1))
    return log2fc


class TestSummary(BaseModel):

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
        extra = Extra.ignore
        validate_all = True

    pvals: NP1D_float
    qvals: NP1D_float
    # Group means
    mean1: Optional[NP1D_float]
    mean2: Optional[NP1D_float]
    log2fc: Optional[NP1D_float]

    @validator('*', pre=True)
    def to_np(cls, v):
        # Convert to numpy before performing any validation
        return v if v is None or isinstance(v, np.ndarray) else np.asarray(v)

    @validator('qvals')
    def init_qvals(cls, qvals, values):
        """Make sure qvals are initialized."""
        return qvals if qvals is None else _correct(values['pvals'])[1]

    def __init__(self, *args, **kwargs):
        """Init qvals using a default correction of fdr_bg if qvals is not
        passed explicitly.
        """
        super().__init__(*args, **kwargs)
        not_none_arrs = self._tuple(exclude_none=True)
        check_consistent_length(*not_none_arrs)

    def _tuple(self, exclude_none: bool = False) -> Tuple[NP1D_float, ...]:
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


class TTest(BaseProcessor):

    class Config(BaseProcessor.Config):
        x_key: str = "X"
        summary_prefix_key: str = f"uns.{UNS.TTEST}"
        group_key: str
        is_logged: bool = False
        # If the data is logged, this should point to the base of the
        # logarithm used. Can be 'e' or a positive float.
        base: Optional[float | str] = Field('e', gt=0, regex='e')
        correction: str = 'fdr_bh'

    cfg: Config

    def _process(self, adata: AnnData) -> None:
        group_labels = self.get_repr(adata, self.cfg.group_key)
        unq_labels, groups = group_indices(group_labels, as_mask=True)

        if len(unq_labels) == 1:
            raise ValueError(f"Found only one unique value under key '{self.cfg.group_key}'")

        x = self.get_repr(adata, self.cfg.x_key)
        x, = indexable(x)

        for label, group in zip(unq_labels, groups):
            x1 = x[group]
            x2 = x[~group]
            mean1 = np.ravel(x1.mean(axis=0))
            mean2 = np.ravel(x2.mean(axis=0))

            _, pvals = ttest(x1, x2)
            qvals = _correct(pvals, method=self.cfg.correction)[1]
            log2fc = _compute_log2fc(mean1, mean2, self.cfg.base, self.cfg.is_logged)

            ts = TestSummary(pvals=pvals, qvals=qvals, mean1=mean1, mean2=mean2, log2fc=log2fc)

            key = f"{self.cfg.summary_prefix_key}.{label}"
            self.set_repr(adata, key, ts.df())
