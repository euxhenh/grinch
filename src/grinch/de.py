import logging
from dataclasses import dataclass, field
from functools import wraps
from typing import Optional

import numpy as np
from anndata import AnnData
from pydantic import validator
from sklearn.utils import indexable
from statsmodels.stats.multitest import multipletests

from .aliases import UNS
from .base_processor import BaseProcessor
from .utils.ops import group_indices
from .utils.stats import ttest

logger = logging.getLogger(__name__)


@wraps(multipletests)
def _correct(pvals, method='fdr_bh'):
    """Simple wrapper for multiplesets."""
    return multipletests(
        pvals=pvals,
        alpha=0.05,
        method=method,
        is_sorted=False,
        returnsorted=False,
    )


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


@dataclass
class TestSummary:
    pvals: np.ndarray
    qvals: np.ndarray
    # Group means
    mean1: Optional[np.ndarray] = field(default=None)
    mean2: Optional[np.ndarray] = field(default=None)
    log2fc: Optional[np.ndarray] = field(default=None)

    def __post_init__(self):
        """Init qvals using a default correction of fdr_bg if qvals is not
        passed explicitly.
        """
        if self.qvals is None:
            self.qvals = _correct(self.pvals)[1]
        if self.mean1 is None:
            self.mean1 = np.full_like(self.pvals, np.nan)
        if self.mean2 is None:
            self.mean2 = np.full_like(self.pvals, np.nan)
        if self.log2fc is None:
            self.log2fc = np.full_like(self.pvals, np.nan)

    def __array__(self, dtype=None):
        """Allows np.array or np.asarray to convert this dataclass to the
        appropriate container."""
        return self.to_array(dtype=dtype)

    def to_array(self, dtype=None) -> np.ndarray:
        """Stacks all numeric vectors in the dataclass and returns an array
        where rows are the tests performed.
        """
        to_stack = [self.pvals, self.qvals, self.log2fc, self.mean1, self.mean2]
        return np.vstack(to_stack).T.astype(dtype)  # type: ignore


class TTest(BaseProcessor):

    class Config(BaseProcessor.Config):
        x_key: str = "X"
        summary_prefix_key: str = f"uns.{UNS.TTEST}"
        group_key: str
        is_logged: bool = False
        # If the data is logged, this should point to the base of the
        # logarithm used.
        base: Optional[float | str] = 'e'
        correction: str = 'fdr_bh'

        @validator('base')
        def valid_base(cls, base):
            """Make sure the base is valid. Accepted are 'e', or any
            positive float.
            """
            if isinstance(base, str):
                if base != 'e':
                    raise ValueError(f"Found invalid base '{base}'.")
                return base
            if base <= 0:
                raise ValueError("Negative bases are not allowed.")
            return base

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

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
            self.set_repr(adata, key, ts.to_array())
