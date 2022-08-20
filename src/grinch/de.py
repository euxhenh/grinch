import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from anndata import AnnData

from .aliases import VARM
from .processors import BaseProcessor
from .utils.ops import group_indices
from .utils.stats import ttest

logger = logging.getLogger(__name__)


@dataclass
class TestSummary:
    names: Optional[np.ndarray] = field(default=None)
    pvals: Optional[np.ndarray] = field(default=None)
    qvals: Optional[np.ndarray] = field(default=None)
    # Group means
    mean1: Optional[np.ndarray] = field(default=None)
    mean2: Optional[np.ndarray] = field(default=None)
    log2fc: Optional[np.ndarray] = field(default=None)
    # Number of points in each group
    n1: Optional[int] = field(default=None)
    n2: Optional[int] = field(default=None)

    def to_array(self):
        to_stack = [self.pvals, self.log2fc, self.mean1, self.mean2]
        if self.qvals is not None:
            to_stack.append(self.qvals)
        return np.vstack(to_stack).T


class TTest(BaseProcessor):

    class Config(BaseProcessor.Config):
        x_key: str = "X"
        summary_prefix_key: str = f"varm.{VARM.TTEST}"
        splitter: str = ':'
        group_key: str
        is_logged: bool = False
        # If the data is logged, this should point to the base of the
        # logarithm used.
        base: Optional[str | float] = 'e'

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

    def _process(self, adata: AnnData) -> None:
        group_labels = self.get_repr(adata, self.cfg.group_key)
        unq_labels, groups = group_indices(group_labels, as_mask=True)

        if len(unq_labels) == 1:
            raise ValueError(f"Found only one unique value under key '{self.cfg.group_key}'")

        x = self.get_repr(adata, self.cfg.x_key)
        for label, group in zip(unq_labels, groups):
            x1 = x[group]
            x2 = x[~group]
            mean1 = np.ravel(x1.mean(axis=0))
            mean2 = np.ravel(x2.mean(axis=0))

            _, pvals = ttest(x1, x2)

            if self.cfg.is_logged:
                log2fc = mean1 - mean2
                # Convert base
                if self.cfg.base != 2:
                    base = np.e if self.cfg.base == 'e' else int(self.cfg.base)
                    log2fc *= np.log2(base)
            else:
                log2fc = np.log2((mean1 + 1) / (mean2 + 1))

            ts = TestSummary(
                pvals=pvals,
                mean1=mean1,
                mean2=mean2,
                log2fc=log2fc,
                n1=x1.shape[0],
                n2=x2.shape[0],
            )

            key = f"{self.cfg.summary_prefix_key}{self.cfg.splitter}{label}"
            self.set_repr(adata, key, ts.to_array())
