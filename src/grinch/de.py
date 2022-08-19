import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from anndata import AnnData
from scipy.stats import ttest_ind

from .aliases import VARM
from .processors import BaseProcessor
from .utils.ops import group_indices

logger = logging.getLogger(__name__)


@dataclass
class TestSummary:
    names: Optional[np.ndarray] = field(None)
    pvals: np.ndarray
    qvals: Optional[np.ndarray] = field(None)
    # Group means
    mean1: np.ndarray
    mean2: np.ndarray
    log2fc: np.ndarray
    # Number of points in each group
    n1: int
    n2: int

    def to_array(self):
        to_stack = [self.pvals, self.mean1, self.mean2, self.log2fc]
        if self.qvals is not None:
            to_stack.append(self.qvals)
        return np.hstack(to_stack)


class TTest(BaseProcessor):

    class Config(BaseProcessor.Config):
        x_key: str = "X"
        summary_prefix_key: str = f"varm.{VARM.TTEST}"
        splitter: str = ':'
        group_key: str
        is_logged: bool = False
        # If the data is logged, this should point to the base of the
        # logarithm used.
        base: str | float = 'e'

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
            mean1 = x1.mean(axis=0)
            mean2 = x2.mean(axis=0)

            _, pvals = ttest_ind(x1, x2, equal_var=False)

            if self.cfg.is_logged:
                log2fc = mean1 - mean2
                # Conver base
                if self.cfg.base != 2:
                    base = np.e if self.cfg.base == 'e' else self.cfg.base
                    log2fc *= np.log2(base)
            else:
                log2fc = np.log2((mean1 + 1) / (mean2 + 1))

            ts = TestSummary(
                pvals=pvals,
                mean1=mean1,
                mean2=mean2,
                log2fc=log2fc,
                n1=len(x1),
                n2=len(x2),
            )

            key = f"{self.cfg.summary_prefix_key}{self.cfg.splitter}{label}"
            self.set_repr(adata, key, ts.to_array())
