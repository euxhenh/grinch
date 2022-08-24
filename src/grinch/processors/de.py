import logging
from typing import Optional

import numpy as np
from anndata import AnnData
from pydantic import Field
from sklearn.utils import indexable

from ..aliases import UNS
from ..custom_types import NP2D_float
from ..de_test_summary import DETestSummary
from ..utils.ops import group_indices
from ..utils.stats import _correct, ttest
from .base_processor import BaseProcessor

logger = logging.getLogger(__name__)


class TTest(BaseProcessor):

    class Config(BaseProcessor.Config):
        x_key: str = "X"
        save_key: str = f"uns.{UNS.TTEST}"
        group_key: str
        is_logged: bool = False
        # If the data is logged, this should point to the base of the
        # logarithm used. Can be 'e' or a positive float.
        base: Optional[float | str] = Field('e', gt=0, regex='e')
        correction: str = 'fdr_bh'

    cfg: Config

    def _ttest(self, x1: NP2D_float, x2: NP2D_float) -> DETestSummary:
        """Perform a single ttest."""
        mean1 = np.ravel(x1.mean(axis=0))
        mean2 = np.ravel(x2.mean(axis=0))

        pvals = ttest(x1, x2)[1]
        qvals = _correct(pvals, method=self.cfg.correction)[1]
        log2fc = _compute_log2fc(mean1, mean2, self.cfg.base, self.cfg.is_logged)

        return DETestSummary(pvals=pvals, qvals=qvals, mean1=mean1, mean2=mean2, log2fc=log2fc)

    def _process(self, adata: AnnData) -> None:
        group_labels = self.get_repr(adata, self.cfg.group_key)
        unq_labels, groups = group_indices(group_labels, as_mask=True)

        if len(unq_labels) == 1:
            raise ValueError(f"Found only one unique value under key '{self.cfg.group_key}'")

        x = self.get_repr(adata, self.cfg.x_key)
        x, = indexable(x)

        for label, group in zip(unq_labels, groups):
            ts: DETestSummary = self._ttest(x[group], x[~group])
            key = f"{self.cfg.save_key}.{label}"
            self.set_repr(adata, key, ts.df())


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
