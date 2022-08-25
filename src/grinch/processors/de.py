import logging
from typing import Optional

import numpy as np
from anndata import AnnData
from pydantic import Field

from ..aliases import UNS
from ..de_test_summary import DETestSummary
from ..utils.stats import (
    PartMeanVar,
    _compute_log2fc,
    _correct,
    ttest_from_mean_var,
)
from .base_processor import BaseProcessor

logger = logging.getLogger(__name__)


class TTest(BaseProcessor):

    class Config(BaseProcessor.Config):
        x_key: str = "X"
        save_key: str = f"uns.{UNS.TTEST}"
        group_key: str
        is_logged: bool = True
        # If the data is logged, this should point to the base of the
        # logarithm used. Can be 'e' or a positive float.
        base: Optional[float | str] = Field('e', gt=0, regex='e')
        correction: str = 'fdr_bh'

    cfg: Config

    def _ttest(self, pmv: PartMeanVar, label) -> DETestSummary:
        """Perform a single ttest."""
        n1, m1, v1 = pmv.compute([label], ddof=1)
        n2, m2, v2 = pmv.compute([label], ddof=1, exclude=True)
        pvals = ttest_from_mean_var(n1, m1, v1, n2, m2, v2)[1]
        qvals = _correct(pvals, method=self.cfg.correction)[1]
        log2fc = _compute_log2fc(m1, m2, self.cfg.base, self.cfg.is_logged)
        return DETestSummary(pvals=pvals, qvals=qvals, mean1=m1, mean2=m2, log2fc=log2fc)

    def _process(self, adata: AnnData) -> None:
        group_labels = self.get_repr(adata, self.cfg.group_key)
        unq_labels = np.unique(group_labels)
        if len(unq_labels) == 1:
            raise ValueError(f"Found only one unique value under key '{self.cfg.group_key}'")

        x = self.get_repr(adata, self.cfg.x_key)
        pmv = PartMeanVar(x, group_labels)

        for label in unq_labels:
            ts: DETestSummary = self._ttest(pmv, label)
            key = f"{self.cfg.save_key}.{label}"
            self.set_repr(adata, key, ts.df())
