import logging
from typing import Optional

import numpy as np
from anndata import AnnData
from pydantic import Field
from sklearn.utils import column_or_1d
from tqdm.auto import tqdm

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
        show_progress_bar: bool = Field(True, exclude=True)
        replace_nan: bool = True

    cfg: Config

    def _ttest(self, pmv: PartMeanVar, label) -> DETestSummary:
        """Perform a single ttest."""
        n1, m1, v1 = pmv.compute([label], ddof=1)  # take label
        n2, m2, v2 = pmv.compute([label], ddof=1, exclude=True)  # take all but label

        pvals = ttest_from_mean_var(n1, m1, v1, n2, m2, v2)[1]
        not_none_mask = ~np.isnan(pvals)

        qvals = np.full_like(pvals, 1.0 if self.cfg.replace_nan else np.nan)
        qvals[not_none_mask] = _correct(pvals[not_none_mask], method=self.cfg.correction)[1]

        log2fc = np.full_like(pvals, 0.0 if self.cfg.replace_nan else np.nan)
        log2fc[not_none_mask] = _compute_log2fc(
            m1[not_none_mask],
            m2[not_none_mask],
            self.cfg.base,
            self.cfg.is_logged,
        )

        if self.cfg.replace_nan:
            pvals[~not_none_mask] = 1.0

        return DETestSummary(pvals=pvals, qvals=qvals, mean1=m1, mean2=m2, log2fc=log2fc)

    def _process(self, adata: AnnData) -> None:
        group_labels = column_or_1d(self.get_repr(adata, self.cfg.group_key))
        unq_labels = np.unique(group_labels)
        if len(unq_labels) == 1:
            raise ValueError(f"Found only one unique value under key '{self.cfg.group_key}'")

        x = self.get_repr(adata, self.cfg.x_key)
        pmv = PartMeanVar(x, group_labels, self.cfg.show_progress_bar)

        to_iter = (
            tqdm(unq_labels, desc="Running t-Tests") if self.cfg.show_progress_bar
            else unq_labels
        )
        for label in to_iter:
            ts: DETestSummary = self._ttest(pmv, label)
            key = f"{self.cfg.save_key}.{label}"
            self.set_repr(adata, key, ts.df())
