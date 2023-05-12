from typing import List, Optional

import numpy as np
from anndata import AnnData
from phenotype_cover import GreedyPC as _GreedyPC
from pydantic import Field
from sklearn.utils import check_X_y

from ..aliases import UNS, VAR
from .base_processor import BaseProcessor


class PhenotypeCover(BaseProcessor):

    class Config(BaseProcessor.Config):
        x_key: str = "X"
        y_key: str
        feature_mask_key: str = f"var.{VAR.PCOVER_M}"
        feature_importance_key: str = f"var.{VAR.PCOVER_I}"

        save_stats: bool = True
        stats_key: str = f"uns.{UNS.PCOVER_}"

        # GreedyPC args
        coverage: int
        multiplier: Optional[int] = None
        ordered: bool = True
        # If 0, run until coverage complete
        max_iters: int = Field(default_factory=int, ge=0)

        verbose: bool = Field(True, exclude=True)

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

        self.processor: _GreedyPC = _GreedyPC(
            ordered=self.cfg.ordered,
            verbose=self.cfg.verbose,
            multiplier=self.cfg.multiplier,
        )

    def _process(self, adata: AnnData) -> None:
        X = self.get_repr(adata, self.cfg.x_key)
        y = self.get_repr(adata, self.cfg.y_key)
        X, y = check_X_y(X, y, accept_sparse='csr')

        if not X.ndim == 2:
            raise ValueError("Data matrix has to be 2 dimensional.")

        self.processor.fit(X, y)
        features = self.processor.select(self.cfg.coverage, max_iters=self.cfg.max_iters)

        mask = np.zeros(X.shape[1], dtype=bool)
        mask[features] = True
        self.store_item(self.cfg.feature_mask_key, mask)
        order = np.zeros(X.shape[1], dtype=int)
        order[features] = np.arange(len(features), 0, -1)
        self.store_item(self.cfg.feature_importance_key, order)

    @staticmethod
    def _processor_stats() -> List[str]:
        return BaseProcessor._processor_stats() + [
            'n_elements_remaining_per_iter_',
            'coverage_per_iter_',
            'pairs_with_incomplete_cover_',
        ]
