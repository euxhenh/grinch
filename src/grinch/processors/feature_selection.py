from typing import TYPE_CHECKING, Callable

import numpy as np
from anndata import AnnData
from phenotype_cover import GreedyPC as _GreedyPC
from pydantic import Field
from sklearn.utils import check_X_y

from ..aliases import VAR
from .base_processor import BaseProcessor, ProcessorParam, ReadKey, WriteKey


class PhenotypeCover(BaseProcessor):
    """Marker selection based on multiset multicover.

    See https://www.sciencedirect.com/science/article/pii/S2667237522002296?via%3Dihub
    """
    __processor_attrs__ = [
        'n_elements_remaining_per_iter_',
        'coverage_per_iter_',
        'pairs_with_incomplete_cover_',
    ]

    class Config(BaseProcessor.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'PhenotypeCover']

        x_key: ReadKey = "X"
        y_key: ReadKey
        feature_mask_key: WriteKey = f"var.{VAR.PCOVER}"
        feature_importance_key: WriteKey = f"var.{VAR.PCOVER_I}"
        attrs_key: WriteKey | None = 'uns.{feature_mask_key}_'

        coverage: ProcessorParam[int]
        multiplier: ProcessorParam[int | None] = None
        ordered: ProcessorParam[bool] = True
        # If 0, run until coverage complete
        max_iters: ProcessorParam[int] = Field(default_factory=int, ge=0)
        verbose: ProcessorParam[bool] = Field(True, exclude=True)

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

        self.processor: _GreedyPC = _GreedyPC(
            ordered=self.cfg.ordered,
            verbose=self.cfg.verbose,
            multiplier=self.cfg.multiplier,
            **self.cfg.kwargs,
        )

    def _process(self, adata: AnnData) -> None:
        X = self.read(adata, self.cfg.x_key)
        y = self.read(adata, self.cfg.y_key)
        X, y = check_X_y(X, y, accept_sparse='csr')

        self.processor.fit(X, y)
        features = self.processor.select(self.cfg.coverage, max_iters=self.cfg.max_iters)

        mask = np.zeros(X.shape[1], dtype=bool)
        mask[features] = True
        self.store_item(self.cfg.feature_mask_key, mask)
        order = np.zeros(X.shape[1], dtype=int)
        order[features] = np.arange(len(features), 0, -1)
        self.store_item(self.cfg.feature_importance_key, order)
