import abc
from typing import Any, Dict, Tuple

import numpy as np
from anndata import AnnData
from pydantic import Field, validator
from scipy.sparse import csr_matrix, spmatrix
from sklearn.neighbors import NearestNeighbors as _NearestNeighbors
from sklearn.utils.validation import _ensure_sparse_format

from ..aliases import OBSM, OBSP
from ..custom_types import NP1D_float, NP1D_int, NP2D_float
from ..utils.validation import pop_args
from .base_processor import BaseProcessor
from .wrappers import FuzzySimplicialSet as _FuzzySimplicialSet


class BaseGraphConstructor(BaseProcessor, abc.ABC):
    """A base class for connectivity graph constructors.

    Parameters
    __________
    x_key: str
        What key to use for the data representation. I.e., distances
        between points will be computed using these values.
    conn_key: str
        Key to store the connectivity sparse matrix.
    dist_key: str
        Key to store the adjacency sparse matrix with edge weights.
    """

    class Config(BaseProcessor.Config):
        x_key: str = f"obsm.{OBSM.X_PCA}"
        conn_key: str
        dist_key: str
        save_stats: bool = True
        stats_key: str | None = None
        kwargs: Dict[str, Any] = {}

    cfg: Config

    def _process(self, adata: AnnData) -> None:
        x = self.get_repr(adata, self.cfg.x_key)
        adj: spmatrix = self._connect(x)  # type: ignore
        adj = _ensure_sparse_format(
            adj, accept_sparse='csr', dtype=None, copy=False,
            force_all_finite=True, accept_large_sparse=True)
        self.store_item(self.cfg.dist_key, adj.copy())
        adj.data = np.ones_like(adj.data)
        self.store_item(self.cfg.conn_key, adj)

    @abc.abstractmethod
    def _connect(self, x: NP2D_float) -> spmatrix:
        raise NotImplementedError

    def assemble(
        self,
        sources: NP1D_int,
        targets: NP1D_int,
        weights: NP1D_float,
        shape: Tuple[int, int] | None = None,
    ) -> csr_matrix:
        """Assembles edges and weights into a csr matrix."""
        adj = csr_matrix((weights, (sources, targets)), shape=shape)
        return adj


class KNNGraph(BaseGraphConstructor):
    """Distance graph constructor based on exact kNN."""

    class Config(BaseGraphConstructor.Config):
        conn_key: str = f"obsp.{OBSP.KNN_CONNECTIVITY}"
        dist_key: str = f"obsp.{OBSP.KNN_DISTANCE}"
        n_neighbors: int = Field(15, gt=0)
        n_jobs: int = Field(4, gt=0)

        @validator('kwargs')
        def remove_explicit_args(cls, val):
            return pop_args(['n_neighbors', 'n_jobs'], val)

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

        self.processor: _NearestNeighbors = _NearestNeighbors(
            n_neighbors=self.cfg.n_neighbors,
            n_jobs=self.cfg.n_jobs,
            **self.cfg.kwargs,
        )

    def _connect(self, x: NP2D_float) -> csr_matrix:
        self.processor.fit(x)
        return self.processor.kneighbors_graph(mode='distance')


class FuzzySimplicialSetGraph(BaseGraphConstructor):
    """A class for computing connectivities of samples based
    on UMAP's fuzzy simplicial set algorithm.
    I.e., higher weights mean closer or similar points.

    Parameters
    __________
    x_key: str
        Can be f"obsp.{OBSP.KNN_DISTANCE}" in which case will use
        precomputed knn distances as a warm start for umap. In this case
        must also set `precomputed=True`.
    precomputed: bool
        If True, will consider x as pre-computed neighbors.
    """

    class Config(BaseGraphConstructor.Config):
        conn_key: str = f"obsp.{OBSP.UMAP_CONNECTIVITY}"
        dist_key: str = f"obsp.{OBSP.UMAP_DISTANCE}"
        affinity_key: str = f"obsp.{OBSP.UMAP_AFFINITY}"
        precomputed: bool = False
        n_neighbors: int = 15
        metric: str = "euclidean"

        @validator('kwargs')
        def remove_explicit_args(cls, val):
            return pop_args(['X', 'n_neighbors', 'random_state', 'metric',
                             'knn_indices', 'knn_dists', 'return_dists'], val)

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

        self.processor: _FuzzySimplicialSet = _FuzzySimplicialSet(
            n_neighbors=self.cfg.n_neighbors,
            metric=self.cfg.metric,
            precomputed=self.cfg.precomputed,
            random_state=self.cfg.seed,
            **self.cfg.kwargs,
        )

    def _connect(self, x: NP2D_float | spmatrix) -> spmatrix:
        self.processor.fit(x)
        self.store_item(self.cfg.affinity_key, self.processor.affinity_adj_)
        return self.processor.distance_adj_
