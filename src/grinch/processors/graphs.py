import abc
from typing import Any, Dict, Tuple

from anndata import AnnData
from pydantic import Field, validator
from scipy.sparse import csr_matrix, spmatrix
from sklearn.neighbors import NearestNeighbors as _NearestNeighbors
from sklearn.utils.validation import _ensure_sparse_format

from ..aliases import OBSM, OBSP
from ..custom_types import NP1D_float, NP1D_int, NP2D_float
from ..utils.validation import pop_args
from .base_processor import BaseProcessor


class BaseGraphConstructor(BaseProcessor, abc.ABC):
    """A base class for connectivity graph constructors"""

    class Config(BaseProcessor.Config):
        x_key: str = f"obsm.{OBSM.X_PCA}"
        adj_key: str
        save_stats: bool = True
        stats_key: str | None = None
        kwargs: Dict[str, Any] = {}

    cfg: Config

    def _process(self, adata: AnnData) -> None:
        x: NP2D_float = self.get_repr(adata, self.cfg.x_key)  # type: ignore
        adj: spmatrix = self._connect(x)
        adj = _ensure_sparse_format(
            adj, accept_sparse='csr', dtype=None, copy=False,
            force_all_finite=True, accept_large_sparse=True)
        self.store_item(self.cfg.adj_key, adj)

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
    """Connectivity graph constructor based on exact kNN."""

    class Config(BaseGraphConstructor.Config):
        adj_key: str = f"obsp.{OBSP.KNN}"
        n_neighbors: int = Field(15, gt=0)
        n_jobs: int = Field(4, gt=0)
        mode: str = "distance"
        include_self: bool = False

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
        adj = self.processor.kneighbors_graph(mode=self.cfg.mode)
        return adj
