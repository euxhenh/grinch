import warnings

import igraph as ig
import leidenalg as la
import numpy as np
from scipy.sparse import csr_matrix, spmatrix
from umap.umap_ import fuzzy_simplicial_set

from ..custom_types import NP1D_int, NP2D_Any, NP2D_float
from ..utils.ops import get_indices_dists_from_adj


class Leiden:
    """A standalone leiden class that takes a graph and
    returns node communities.

    Parameters
    __________
    resolution: float
        Higher resolution results in more clusters. Default: 1.0.
    partition_type: str
        What partitioning algorithm to use. See
        https://leidenalg.readthedocs.io/en/stable/reference.html
        for a full list.
    directed: bool
        Whether to consider direction of edges in a graph.
    weighted: bool
        If True, will also use the weights from the graph.
        Note, higher weights mean greater 'similarity' between
        the nodes.
    seed: int
        Random seed.
    kwargs:
        Will be passed to la.find_partition()

    Attributes
    __________
    membership_: array
        The community assignments for each node.
    modularity_: float
        The value of modularity for the resulting communities.
    """
    def __init__(
        self,
        resolution: float = 1.0,
        partition_type: str = 'RBConfigurationVertexPartition',
        directed: bool = True,
        weighted: bool = True,
        seed: int | None = None,
        **kwargs,
    ):
        self.resolution = resolution
        self.partition_type = partition_type
        self.directed = directed
        self.weighted = weighted
        self.kwargs = kwargs
        self.seed = seed

        self.membership_: NP1D_int | None = None
        self.modularity_: float | None = None

    def fit(self, G: ig.Graph | NP2D_Any | spmatrix):
        """Given a graph G or a sparse adjacency matrix, find communities."""
        kwargs = self.kwargs.copy()
        kwargs = {'partition_type': getattr(la, self.partition_type)}
        if self.partition_type not in [
            'ModularityVertexPartition',
            'SurpriseVertexPartition',
        ]:
            kwargs['resolution_parameter'] = self.resolution

        if not isinstance(G, ig.Graph):
            mode = 'directed' if self.directed else 'undirected'
            if self.weighted:
                G = ig.Graph.Weighted_Adjacency(G, mode=mode, attr='weight')
                kwargs['weights'] = G.es['weight']
            else:
                G = (G != 0).astype(int)
                G = ig.Graph.Adjacency(G, mode=mode)

        partition = la.find_partition(G, seed=self.seed, **kwargs)
        self.membership_ = np.array(partition.membership, dtype=int)
        self.modularity_ = G.modularity(partition.membership)

    def predict(self, _):
        raise NotImplementedError(
            f"Class of type {self.__class__} does "
            "not implement a `predict` method."
        )

    def fit_predict(self, G: ig.Graph | NP2D_Any | spmatrix):
        self.fit(G)
        return self.membership_


class FuzzySimplicialSet:
    """Standalone class for computing nearest neighbors
    based on UMAP's fuzzy simplicial set algorithm.
    """
    def __init__(
        self,
        n_neighbors: int = 15,
        metric: str = 'euclidean',
        precomputed: bool = False,
        random_state: int | None = None,
        **kwargs,
    ):
        self.n_neighbors = n_neighbors
        self.precomputed = precomputed
        self.metric = metric
        self.random_state = random_state
        self.kwargs = kwargs

        self.adj: spmatrix = None

    def fit(self, X: NP2D_float | spmatrix):
        kwargs = self.kwargs.copy()
        if self.precomputed:
            knn_indices, knn_dists = get_indices_dists_from_adj(X)
            kwargs['knn_indices'] = knn_indices
            kwargs['knn_dists'] = knn_dists
            kwargs['X'] = csr_matrix((X.shape[0], 1))
            kwargs['n_neighbors'] = knn_indices.shape[1]
        else:
            kwargs['X'] = X
            kwargs['n_neighbors'] = self.n_neighbors

        adj = fuzzy_simplicial_set(
            metric=self.metric,
            random_state=self.random_state,
            **kwargs,
        )[0]

        self.adj = adj

    def predict(self, _):
        raise NotImplementedError(
            f"Class of type {self.__class__} does "
            "not implement a `predict` method."
        )

    def fit_predict(self, X: NP2D_float | spmatrix) -> spmatrix:
        self.fit(X)
        return self.adj
