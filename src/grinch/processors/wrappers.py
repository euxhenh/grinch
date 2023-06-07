import igraph as ig
import leidenalg as la
import numpy as np
from scipy.sparse import spmatrix

from ..custom_types import NP1D_int, NP2D_Any


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
                weights = [e['weight'] for e in G.es]
                kwargs['weights'] = weights
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
