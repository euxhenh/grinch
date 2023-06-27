import abc
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from anndata import AnnData
from pydantic import Field, validate_arguments, validator
from sklearn.cluster import KMeans as _KMeans
from sklearn.linear_model import LogisticRegression as _LogisticRegression
from sklearn.utils import indexable

from ..aliases import OBS, OBSM, OBSP, UNS
from ..custom_types import NP1D_Any, NP1D_float, NP2D_float
from ..utils.ops import group_indices
from ..utils.validation import check_has_processor, pop_args
from .base_processor import BaseProcessor, adata_modifier
from .wrappers import Leiden as _Leiden

logger = logging.getLogger(__name__)


class BasePredictor(BaseProcessor, abc.ABC):
    """A base class for estimators, clustering algorithms, etc."""

    class Config(BaseProcessor.Config):
        x_key: str = f"obsm.{OBSM.X_PCA}"
        labels_key: str
        save_stats: bool = True
        kwargs: Dict[str, Any] = {}

    cfg: Config

    @staticmethod
    def _processor_must_implement() -> List[str]:
        return BaseProcessor._processor_must_implement() + ['predict']

    @adata_modifier
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(self, adata: AnnData) -> None:
        """Calls predict on the underlying predictor."""
        check_has_processor(self)

        x = self.get_repr(adata, self.cfg.x_key)
        labels = self.processor.predict(x)
        self.store_item(self.cfg.labels_key, labels)


class BaseUnsupervisedPredictor(BasePredictor, abc.ABC):
    """A base class for unsupervised predictors, e.g., clustering."""

    class Config(BasePredictor.Config):
        ...

    cfg: Config

    @staticmethod
    def _processor_must_implement() -> List[str]:
        return BasePredictor._processor_must_implement() + ['fit_predict']

    def _process(self, adata: AnnData) -> None:
        check_has_processor(self)
        # Fits the data and stores predictions.
        x = self.get_repr(adata, self.cfg.x_key)
        labels = self.processor.fit_predict(x)
        self.store_item(self.cfg.labels_key, labels)


def centroids_from_Xy(X, y: NP1D_Any) -> Dict[str, NP1D_float]:
    """Computes the X centroids of each group in y.
    """
    X, = indexable(X)
    assert X.shape[0] == len(y)  # type: ignore

    # Compute centers and store them as well
    label_to_centroid = {}
    unq_labels, groups = group_indices(y)
    for label, group in zip(unq_labels, groups):
        label_to_centroid[str(label)] = np.ravel(X[group].mean(axis=0))
    return label_to_centroid


class KMeans(BaseUnsupervisedPredictor):

    class Config(BaseUnsupervisedPredictor.Config):
        labels_key: str = f"obs.{OBS.KMEANS}"
        stats_key: str = f"uns.{UNS.KMEANS_}"
        # KMeans args
        n_clusters: int = Field(8, ge=1)

        @validator('kwargs')
        def remove_explicit_args(cls, val):
            return pop_args(['n_clusters', 'random_state'], val)

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

        self.processor: _KMeans = _KMeans(
            n_clusters=self.cfg.n_clusters,
            random_state=self.cfg.seed,
            n_init='auto',  # temporary warning supress
            **self.cfg.kwargs,
        )

    @staticmethod
    def _processor_stats() -> List[str]:
        return BaseUnsupervisedPredictor._processor_stats() + \
            ['cluster_centers_']


class Leiden(BaseUnsupervisedPredictor):

    class Config(BaseUnsupervisedPredictor.Config):
        x_key: str = f"obsp.{OBSP.UMAP_AFFINITY}"
        labels_key: str = f"obs.{OBS.LEIDEN}"
        stats_key: str = f"uns.{UNS.LEIDEN_}"
        resolution: float = Field(1.0, gt=0)
        n_iterations: int = -1
        partition_type: str = 'RBConfigurationVertexPartition'
        directed: bool = True
        weighted: bool = True
        # Set to True if should comute cluster centers based
        # on community assignment too.
        compute_centroids: bool = True
        x_key_for_centroids: str = "X"

        @validator('kwargs')
        def remove_explicit_args(cls, val):
            return pop_args(['partition_type', 'graph', 'weights',
                             'n_iterations', 'seed'], val)

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

        self.processor: _Leiden = _Leiden(
            resolution=self.cfg.resolution,
            partition_type=self.cfg.partition_type,
            directed=self.cfg.directed,
            weighted=self.cfg.weighted,
            seed=self.cfg.seed,
            n_iterations=self.cfg.n_iterations,
            **self.cfg.kwargs,
        )

    def _process(self, adata: AnnData) -> None:
        super()._process(adata)

        if not self.cfg.compute_centroids:
            return

        x = self.get_repr(adata, self.cfg.x_key_for_centroids)
        labels = np.asarray(self.processor.membership_)
        label_to_centroid = centroids_from_Xy(x, labels)
        self.store_item(f"{self.cfg.stats_key}.cluster_centers_",
                        label_to_centroid)

    @staticmethod
    def _processor_stats() -> List[str]:
        return BaseUnsupervisedPredictor._processor_stats() + \
            ['modularity_']


class BaseSupervisedPredictor(BasePredictor, abc.ABC):
    """A base class for unsupervised predictors, e.g., clustering."""

    class Config(BasePredictor.Config):
        y_key: str

    cfg: Config

    @staticmethod
    def _processor_must_implement() -> List[str]:
        return BasePredictor._processor_must_implement() + ['fit']

    def _process(self, adata: AnnData) -> None:
        check_has_processor(self)
        # Override process method since LogisticRegression does not have a
        # fit_predict method and also requires labels to fit.
        x = self.get_repr(adata, self.cfg.x_key)
        y = self.get_repr(adata, self.cfg.y_key)
        if hasattr(self.processor, 'fit_predict'):
            labels = self.processor.fit_predict(x, y)
        else:
            self.processor.fit(x, y)
            labels = self.processor.predict(x)
        self.store_item(self.cfg.labels_key, labels)


class LogisticRegression(BaseSupervisedPredictor):

    class Config(BaseSupervisedPredictor.Config):
        labels_key: str = f"obs.{OBS.LOG_REG}"
        stats_key: str = f"uns.{UNS.LOG_REG_}"
        # LogisticRegression kwargs
        penalty: str = "l2"
        C: float = Field(1.0, gt=0)  # inverse regularization trade-off
        max_iter: int = Field(500, gt=0)
        n_jobs: Optional[int] = Field(-1, ge=-1)

        @validator('kwargs')
        def remove_explicit_args(cls, val):
            return pop_args(['penalty', 'C', 'max_iter',
                             'n_jobs', 'random_state'], val)

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

        self.processor: _LogisticRegression = _LogisticRegression(
            penalty=self.cfg.penalty,
            C=self.cfg.C,
            max_iter=self.cfg.max_iter,
            n_jobs=self.cfg.n_jobs,
            random_state=self.cfg.seed,
            **self.cfg.kwargs,
        )

    @staticmethod
    def _processor_stats() -> List[str]:
        return BaseSupervisedPredictor._processor_stats() + \
            ['classes_', 'coef_', 'intercept_']
