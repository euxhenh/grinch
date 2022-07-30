import abc
import logging
from typing import Any, Dict, Optional

from anndata import AnnData
from pydantic import Field, validator
from sklearn.cluster import KMeans as _KMeans
from sklearn.linear_model import LogisticRegression as _LogisticRegression

from .aliases import OBS, OBSM
from .processors import BaseProcessor
from .utils.validation import pop_args

logger = logging.getLogger(__name__)


class BaseEstimator(BaseProcessor, abc.ABC):
    """A base class for estimators, clustering algorithms, etc."""

    class Config(BaseProcessor.Config):
        ...

    @BaseProcessor.processor.setter  # type: ignore[attr-defined]
    def processor(self, value):
        """Check if processor implements a fit_predict method."""
        fit_predict = getattr(value, 'fit_predict', None)
        if not callable(fit_predict):
            raise ValueError(
                f"Object of type '{type(value)}' does not implement "
                "a callable 'fit_predict' method."
            )
        super(BaseEstimator, self.__class__).processor.fset(self, value)

    def _process(self, adata: AnnData) -> None:
        if self.processor is None:
            raise NotImplementedError(
                f"Object of type {self.__class__} does not contain a processor object."
            )
        x_rep = self._get_repr(adata)
        preds = self.processor.fit_predict(x_rep)
        self._set_repr(adata, preds)


class KMeans(BaseEstimator):

    class Config(BaseEstimator.Config):
        read_key: str = f"obsm.{OBSM.X_PCA}"
        save_key: str = f"obs.{OBS.KMEANS}"
        # KMeans args
        n_clusters: int = Field(8, ge=2)
        kwargs: Dict[str, Any] = {}

        @validator('kwargs')
        def remove_explicit_args(cls, val):
            return pop_args(['n_clusters', 'random_state'], val)

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

        self.processor: _KMeans = _KMeans(
            n_clusters=self.cfg.n_clusters,
            random_state=self.cfg.seed,
            **self.cfg.kwargs,
        )


class LogisticRegression(BaseEstimator):

    class Config(BaseEstimator.Config):
        read_key: str = f"obsm.{OBSM.X_PCA}"
        save_key: str = f"obs.{OBS.LOG_REG_PREDS}"
        # LogisticRegression kwargs
        penalty: str = "l2"
        C: float = Field(1.0, gt=0)  # inverse regularization trade-off
        max_iter: int = Field(100, gt=0)
        n_jobs: Optional[int] = Field(-1, ge=-1)
        kwargs: Dict[str, Any] = {}

        @validator('kwargs')
        def remove_explicit_args(cls, val):
            return pop_args(['penalty', 'C', 'max_iter', 'n_jobs', 'random_state'], val)

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
