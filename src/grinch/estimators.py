import abc
import logging
import types
from typing import Any, Dict, Optional

from anndata import AnnData
from pydantic import Field, validator
from sklearn.cluster import KMeans as _KMeans
from sklearn.linear_model import LogisticRegression as _LogisticRegression

from .aliases import OBS, OBSM, REP_KEY
from .processors import BaseProcessor
from .utils.validation import pop_args

logger = logging.getLogger(__name__)


def adata_to_rep(func):
    """A decorator that obtains a rep from adata and calls a self.processor
    method with the same name as func using the new rep. Useful to convert
    adata to a data matrix X based on self.read_key.

    Parameters
    func: callable
        Must return another callable method when called.
    """
    def _wrapper(self, adata: AnnData, *args, **kwargs):
        x_rep = self._get_repr(adata)
        match self.cfg.read_key:
            case str():
                return func(x_rep, *args, **kwargs)
            case [*vals]:
                return func(*x_rep, *args, **kwargs)
            case {**vals}:  # noqa
                return func({**x_rep, **kwargs})
    return _wrapper


class BaseEstimator(BaseProcessor, abc.ABC):
    """A base class for estimators, clustering algorithms, etc."""

    class Config(BaseProcessor.Config):
        ...

    @BaseProcessor.processor.setter  # type: ignore[attr-defined]
    def processor(self, value):
        """Check if processor implements fit predict methods."""
        must_implement = ['fit', 'predict', 'fit_predict']
        for method_name in must_implement:
            method = getattr(value, method_name, None)
            if not callable(method):
                raise ValueError(
                    f"Object of type '{type(value)}' does not implement "
                    f"a callable '{method_name}' method."
                )
        super(BaseEstimator, self.__class__).processor.fset(self, value)

        # Wrap function signatures to accept AnnData representations.
        self.fit = types.MethodType(adata_to_rep(self.processor.fit), self)
        self.predict = types.MethodType(adata_to_rep(self.processor.predict), self)
        self.fit_predict = types.MethodType(adata_to_rep(self.processor.fit_predict), self)

    def _process(self, adata: AnnData) -> None:
        if self.processor is None:
            raise NotImplementedError(
                f"Object of type {self.__class__} does not contain a processor object."
            )
        preds = self.fit_predict(adata)
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
        read_key: REP_KEY = None
        save_key: REP_KEY = f"obs.{OBS.LOG_REG}"
        # LogisticRegression kwargs
        penalty: str = "l2"
        C: float = Field(1.0, gt=0)  # inverse regularization trade-off
        max_iter: int = Field(500, gt=0)
        n_jobs: Optional[int] = Field(-1, ge=-1)
        kwargs: Dict[str, Any] = {}

        @validator('kwargs')
        def remove_explicit_args(cls, val):
            return pop_args(['penalty', 'C', 'max_iter', 'n_jobs', 'random_state'], val)

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

        processor: _LogisticRegression = _LogisticRegression(
            penalty=self.cfg.penalty,
            C=self.cfg.C,
            max_iter=self.cfg.max_iter,
            n_jobs=self.cfg.n_jobs,
            random_state=self.cfg.seed,
            **self.cfg.kwargs,
        )

        # Manually implement a fit_predict function
        def _fit_predict(X, y):
            processor.fit(X, y)
            return processor.predict(X)

        processor.fit_predict = _fit_predict
        self.processor = processor
