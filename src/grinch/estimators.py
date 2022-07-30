import abc
import logging
from typing import Any, Dict

from anndata import AnnData
from pydantic import Field, validator
from sklearn.cluster import KMeans as _KMeans

from .aliases import OBS, OBSM
from .processors import BaseProcessor

logger = logging.getLogger(__name__)


class BaseEstimator(BaseProcessor, abc.ABC):
    """A base class for estimators, clustering algorithms, etc."""

    class Config(BaseProcessor.Config):
        ...

    @BaseProcessor.processor.setter
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
            """Don't allow any of the keys explicitly defined in Config
            to also be set in kwargs.
            """
            for explicit_key in ['n_clusters', 'random_state']:
                if val.pop(explicit_key, None) is not None:
                    logger.warning(
                        f"Popping '{explicit_key}' from kwargs. If you wish"
                        " to overwrite this key, pass it directly in the config."
                    )
            return val

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

        self.processor: _KMeans = _KMeans(
            n_clusters=self.cfg.n_clusters,
            random_state=self.cfg.seed,
            **self.cfg.kwargs,
        )
