import abc
import logging
from typing import Any, Dict, Optional, Union

from anndata import AnnData
from pydantic import Field, validator
from sklearn.decomposition import PCA as _PCA
from sklearn.decomposition import TruncatedSVD as _TruncatedSVD
from umap import UMAP as _UMAP

from .aliases import OBSM
from .processors import BaseProcessor

logger = logging.getLogger(__name__)


class BaseTransformer(BaseProcessor, abc.ABC):
    """A base estimator class for objects that implement `fit_transform`."""

    class Config(BaseProcessor.Config):
        ...

    @BaseProcessor.processor.setter  # type: ignore[attr-defined]
    def processor(self, value):
        """Check if the processor implements a `fit_transform` method."""
        fit_transform = getattr(value, 'fit_transform', None)
        if not callable(fit_transform):
            raise ValueError(
                f"Object of type '{type(value)}' does not implement "
                "a callable 'fit_transform' method."
            )
        super(BaseTransformer, self.__class__).processor.fset(self, value)

    def _process(self, adata: AnnData) -> None:
        if self.processor is None:
            raise NotImplementedError(
                f"Object of type {self.__class__} does not contain a processor object."
            )
        x_rep = self._get_repr(adata)
        x_rep_out = self.processor.fit_transform(x_rep)
        self._set_repr(adata, x_rep_out)


class PCA(BaseTransformer):

    class Config(BaseTransformer.Config):
        read_key: str = "X"
        save_key: str = f"obsm.{OBSM.X_PCA}"
        # PCA args
        n_components: Optional[Union[int, float, str]] = None
        whiten: bool = False
        svd_solver: str = 'auto'

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

        # Typing here is useful for editor autocompletion
        self.processor: _PCA = _PCA(
            n_components=self.cfg.n_components,
            whiten=self.cfg.whiten,
            svd_solver=self.cfg.svd_solver,
            random_state=self.cfg.seed,
        )


class TruncatedSVD(BaseTransformer):

    class Config(BaseTransformer.Config):
        read_key: str = "X"
        save_key: str = f"obsm.{OBSM.X_TRUNCATED_SVD}"
        # PCA args
        n_components: int = Field(2, ge=1)
        algorithm: str = 'randomized'
        n_iter: int = Field(5, ge=1)

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

        self.processor: _TruncatedSVD = _TruncatedSVD(
            n_components=self.cfg.n_components,
            algorithm=self.cfg.algorithm,
            n_iter=self.cfg.n_iter,
            random_state=self.cfg.seed,
        )


class UMAP(BaseTransformer):

    class Config(BaseTransformer.Config):
        read_key: str = "X"
        save_key: str = f"obsm.{OBSM.X_UMAP}"
        # PCA args
        n_neighbors: int = Field(15, ge=1)
        n_components: int = Field(2, ge=1)
        # Use a smaller spread by default, for tighter scatterplots
        spread: float = Field(0.8, gt=0)
        # Other arguments to pass to UMAP
        kwargs: Dict[str, Any] = {}

        @validator('kwargs')
        def remove_explicit_args(cls, val):
            """Don't allow any of the keys explicitly defined in Config
            to also be set in kwargs.
            """
            for explicit_key in ['n_neighbors', 'n_components', 'spread', 'random_state']:
                if val.pop(explicit_key, None) is not None:
                    logger.warning(
                        f"Popping '{explicit_key}' from kwargs. If you wish"
                        " to overwrite this key, pass it directly in the config."
                    )
            return val

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

        if self.cfg.seed is not None and 'transform_seed' not in self.cfg.kwargs:
            self.cfg.kwargs['transform_seed'] = self.cfg.seed

        self.processor: _UMAP = _UMAP(
            n_components=self.cfg.n_components,
            n_neighbors=self.cfg.n_neighbors,
            spread=self.cfg.spread,
            random_state=self.cfg.seed,
            **self.cfg.kwargs,
        )
