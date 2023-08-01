import abc
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List

from anndata import AnnData
from pydantic import Field, field_validator, validate_call
from sklearn.decomposition import PCA as _PCA
from sklearn.decomposition import TruncatedSVD as _TruncatedSVD
from sklearn.manifold import MDS as _MDS
from umap import UMAP as _UMAP

from ..aliases import OBSM
from ..utils.validation import check_has_processor
from .base_processor import BaseProcessor, ProcessorParam, adata_modifier

logger = logging.getLogger(__name__)


class BaseTransformer(BaseProcessor, abc.ABC):
    """A base estimator class for objects that implement `fit_transform`."""

    class Config(BaseProcessor.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'BaseTransformer']

        x_key: str = "X"
        x_emb_key: str
        save_stats: bool = True
        stats_key: str | None = None
        kwargs: Dict[str, Any] = {}

        @field_validator('stats_key')
        def init_stats_key_with_x_emb(cls, val, info):
            return val or f"uns.{info.data['x_emb_key'].split('.', 1)[1]}_"

    cfg: Config

    @staticmethod
    def _processor_must_implement() -> List[str]:
        return BaseProcessor._processor_must_implement() + ['transform', 'fit_transform']

    def _process(self, adata: AnnData) -> None:
        """Gets the data representation to use and applies the transformer.
        """
        check_has_processor(self)
        x = self.get_repr(adata, self.cfg.x_key)
        x_emb = self.processor.fit_transform(x)
        self.store_item(self.cfg.x_emb_key, x_emb)

    @adata_modifier
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def transform(self, adata: AnnData) -> None:
        """Applies a transform only. Uses the same key as x_key.
        """
        check_has_processor(self)

        x = self.get_repr(adata, self.cfg.x_key)
        x_emb = self.processor.transform(x)
        self.store_item(self.cfg.x_emb_key, x_emb)


class PCA(BaseTransformer):

    class Config(BaseTransformer.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'PCA']

        x_emb_key: str = f"obsm.{OBSM.X_PCA}"
        n_components: ProcessorParam[int | float | str | None] = 50
        whiten: ProcessorParam[bool] = False
        svd_solver: ProcessorParam[str] = 'auto'

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

    @staticmethod
    def _processor_stats() -> List[str]:
        return BaseTransformer._processor_stats() + [
            'singular_values_',
            'explained_variance_',
            'explained_variance_ratio_',
            'components_',
        ]


class TruncatedSVD(BaseTransformer):

    class Config(BaseTransformer.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'TruncatedSVD']

        x_emb_key: str = f"obsm.{OBSM.X_TRUNCATED_SVD}"
        # Truncated SVD args
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

    @staticmethod
    def _processor_stats() -> List[str]:
        return BaseTransformer._processor_stats() + [
            'singular_values_',
            'explained_variance_',
            'explained_variance_ratio_',
            'components_',
        ]


class MDS(BaseTransformer):

    class Config(BaseTransformer.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'MDS']

        x_emb_key: str = f"obsm.{OBSM.X_MDS}"
        n_components: int = Field(2, ge=1)

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

        self.processor: _MDS = _MDS(
            n_components=self.cfg.n_components,
            random_state=self.cfg.seed,
            **self.cfg.kwargs,
        )


class UMAP(BaseTransformer):

    class Config(BaseTransformer.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'UMAP']

        x_key: str = f"obsm.{OBSM.X_PCA}"  # Different x key from parent
        x_emb_key: str = f"obsm.{OBSM.X_UMAP}"
        # UMAP args
        n_neighbors: int = Field(15, ge=1)
        n_components: int = Field(2, ge=1)
        # Use a smaller spread by default, for tighter scatterplots
        spread: float = Field(0.8, gt=0)
        # Other arguments to pass to UMAP
        kwargs: Dict[str, Any] = {}

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
