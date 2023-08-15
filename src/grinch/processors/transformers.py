import abc
import logging
from typing import TYPE_CHECKING, Callable

from anndata import AnnData
from pydantic import Field, PositiveFloat, PositiveInt, validate_call
from sklearn.decomposition import PCA as _PCA
from sklearn.decomposition import TruncatedSVD as _TruncatedSVD
from sklearn.manifold import MDS as _MDS
from umap import UMAP as _UMAP

from ..aliases import OBSM
from ..base import StorageMixin
from ..utils.validation import check_has_processor
from .base_processor import BaseProcessor, ProcessorParam, ReadKey, WriteKey

logger = logging.getLogger(__name__)


class BaseTransformer(BaseProcessor, abc.ABC):
    """A base estimator class for objects that implement `fit_transform`."""
    __processor_reqs__ = ['transform', 'fit_transform']

    class Config(BaseProcessor.Config):
        """
        Parameters
        ----------
        x_key : str
            Key holding the data matrix X to be transformed.

        write_key : str
            The key to store the transformed data.

        attrs_key : str, default='{write_key}_'
            The key to store processors attributes in (post fit). Curly
            brackets will be formatted. By default use `self.write_key`
            followed by an underscore.
        """

        if TYPE_CHECKING:
            create: Callable[..., 'BaseTransformer']

        x_key: ReadKey = "X"
        write_key: WriteKey = Field(alias='x_emb_key')
        attrs_key: WriteKey | None = 'uns.{write_key}_'

    cfg: Config

    def _process(self, adata: AnnData) -> None:
        check_has_processor(self)
        x = self.read(adata, self.cfg.x_key)
        x_emb = self.processor.fit_transform(x)
        self.store_item(self.cfg.write_key, x_emb)

    @StorageMixin.lazy_writer
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def transform(self, adata: AnnData) -> None:
        check_has_processor(self)
        x = self.read(adata, self.cfg.x_key)
        x_emb = self.processor.transform(x)
        self.store_item(self.cfg.write_key, x_emb)


class PCA(BaseTransformer):
    """Principal Component Analysis.

    See https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    """
    __processor_attrs__ = [
        'singular_values_',
        'explained_variance_',
        'explained_variance_ratio_',
        'components_',
    ]

    class Config(BaseTransformer.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'PCA']

        write_key: WriteKey = f"obsm.{OBSM.X_PCA}"
        n_components: ProcessorParam[PositiveInt | PositiveFloat | str | None] = 50
        whiten: ProcessorParam[bool] = False
        svd_solver: ProcessorParam[str] = 'auto'

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

        self.processor: _PCA = _PCA(
            n_components=self.cfg.n_components,
            whiten=self.cfg.whiten,
            svd_solver=self.cfg.svd_solver,
            random_state=self.cfg.seed,
            **self.cfg.kwargs,
        )


class TruncatedSVD(BaseTransformer):
    """Truncated Singular Value Decomposition.

    See https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    """
    __processor_attrs__ = [
        'singular_values_',
        'explained_variance_',
        'explained_variance_ratio_',
        'components_',
    ]

    class Config(BaseTransformer.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'TruncatedSVD']

        write_key: WriteKey = f"obsm.{OBSM.X_TRUNCATED_SVD}"
        n_components: ProcessorParam[PositiveInt] = 2
        algorithm: ProcessorParam[str] = 'randomized'
        n_iter: ProcessorParam[PositiveInt] = 5

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

        self.processor: _TruncatedSVD = _TruncatedSVD(
            n_components=self.cfg.n_components,
            algorithm=self.cfg.algorithm,
            n_iter=self.cfg.n_iter,
            random_state=self.cfg.seed,
            **self.cfg.kwargs,
        )


class MDS(BaseTransformer):
    """Multidimensional scaling.

    See https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html
    """

    class Config(BaseTransformer.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'MDS']

        write_key: WriteKey = f"obsm.{OBSM.X_MDS}"
        n_components: ProcessorParam[PositiveInt] = 2

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

        self.processor: _MDS = _MDS(
            n_components=self.cfg.n_components,
            random_state=self.cfg.seed,
            **self.cfg.kwargs,
        )


class UMAP(BaseTransformer):
    """Uniform Manifold Approximation and Projection

    See https://umap-learn.readthedocs.io/en/latest/
    """

    class Config(BaseTransformer.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'UMAP']

        x_key: ReadKey = f"obsm.{OBSM.X_PCA}"  # Different x key from parent
        write_key: WriteKey = f"obsm.{OBSM.X_UMAP}"
        n_neighbors: ProcessorParam[PositiveInt] = 15
        n_components: ProcessorParam[PositiveInt] = 2
        # Use a smaller spread by default, for tighter scatterplots
        spread: ProcessorParam[PositiveFloat] = 0.8

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

        self.processor: _UMAP = _UMAP(
            n_components=self.cfg.n_components,
            n_neighbors=self.cfg.n_neighbors,
            spread=self.cfg.spread,
            random_state=self.cfg.seed,
            **self.cfg.kwargs,
        )
