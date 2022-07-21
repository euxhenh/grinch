import abc
from typing import Any, Optional

from anndata import AnnData
from pydantic import ValidationError, validate_arguments, validator
from sklearn.decomposition import PCA as _PCA
from sklearn.decomposition import TruncatedSVD as _TruncatedSVD

from .aliases import OBSM
from .conf import BaseConfigurable


class BaseProcessor(BaseConfigurable):
    """A base class for dimensionality reduction, clustering and other
    processor methods.
    """

    class Config(BaseConfigurable.Config):
        inplace: bool = True
        # Select the representation to use. If 'X', will use adata.X,
        # otherwise it must contain a dot that points to the annotation key
        # that will be used. E.g., 'obsm.x_emb' will use
        # 'adata.obsm['x_emb']'.
        read_key: Optional[str] = None
        # Key where to store results.
        save_key: Optional[str] = None

        @validator('read_key')
        def rep_format_is_correct(cls, val):
            if val is None or val == 'X':
                return
            if '.' not in val:
                raise ValidationError(
                    "read and save keys must equal 'X' or must contain a "
                    "dot '.' that points to the representation to use."
                )
            if len(parts := val.split('.')) > 2:
                raise ValidationError(
                    "There can only be one dot '.' in read and save keys."
                )
            if parts[0] not in (allowed_keys := [
                'obs', 'obsm', 'uns', 'var', 'varm', 'layers',
            ]):
                raise ValidationError(
                    f"AnnData annotation key should be one of {allowed_keys}."
                )

        @validator('save_key')
        def ensure_save_key_not_X(cls, val):
            if val == 'X':
                raise ValidationError(
                    "'save_key' cannot equal X. Maybe you meant to use a "
                    "transform instead."
                )

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

    @property
    def obj(self):
        """Points to the object that is being wrapped by the derived class."""
        return getattr(self, '_obj', None)

    @obj.setter
    def obj(self, value):
        self._obj = value

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __call__(self, adata: AnnData) -> Optional[AnnData]:
        if not self.cfg.inplace:
            adata = adata.copy()

        self._process(adata)

        return adata if not self.cfg.inplace else None

    @abc.abstractmethod
    def _process(self, adata: AnnData) -> None:
        raise NotImplementedError

    def _get_repr(self, adata: AnnData) -> Any:
        """Get the data representation that read_key points to."""
        if self.cfg.read_key is None:
            raise ValueError("Cannot get representation as 'read_key' is None.")

        if self.cfg.read_key == 'X':
            return adata.X

        read_class, read_key = self.cfg.read_key.split('.')
        return getattr(adata, read_class)[read_key]

    def _set_repr(self, adata: AnnData, value: Any, save_config: bool = True) -> None:
        """Save value under the key pointed to by save_key. Also saves
        config under `uns` if `save_config` is True.
        """
        if self.cfg.save_key is None:
            raise ValueError("Cannot save representation as 'save_key' is None.")

        save_class, save_key = self.cfg.save_key.split('.')
        try:
            getattr(adata, save_class)[save_key] = value
        except Exception:
            # Try initializing to an empty dictionary on fail
            setattr(adata, save_class, {})
            getattr(adata, save_class)[save_key] = value
        adata.uns[save_key] = self.cfg.dict()


class BaseEstimator(BaseProcessor, abc.ABC):
    """A base estimator class for objects that implement `fit_transform`."""

    class Config(BaseProcessor.Config):
        ...

    def _process(self, adata: AnnData) -> None:
        x_rep = self._get_repr(adata)
        x_rep_out = self.obj.fit_transform(x_rep)
        self._set_repr(adata, x_rep_out)


class PCA(BaseEstimator):

    class Config(BaseEstimator.Config):
        read_key: str = "X"
        save_key: str = f"obsm.{OBSM.X_PCA}"
        # PCA args
        n_components: Optional[int | float | str] = None
        whiten: bool = False
        svd_solver: str = 'auto'

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

        # Types here are useful for editor autocompletion
        self.obj: _PCA = _PCA(
            n_components=self.cfg.n_components,
            whiten=self.cfg.whiten,
            svd_solver=self.cfg.svd_solver,
            random_state=self.cfg.seed,
        )


class TruncatedSVD(BaseEstimator):

    class Config(BaseEstimator.Config):
        read_key: str = "X"
        save_key: str = f"obsm.{OBSM.X_TRUNCATED_SVD}"
        # PCA args
        n_components: int = 2
        algorithm: str = 'randomized'
        n_iter: int = 5

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

        self.obj: _TruncatedSVD = _TruncatedSVD(
            n_components=self.cfg.n_components,
            algorithm=self.cfg.algorithm,
            n_iter=self.cfg.n_iter,
            random_state=self.cfg.seed,
        )
