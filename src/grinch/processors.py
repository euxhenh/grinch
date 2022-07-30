import abc
import logging
from typing import Any, Optional

from anndata import AnnData
from pydantic import validate_arguments, validator

from .aliases import ALLOWED_KEYS
from .conf import BaseConfigurable

logger = logging.getLogger(__name__)


class BaseProcessor(BaseConfigurable):
    """A base class for dimensionality reduction, clustering and other
    processors. A processor cannot update the data matrix X, but can use it
    to perform any kind of fitting.
    """

    class Config(BaseConfigurable.Config):
        inplace: bool = True
        # Select the representation to use. If 'X', will use adata.X,
        # otherwise it must contain a dot that splits the annotation key
        # that will be used and the column key. E.g., 'obsm.x_emb' will use
        # 'adata.obsm['x_emb']'.
        read_key: Optional[str] = None
        # Key where to store results.
        save_key: Optional[str] = None

        @validator('read_key', 'save_key')
        def rep_format_is_correct(cls, val):
            if val is None or val == 'X':
                return val
            if '.' not in val:
                raise ValueError(
                    "read and save keys must equal 'X' or must contain a "
                    "dot '.' that points to the representation to use."
                )
            if len(parts := val.split('.')) > 2:
                raise ValueError("There can only be one dot '.' in read and save keys.")
            if parts[0] not in ALLOWED_KEYS:
                raise ValueError(f"AnnData annotation key should be one of {ALLOWED_KEYS}.")
            if len(parts[1]) >= 120:
                raise ValueError("Columns keys should be less than 120 characters.")
            return val

        @validator('save_key')
        def ensure_save_key_not_X(cls, val):
            if val == 'X':
                raise ValueError(
                    "'save_key' cannot equal X. Maybe you meant to use a transform instead."
                )
            return val

    cfg: Config

    @property
    def processor(self):
        """Points to the object that is being wrapped by the derived class."""
        return getattr(self, '_processor', None)

    @processor.setter
    def processor(self, value):
        self._processor = value

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __call__(self, adata: AnnData) -> Optional[AnnData]:
        if not self.cfg.inplace:
            adata = adata.copy()

        self._process(adata)

        return adata if not self.cfg.inplace else None

    @abc.abstractmethod
    def _process(self, adata: AnnData) -> None:
        """To be implemented by a derived class."""
        raise NotImplementedError

    def _get_repr(self, adata: AnnData) -> Any:
        """Get the data representation that read_key points to."""
        if self.cfg.read_key is None:
            raise ValueError("Cannot get representation if 'read_key' is None.")

        if self.cfg.read_key == 'X':
            return adata.X

        read_class, read_key = self.cfg.read_key.split('.')
        return getattr(adata, read_class)[read_key]

    def _set_repr(
        self,
        adata: AnnData,
        value: Any,
        save_config: bool = True
    ) -> None:
        """Save value under the key pointed to by save_key. Also saves
        config under `uns` if `save_config` is True.
        """
        if self.cfg.save_key is None:
            raise ValueError("Cannot save representation if 'save_key' is None.")

        save_class, save_key = self.cfg.save_key.split('.')
        try:
            getattr(adata, save_class)[save_key] = value
        except Exception:
            # Try initializing to an empty dictionary on fail
            setattr(adata, save_class, {})
            getattr(adata, save_class)[save_key] = value

        if save_config:
            adata.uns[save_key] = self.cfg.dict()
