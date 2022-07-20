import abc
from typing import Any, Optional

import numpy as np
from anndata import AnnData
from pydantic import ValidationError, validate_arguments, validator

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
        repr_key: Optional[str] = None
        # Key where to store results.
        save_key: Optional[str] = None

        @validator('repr_key', 'save_key')
        def rep_format_is_correct(cls, val):
            if val is None or val == 'X':
                return
            if '.' not in val:
                raise ValidationError(
                    "repr and save keys must equal 'X' or must contain a "
                    "dot '.' that points to the representation to use."
                )
            if len(parts := val.split('.')) > 2:
                raise ValidationError(
                    "There can only be one dot '.' in repr and save keys."
                )
            if parts[0] not in (allowed_keys := [
                'obs', 'obsm', 'uns', 'var', 'varm', 'layers',
            ]):
                raise ValidationError(
                    f"AnnData annotation key should be one of {allowed_keys}."
                )

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

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
        """Get the data representation that repr_key points to."""
        if self.cfg.repr_key is None:
            raise ValueError("Cannot get representation as 'repr_key' is None.")

        if self.cfg.repr_key == 'X':
            return adata.X
        repr_class, repr_key = self.cfg.repr_key.split('.')
        return getattr(adata, repr_class)[repr_key]

    def _set_repr(self, adata: AnnData, value: Any) -> None:
        """Save value under the key pointed to by save_key."""
        if self.cfg.save_key is None:
            raise ValueError("Cannot save representation as 'save_key' is None.")

        if self.cfg.save_key == 'X':
            adata.X = value
        else:
            save_class, save_key = self.cfg.save_key.split('.')
            try:
                getattr(adata, save_class)[save_key] = value
            except Exception:
                # Try initializing to an empty dictionary on fail
                setattr(adata, save_class, {})
                getattr(adata, save_class)[save_key] = value


class PCA(BaseProcessor):

    class Config(BaseProcessor.Config):
        ...
