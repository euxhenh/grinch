import numpy as np
from anndata import AnnData
from pydantic import validator
from sklearn.utils import column_or_1d

from ..utils.validation import validate_axis
from .base_processor import BaseProcessor


class InplaceIndexer(BaseProcessor):
    """Inexes adata over obs or var axis using a mask stored in obs/var or
    an uns list of indices.
    """

    class Config(BaseProcessor.Config):
        mask_key: str
        # If save key is None, will init anndata inplace, otherwise it will
        # store it under key.
        # save_key: Optional[str] = None
        # Can be 0, 1 or 'obs', 'var'
        axis: int | str = 0
        # Can set to False if indexing from some uns list of indices
        as_bool: bool = True

        @validator('axis')
        def ensure_correct_axis(cls, axis):
            return validate_axis(axis)

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

    def _process(self, adata: AnnData) -> None:
        mask = self.get_repr(adata, self.cfg.mask_key)
        mask = column_or_1d(mask)
        if self.cfg.as_bool:
            mask = mask.astype(bool)
        else:
            unq_indices = np.unique(mask)
            if len(unq_indices) != len(mask):
                raise ValueError("Found duplicate IDs in mask.")

        if self.cfg.axis == 0:
            adata._inplace_subset_obs(mask)
        else:
            adata._inplace_subset_var(mask)
