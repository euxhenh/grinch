import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from anndata import AnnData
from pydantic import Field, validate_arguments, validator
from sklearn.model_selection import train_test_split

from ..conf import BaseConfigurable
from ..utils.validation import all_not_None, any_not_None
from .base_processor import BaseProcessor

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class DataSplitter:
    TRAIN_SPLIT: AnnData = None
    VAL_SPLIT: Optional[AnnData] = None
    TEST_SPLIT: Optional[AnnData] = None

    @property
    def is_split(self) -> bool:
        return any_not_None(self.VAL_SPLIT, self.TEST_SPLIT)

    def write_h5ad(self, path: str) -> None:
        """Writes anndata to path. If any of VAL or TEST splits are not
        None, will instead write both to a folder with the name specified
        in path.
        """
        if not any_not_None(self.VAL_SPLIT, self.TEST_SPLIT):
            self.TRAIN_SPLIT.write_h5ad(path)
            return

        if path.endswith('.h5ad'):
            path = path.rsplit('.', 1)[0]
        os.makedirs(path, exist_ok=True)

        for split in ['TRAIN_SPLIT', 'VAL_SPLIT', 'TEST_SPLIT']:
            if (sp := getattr(self, split)) is not None:
                path_to_write = os.path.join(path, split + '.h5ad')
                if os.path.exists(path_to_write):
                    logger.warning(f"Object {path_to_write} exists. This will be overwritten.")
                sp.write_h5ad(path_to_write)


class Splitter(BaseConfigurable):
    """A class for train/validation/test splitting the data."""

    class Config(BaseConfigurable.Config):
        val_fraction: Optional[float] = Field(None, gt=0, lt=1)
        test_fraction: Optional[float] = Field(None, gt=0, lt=1)
        shuffle: bool = True
        stratify_key: Optional[str] = None

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            if all_not_None(self.val_fraction, self.test_fraction):
                if self.val_fraction + self.test_fraction >= 1:
                    raise ValueError("Val and test fraction should sum less than one.")

        @validator('stratify_key')
        def rep_format_is_correct(cls, val):
            return BaseProcessor._validate_single_rep_key(val) if val is not None else val

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __call__(self, adata: AnnData | DataSplitter) -> DataSplitter:
        """Splits the adata into train/validation/test subsets and return a
        DataSplitter. If adata is already a DataSplitter, will raise an
        error if one of VAL_SPLIT or TEST_SPLIT is not None.
        """
        if isinstance(adata, DataSplitter):
            if adata.is_split:
                raise ValueError("Data has been split already.")
            adata = adata.TRAIN_SPLIT

        if not any_not_None(self.cfg.val_fraction, self.cfg.test_fraction):
            return DataSplitter(adata)

        train_idx = np.arange(adata.shape[0])
        stratify = (
            None if self.cfg.stratify_key is None
            else BaseProcessor.get_repr(adata, self.cfg.stratify_key)
        )
        val_frac, test_frac = self.cfg.val_fraction, self.cfg.test_fraction

        if all_not_None(val_frac, test_frac):
            # Update test frac to reflect original fraction after validation split.
            test_frac = test_frac / (1 - val_frac)  # type: ignore

        data_splitter = DataSplitter()
        if val_frac is not None:
            train_idx, val_idx = train_test_split(
                train_idx,
                test_size=val_frac,
                random_state=self.cfg.seed,
                shuffle=self.cfg.shuffle,
                stratify=stratify,
            )
            data_splitter.VAL_SPLIT = adata[val_idx]
            stratify = stratify[train_idx] if stratify is not None else None  # type: ignore

        if self.cfg.test_fraction is not None:
            train_idx, test_idx = train_test_split(
                train_idx,
                test_size=test_frac,
                random_state=self.cfg.seed,
                shuffle=self.cfg.shuffle,
                stratify=stratify,
            )
            data_splitter.TEST_SPLIT = adata[test_idx]

        data_splitter.TRAIN_SPLIT = adata[train_idx]

        return data_splitter
