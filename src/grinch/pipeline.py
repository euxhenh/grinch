import gc
import logging
import traceback
from os.path import expanduser
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List

import anndata
import scanpy as sc
from anndata import AnnData
from pydantic import Field, FilePath, field_validator, validate_call
from tqdm.auto import tqdm

from .base import StorageMixin
from .conf import BaseConfigurable
from .processors import (
    BasePredictor,
    BaseTransformer,
    DataSplitter,
    GroupProcess,
    Splitter,
    WriteKey,
)

logger = logging.getLogger(__name__)


class ReadMixin:
    """Mixin class for reading data files."""

    @staticmethod
    def read(filepath: FilePath) -> AnnData:
        """Reads AnnData from filepath"""
        if filepath.suffix == '.h5':
            return sc.read_10x_h5(filepath)
        return anndata.read(filepath)


class MultiRead(BaseConfigurable, ReadMixin):
    """Reads multiple adatas and concatenates them."""

    class Config(BaseConfigurable.Config):
        """MultiRead.Config

        Parameters
        ----------
        data_readpath: Dict
            Maps the ID of a dataset to the path of the AnnData.
        id_key: str
            The ID will be stored as a key under `id_key` if not None.
        [obs|var]_names_make_unique: bool
            If True, will make the corresponding axis labels unique.
        kwargs: Dict
            Arguments to pass to `concat`.
        """

        if TYPE_CHECKING:
            create: Callable[..., 'MultiRead']

        paths: Dict[str, FilePath] = {}
        id_key: WriteKey | None = 'obs.batch_ID'
        obs_names_make_unique: bool = True
        var_names_make_unique: bool = True
        kwargs: Dict[str, Any] = {}

        @field_validator('paths', mode='before')
        def expand_paths(cls, val):
            return {k: expanduser(v) for k, v in val.items()}

    cfg: Config

    def __call__(self) -> AnnData:
        adatas = []
        for idx, readpath in self.cfg.paths.items():
            logger.info(f"Reading AnnData from '{readpath}'...")
            adata = self.read(readpath)
            if self.cfg.obs_names_make_unique:
                adata.obs_names_make_unique()
            if self.cfg.var_names_make_unique:
                adata.var_names_make_unique()
            if self.cfg.id_key is not None:
                StorageMixin.write(adata, self.cfg.id_key, idx)
            adatas.append(adata)
        adata = anndata.concat(adatas, **self.cfg.kwargs)
        del adatas
        gc.collect()
        adata.obs_names_make_unique()
        return adata


class GRPipeline(BaseConfigurable, ReadMixin):

    class Config(BaseConfigurable.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'GRPipeline']

        # FilePath ensures file exists
        data_readpath: FilePath | MultiRead.Config | None = None
        data_writepath: Path | None = None
        processors: List[BaseConfigurable.Config]
        verbose: bool = Field(True, exclude=True)
        write_key: str = "pipeline"
        # It may be desirable to write only the columns of adata without
        # the data matrix so save memory. In that case, set no_data_write
        # to True. This will replace the data matrix with a sparse matrix
        # of all zeros.
        no_data_write: bool = False

        @field_validator('data_readpath', 'data_writepath', mode='before')
        def expand_paths(cls, val):
            if not isinstance(val, MultiRead.Config):
                return expanduser(val) if val is not None else None
            return val

    cfg: Config

    def __init__(self, cfg: Config, /) -> None:
        super().__init__(cfg)
        self.processors = []

        for c in self.cfg.processors:
            if self.cfg.seed is not None:
                c.seed = self.cfg.seed
            self.processors.append(c.create())

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __call__(self, adata: AnnData | None = None, **kwargs) -> DataSplitter:
        """Applies processor to the different data splits in DataSplitter.
        It differentiates between predictors (calls processor.predict),
        transformers (calls processor.transform) and it defaults to
        processor.__call__ for all other processors.
        """
        if adata is None:
            if self.cfg.data_readpath is None:
                raise ValueError("A path to adata or an adata object is required.")
            if isinstance(self.cfg.data_readpath, MultiRead.Config):
                multi_read = self.cfg.data_readpath.create()
                adata = multi_read()
            else:
                logger.info(f"Reading AnnData from '{self.cfg.data_readpath}'...")
                adata = self.read(self.cfg.data_readpath)
        logger.info(adata)
        ds = DataSplitter(adata) if not isinstance(adata, DataSplitter) else adata

        it = tqdm(self.processors) if self.cfg.verbose else self.processors
        for processor in it:
            logger.info(f"Running '{processor.__class__.__name__}'.")
            try:
                if not isinstance(processor, Splitter):
                    self._apply(ds, processor)
                else:
                    ds = processor(ds)  # Perform a data split
            except Exception:
                print(traceback.format_exc())
                logger.warning("Error occured in pipeline.")
                if self.cfg.data_writepath is not None:
                    logger.warning("Saving anndata columns under 'data/_incomp.h5ad'.")
                    ds.write_h5ad('data/_incomp.h5ad', no_data_write=True)
                else:
                    logger.warning("Returning incomplete adata.")
                return ds

        ds.TRAIN_SPLIT.uns[self.cfg.write_key] = self.cfg.model_dump_json()
        if self.cfg.data_writepath is not None:
            logger.info(f"Writting AnnData at '{self.cfg.data_writepath}'...")
            ds.write_h5ad(str(self.cfg.data_writepath),
                          no_data_write=self.cfg.no_data_write)
        return ds

    def _apply(self, ds: DataSplitter, processor: BaseConfigurable) -> None:
        if not callable(processor):
            raise ValueError("Processor is not callable.")

        # main processor call
        processor(ds.TRAIN_SPLIT)

        if ds.VAL_SPLIT is not None:
            match processor:
                case BasePredictor():
                    processor.predict(ds.VAL_SPLIT)
                case BaseTransformer():
                    processor.transform(ds.VAL_SPLIT)
                case Splitter():
                    raise TypeError("Splitters cannot be used with the validation set.")
                case GroupProcess():
                    raise TypeError("GroupProcess cannot be used with the validation set.")
                case _:  # default to simple processing
                    processor(ds.VAL_SPLIT)
