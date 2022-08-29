import logging
from typing import List, Optional

import anndata
from anndata import AnnData
from pydantic import Field, validate_arguments
from tqdm.auto import tqdm

from .conf import BaseConfigurable
from .processors import (
    BasePredictor,
    BaseTransformer,
    DataSplitter,
    GroupProcess,
    Splitter,
)

logger = logging.getLogger(__name__)


class GRPipeline(BaseConfigurable):

    class Config(BaseConfigurable.Config):
        data_readpath: Optional[str]
        data_writepath: Optional[str]
        processors: List[BaseConfigurable.Config]
        verbose: bool = Field(True, exclude=True)

    cfg: Config

    def __init__(self, cfg: Config, /) -> None:
        super().__init__(cfg)

        self.processors = []

        for c in self.cfg.processors:
            if self.cfg.seed is not None:
                c = c.copy(update={'seed': self.cfg.seed})
            self.processors.append(c.initialize())

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __call__(self, adata: Optional[AnnData], *args, **kwargs) -> DataSplitter:
        """Applies processor to the different data splits in DataSplitter.
        It differentiates between predictors (calls processor.predict),
        transformers (calls processor.transform) and it defaults to
        processor.__call__ for all other processors.
        """
        if adata is None:
            if self.cfg.data_readpath is None:
                raise ValueError("A path to adata or an adata object is required.")
            adata = anndata.read_h5ad(self.cfg.data_readpath)
        ds = DataSplitter(adata) if not isinstance(adata, DataSplitter) else adata

        it = tqdm(self.processors) if self.cfg.verbose else self.processors
        for processor in it:
            logger.info(f"Running processor {processor.__class__.__name__}")
            if not isinstance(processor, Splitter):
                self._apply(ds, processor)
            else:
                # Perform a data split
                ds = processor(ds)

        if self.cfg.data_writepath is not None:
            ds.write_h5ad(self.cfg.data_writepath)
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
