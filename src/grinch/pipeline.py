from typing import Dict

from anndata import AnnData
from pydantic import validate_arguments

from .conf import BaseConfigurable
from .group import GroupProcess
from .predictors import BasePredictor
from .splitter import DataSplitter, Splitter
from .transformers import BaseTransformer


class GRPipeline(BaseConfigurable):

    class Config(BaseConfigurable.Config):
        processors: Dict[str, BaseConfigurable.Config]  # Maps a processor name to a config

    cfg: Config

    def __init__(self, cfg: Config, /) -> None:
        super().__init__(cfg)

        self.processors = []

        for c in self.cfg.processors.values():
            if self.cfg.seed is not None:
                c = c.copy(update={'seed': self.cfg.seed})
            self.processors.append(c.initialize())

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __call__(self, adata: AnnData | DataSplitter, *args, **kwargs) -> DataSplitter:
        """Applies processor to the different data splits in DataSplitter.
        It differentiates between predictors (calls processor.predict),
        transformers (calls processor.transform) and it defaults to
        processor.__call__ for all other processors.
        """
        ds = DataSplitter(adata) if not isinstance(adata, DataSplitter) else adata

        for processor in self.processors:
            if not isinstance(processor, Splitter):
                self._apply(ds, processor)
            else:
                # Perform a data split
                ds = processor(ds)

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
