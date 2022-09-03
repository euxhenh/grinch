import logging
from typing import Any, List

from anndata import AnnData
from pydantic import validator

from .base_processor import BaseProcessor

logger = logging.getLogger(__name__)


class RepeatProcessor(BaseProcessor):

    class Config(BaseProcessor.Config):
        # NOTE: If the field that is to be repeated is mandatory in the
        # processor config, it needs to be initialized to any value
        # otherwise hydra will complain.
        processor: BaseProcessor.Config
        repeat_var: str
        repeat_vals: List[Any]

        repeat_prefix: str = 'r-{repeat_var}-{repeat_val}-'

        @validator('repeat_var')
        def has_field(cls, val, values):
            if not hasattr(values['processor'], val):
                raise KeyError(
                    f"Processor Config {values['processor'].__qualname__} "
                    f"has no key '{val}'."
                )
            return val

        def update_processor_save_key_prefix(self, repeat_val):
            prefix = self.get_save_key_prefix(
                self.repeat_prefix,
                splitter="-",
                repeat_var=self.repeat_var,
                repeat_val=repeat_val,
            )
            self.processor.save_key_prefix = prefix

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

        self.cfg.processor.inplace = True

    def _process(self, adata: AnnData) -> None:

        for val in self.cfg.repeat_vals:
            logger.info(f"Repeating '{self.cfg.processor.init_type.__name__}' with value {val}.")
            setattr(self.cfg.processor, self.cfg.repeat_var, val)
            self.cfg.update_processor_save_key_prefix(val)
            processor: BaseProcessor = self.cfg.processor.initialize()
            processor(adata)
