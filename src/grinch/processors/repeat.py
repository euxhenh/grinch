import logging
from typing import TYPE_CHECKING, Any, Callable, List

from anndata import AnnData
from pydantic import field_validator

from .base_processor import BaseProcessor

logger = logging.getLogger(__name__)


class RepeatProcessor(BaseProcessor):

    class Config(BaseProcessor.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'RepeatProcessor']

        # NOTE: If the field that is to be repeated is mandatory in the
        # processor config, it needs to be initialized to any value
        # otherwise hydra will complain.
        processor: BaseProcessor.Config
        repeat_var: str
        repeat_vals: List[Any]

        repeat_prefix: str = 'r-{repeat_var}-{repeat_val}-'
        upstream_splitter: str = '-'

        @field_validator('repeat_var')
        def has_field(cls, val, info):
            if not hasattr(info.data['processor'], val):
                raise KeyError(
                    f"Processor Config {info.data['processor'].__qualname__} "
                    f"has no key '{val}'."
                )
            return val

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

        self.cfg.processor.inplace = True

    def update_processor_prefix(self, repeat_val) -> str:
        return self.insert_prefix(
            self.repeat_prefix,
            splitter=self.upstream_splitter,
            repeat_var=self.repeat_var,
            repeat_val=repeat_val,
        )

    def _process(self, adata: AnnData) -> None:

        for val in self.cfg.repeat_vals:
            logger.info(f"Repeating '{self.cfg.processor._init_cls.__name__}' with value {val}.")
            setattr(self.cfg.processor, self.cfg.repeat_var, val)
            processor = self.cfg.processor.create()
            processor.prefix = self.update_processor_prefix(val)
            processor(adata)
