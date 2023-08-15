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
        processor_cfg: BaseProcessor.Config
        repeat_var: str
        repeat_vals: List[Any]

        repeat_prefix: str = 'r-{repeat_var}-{repeat_val}-'
        upstream_splitter: str = '-'

        @field_validator('repeat_var')
        def has_field(cls, val, info):
            if not hasattr(info.data['processor_cfg'], val):
                raise KeyError(
                    f"Processor Config {info.data['processor'].__qualname__} "
                    f"has no key '{val}'."
                )
            return val

    cfg: Config

    def _process(self, adata: AnnData) -> None:
        for repeat_val in self.cfg.repeat_vals:
            logger.info(
                f"Repeating '{self.cfg.processor_cfg._init_cls.__name__}' "
                f"with value {repeat_val}."
            )
            setattr(self.cfg.processor_cfg, self.cfg.repeat_var, repeat_val)
            processor = self.cfg.processor_cfg.create()
            processor.update_prefix(
                self.cfg.repeat_prefix,
                splitter=self.cfg.upstream_splitter,
                repeat_var=self.cfg.repeat_var,
                repeat_val=repeat_val,
            )
            processor(adata)
