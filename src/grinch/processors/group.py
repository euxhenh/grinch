import logging
from typing import TYPE_CHECKING, Callable, Literal

from anndata import AnnData
from pydantic import Field, field_validator

from ..aliases import GROUP_SEP
from ..custom_types import NP1D_str
from ..utils.ops import group_indices
from ..utils.validation import validate_axis
from .base_processor import BaseProcessor, ReadKey, WriteKey

logger = logging.getLogger(__name__)


class GroupProcess(BaseProcessor):
    """Processes adata in groups separately. Useful, for example, when one
    has multiple conditions that need to be processed separately.

    Parameters
    __________
    processor: BaseProcessor
        A processor to apply separately to each of the groups. Currently,
        different processors for different groups are not supported.
    group_key: str
        Must point to a 1D vector that will determine the groups.
    axis: int or str
        The axis along which to group.
    group_prefix: str
        If the stats that are stored in anndata for each group should not
        be merged, this key can be used to specify different prefixes for
        the save keys, based on the group label. The substring '{label}'
        will be replaced by the group name.
    min_points_per_group: int
        If a group contains less than this many points, it will not be
        processed.
    drop_small_groups: bool
        If a group contains less than 'min_points_per_group' and this is
        set to True, these groups will not be included in the merged
        anndata.
    """

    class Config(BaseProcessor.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'GroupProcess']

        processor: BaseProcessor.Config
        # Key to group by, must be recognized by np.unique.
        group_key: ReadKey
        axis: int | Literal['obs', 'var'] = Field(0, ge=0, le=1)
        group_prefix: WriteKey = f'g-{{group_key}}{GROUP_SEP}{{label}}.'
        min_points_per_group: int = Field(default_factory=int, ge=0)
        # Whether to drop the groups which have less than
        # `min_points_per_group` points or not.
        drop_small_groups: bool = False

        @field_validator('axis')
        def ensure_correct_axis(cls, axis):
            return validate_axis(axis)

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

        self.cfg.processor.inplace = False

    def update_processor_prefix(self, label):
        return self.insert_prefix(
            self.group_prefix,
            label=label,
            group_key=self.group_key.rsplit('.', maxsplit=1)[-1],
        )

    def _get_names_along_axis(self, adata: AnnData) -> NP1D_str:
        """Gets obs_names or var_names depending on self.cfg.axis."""
        return (
            adata.obs_names.to_numpy().astype(str) if self.cfg.axis == 0
            else adata.var_names.to_numpy().astype(str)
        )

    def _process(self, adata: AnnData) -> None:
        # Determine groups to process separately
        group_labels = self.get_repr(adata, self.cfg.group_key, to_numpy=True)
        if len(group_labels) != adata.shape[self.cfg.axis]:
            raise ValueError(
                "Length of 'group_labels' should "
                "match the dimension of adata."
            )
        unq_labels, groups = group_indices(group_labels)

        # TODO multithread
        for label, group in zip(unq_labels, groups):
            # Determine if this group is small or not
            if self.cfg.drop_small_groups:
                if len(group) < self.cfg.min_points_per_group:
                    logger.info(f"Skipping small group '{label}'.")
                    continue

            logger.info(
                f"Running '{self.cfg.processor._init_cls.__name__}' "
                f"for group '{self.cfg.group_key}={label}'."
            )

            processor: BaseProcessor = self.cfg.processor.create()
            processor.prefix = self.update_processor_prefix(label)
            arg = 'obs_indices' if self.cfg.axis == 0 else 'var_indices'
            storage = processor(adata, return_storage=True, **{arg: group})
            # Prefix should be handled by the processor
            self.store_items(storage, add_prefix=False)
