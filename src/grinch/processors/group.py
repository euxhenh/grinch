import gc
import logging
from typing import List

import anndata
from anndata import AnnData
from pydantic import Field, validator

from ..custom_types import NP1D_str
from ..utils.ops import group_indices, order_by
from ..utils.validation import validate_axis
from .base_processor import BaseProcessor

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
        set to True, these groups will not be included in the merged anndata.
    """

    class Config(BaseProcessor.Config):
        processor: BaseProcessor.Config
        # Key to group by, must be recognized by np.unique.
        group_key: str
        axis: int | str = Field(0, ge=0, le=1, regex='^(obs|var)$')
        group_prefix: str = 'g-{group_key}.{label}.'
        min_points_per_group: int = Field(default_factory=int, ge=0)
        # Whether to drop the groups which have less than
        # `min_points_per_group` points or not.
        drop_small_groups: bool = False

        @validator('axis')
        def ensure_correct_axis(cls, axis):
            return validate_axis(axis)

        def update_processor_save_key_prefix(self, label):
            self.processor.save_key_prefix = self.get_save_key_prefix(
                self.group_prefix,
                label=label,
                group_key=self.group_key.rsplit('.', maxsplit=1)[-1],
            )

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

        self.cfg.processor.inplace = False

    def _get_names_along_axis(self, adata: AnnData) -> NP1D_str:
        """Gets obs_names or var_names depending on self.cfg.axis."""
        return (
            adata.obs_names.to_numpy().astype(str) if self.cfg.axis == 0
            else adata.var_names.to_numpy().astype(str)
        )

    def _process(self, adata: AnnData) -> None:
        # We will use the index to reorder adata, so these have to be unique
        if self.cfg.axis == 0 and not adata.obs.index.is_unique:
            adata.obs_names_make_unique()
        elif self.cfg.axis == 1 and not adata.var.index.is_unique:
            adata.var_names_make_unique()

        # Determine groups to process separately
        group_labels = self.get_repr(adata, self.cfg.group_key)
        if len(group_labels) != adata.shape[self.cfg.axis]:
            raise ValueError("Length of 'group_labels' should match the dimension of adata.")
        unq_labels, groups = group_indices(group_labels)

        adata_list: List[AnnData] = []  # Will hold group adatas without their data matrices.

        # TODO multithread
        for label, group in zip(unq_labels, groups):
            logger.info(
                f"Running '{self.cfg.processor.init_type.__name__}' for group '{label}'."
            )
            self.cfg.update_processor_save_key_prefix(label)
            processor: BaseProcessor = self.cfg.processor.initialize()
            _adata = adata[group] if self.cfg.axis == 0 else adata[:, group]

            # Determine if this group is small or not
            if len(group) < self.cfg.min_points_per_group:
                if not self.cfg.drop_small_groups:
                    adata_list.append(_adata)
                continue

            _adata = processor(_adata, no_data_matrix=True)
            adata_list.append(_adata)

        # Outer join will fill missing values with nan's.
        # uns_merge = same will only merge uns keys which are the same.
        concat_adata = anndata.concat(adata_list, join='outer', uns_merge='first')
        # Reorder so that obs or vars have original order
        original_order = self._get_names_along_axis(adata)

        if concat_adata.shape != adata.shape:
            # Since some adatas may have been dropped, we only take obs and
            # vars which exist in concat adata.
            concat_names = self._get_names_along_axis(concat_adata)
            original_order = order_by(concat_names, original_order, unique_x=True)
            X = adata[original_order].X if self.cfg.axis == 0 else adata[:, original_order].X
            logger.info(f"Dropping {len(original_order) - len(concat_names)} points.")
        else:
            # doesn't create view
            X = adata.X

        concat_adata = (
            concat_adata[original_order] if self.cfg.axis == 0
            else concat_adata[:, original_order]
        )

        adata._init_as_actual(
            # concat_adata has no data matrix, so we take this from adata
            X=X,
            obs=concat_adata.obs,
            var=concat_adata.var,
            uns=concat_adata.uns,
            obsm=concat_adata.obsm,
            varm=concat_adata.varm,
            varp=concat_adata.varp,
            obsp=concat_adata.obsp,
            raw=concat_adata.raw,
            dtype=concat_adata.X.dtype,
            layers=concat_adata.layers,
            filename=concat_adata.filename,
        )

        del concat_adata, adata_list, _adata
        gc.collect()
