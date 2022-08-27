import logging
from typing import Optional

import anndata
import numpy as np
from anndata import AnnData
from pydantic import Field, validator

from ..custom_types import NP1D_str
from ..utils.ops import group_indices
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
        axis: int | str = 0
        group_prefix: str = 'g-{label}/'
        min_points_per_group: Optional[int] = Field(None, ge=0)
        # Whether to drop the groups which have less than
        # `min_points_per_group` points or not.
        drop_small_groups: bool = False

        @validator('axis')
        def ensure_correct_axis(cls, axis):
            return validate_axis(axis)

        @validator('processor')
        def ensure_not_inplace(cls, processor):
            if not processor.inplace:
                logger.warning('Group processor not `inplace` mode will have no effect.')
            return processor

        @staticmethod
        def replace_label(group_prefix, label):
            return group_prefix.format(label=label) if "{label}" in group_prefix else group_prefix

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

        # We change this to not inplace so that we can concatenate the
        # resulting adatas into a single adata.
        self.cfg_not_inplace = self.cfg.processor.copy(update={'inplace': False})

    def _get_names_along_axis(self, adata: AnnData) -> NP1D_str:
        """Gets obs_names or var_names depending on self.cfg.axis."""
        return (
            adata.obs_names.to_numpy().astype(str) if self.cfg.axis == 0
            else adata.var_names.to_numpy().astype(str)
        )

    def _process(self, adata: AnnData) -> None:
        # Determine groups to process separately
        group_labels = self.get_repr(adata, self.cfg.group_key)
        if len(group_labels) != adata.shape[self.cfg.axis]:
            raise ValueError("Length of 'group_labels' should match the dimension of adata.")
        unq_labels, groups = group_indices(group_labels)

        adata_list = []  # Will hold all group adatas.
        self.processor_dict = {}  # Maps a group name to the fitted processor for that group.

        # TODO multithread
        for label, group in zip(unq_labels, groups):
            cfg = self.cfg_not_inplace.copy(update={
                'save_key_prefix': self.cfg.replace_label(self.cfg.group_prefix, label)
            })
            processor = cfg.initialize()
            _adata = adata[group] if self.cfg.axis == 0 else adata[:, group]

            # Determine if this group is small or not
            if self.cfg.min_points_per_group is not None:
                if len(group) < self.cfg.min_points_per_group:
                    if not self.cfg.drop_small_groups:
                        adata_list.append(_adata)
                    continue

            _adata = processor(_adata)
            # Save the fitted processor
            self.processor_dict[label] = processor
            adata_list.append(_adata)

        # Outer join will fill missing values with nan's.
        # uns_merge = same will only merge uns keys which are the same.
        concat_adata = anndata.concat(adata_list, join='outer', uns_merge='same')
        # Reorder so that obs or vars have original order
        names_to_keep = self._get_names_along_axis(adata)

        if concat_adata.shape != adata.shape:
            # Since some adatas may have been dropped, we only take obs and
            # vars which exist in concat adata.
            concat_names = self._get_names_along_axis(concat_adata)
            names_to_keep = names_to_keep[np.isin(names_to_keep, concat_names)]

        concat_adata = (
            concat_adata[names_to_keep] if self.cfg.axis == 0
            else concat_adata[:, names_to_keep]
        )

        adata._init_as_actual(
            X=concat_adata.X,
            obs=concat_adata.obs,
            var=concat_adata.var,
            uns=concat_adata.uns,
            obsm=concat_adata.obsm,
            varm=concat_adata.varm,
            varp=concat_adata.varp,
            obsp=concat_adata.obsp,
            raw=concat_adata.raw,
            layers=concat_adata.layers,
            filename=concat_adata.filename,
        )
