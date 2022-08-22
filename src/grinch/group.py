import logging

import anndata
from anndata import AnnData
from pydantic import validator

from .base_processor import BaseProcessor
from .utils.ops import group_indices
from .utils.validation import validate_axis

logger = logging.getLogger(__name__)


class GroupProcess(BaseProcessor):

    class Config(BaseProcessor.Config):
        processor: BaseProcessor.Config
        # Key to group by, must be recognized by np.unique.
        group_key: str
        axis: int | str = 0
        group_prefix: str = 'g-{label}/'

        @validator('axis')
        def ensure_correct_axis(cls, axis):
            return validate_axis(axis)

        @validator('processor')
        def ensure_not_inplace(cls, processor):
            if not processor.inplace:
                logger.warn('Group processor not `inplace` mode will have no effect.')
            return processor

        @staticmethod
        def replace_label(group_prefix, label):
            if "{label}" in group_prefix:
                return group_prefix.format(label=label)
            return group_prefix

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

        # We change this to not inplace so that we can concatenate the
        # resulting adatas into a single adata.
        self.cfg_not_inplace = self.cfg.processor.copy(update={'inplace': False})

    def _process(self, adata: AnnData) -> None:
        # Determine groups to process separately
        group_labels = self.get_repr(adata, self.cfg.group_key)
        unq_labels, groups = group_indices(group_labels)

        adata_list = []
        self.processor_dict = {}

        # TODO multithread
        for label, group in zip(unq_labels, groups):
            cfg = self.cfg_not_inplace.copy(update={
                'save_key_prefix': self.cfg.replace_label(self.cfg.group_prefix, label)
            })
            processor = cfg.initialize()
            if self.cfg.axis == 0:
                _adata = processor(adata[group])
            else:
                _adata = processor(adata[:, group])
            self.processor_dict[label] = processor

            adata_list.append(_adata)

        # Outer join will fill missing values with nan's.
        # uns_merge = same will only merge uns keys which are the same.
        concat_adata = anndata.concat(adata_list, join='outer', uns_merge='same')
        # Reorder so that they have original order
        if self.cfg.axis:
            concat_adata = concat_adata[adata.obs_names]
        else:
            concat_adata = concat_adata[:, adata.var_names]

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
