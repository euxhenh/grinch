import logging
import os

import anndata

from .conf import BaseConfigurable

logger = logging.getLogger(__name__)


class DataLoader(BaseConfigurable):

    class Config(BaseConfigurable.Config):
        data_path: str
        write_path: str
        worker: BaseConfigurable.Config

    cfg: Config

    def __call__(self) -> None:
        """Loads data calls processor."""
        adata = anndata.read_h5ad(self.cfg.data_path)
        worker = self.cfg.worker.initialize()

        worker(adata)

        if os.path.exists(self.cfg.write_path):
            logger.warn(f"Object {self.cfg.write_path} exists. Overwriting...")

        adata.write_h5ad(self.cfg.write_path)
