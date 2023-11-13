#!/Users/ehasanaj/mambaforge/envs/m10/bin/python
import argparse
import logging
import os
import warnings

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from rich.logging import RichHandler

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[RichHandler(omit_repeated_times=False, rich_tracebacks=True)],
)
logging.captureWarnings(True)


def instantiate_config(config_name):
    head, tail = os.path.split(config_name)
    config_dir = os.path.join(os.getcwd(), 'conf')
    # context initialization
    with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = hydra.compose(config_name=tail)
    return instantiate(cfg, _convert_='all')


parser = argparse.ArgumentParser(description="DE gene and enrichment toolbox.")
parser.add_argument('conf', metavar='C', type=str, help="path to config file")
parser.add_argument('--pc', action='store_true', default=False)

if __name__ == "__main__":
    args = parser.parse_args()
    cfg = instantiate_config(args.conf)
    if args.pc:
        cfg_obj = OmegaConf.to_yaml(cfg.model_dump())
        logging.info(cfg_obj)
    obj = cfg.create()
    obj()
