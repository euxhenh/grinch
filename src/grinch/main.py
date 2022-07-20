import argparse
import os
import warnings

import hydra
from hydra.utils import instantiate

try:
    import grinch
    _src_dir = os.path.dirname(grinch.__file__)
    _root_dir = os.path.join(_src_dir, os.pardir, os.pardir)
except ImportError:
    _this_dir = os.path.dirname(os.path.realpath(__file__))
    _root_dir = os.path.join(_this_dir, os.pardir, os.pardir)

if 'conf' in os.listdir(_root_dir):
    CONF_DIR = os.path.join(_root_dir, 'conf')
else:
    warnings.warn('Could not find default conf directory.')
    CONF_DIR = '.'

CONF_DIR = os.path.abspath(CONF_DIR)


def instantiate_config(config_name):
    # context initialization
    with hydra.initialize_config_dir(config_dir=CONF_DIR):
        cfg = hydra.compose(config_name=config_name)
    return instantiate(cfg, _convert_='all')


parser = argparse.ArgumentParser(description="DE gene and enrichment toolbox.")
parser.add_argument('conf', metavar='C', type=str, help="path to config file")

if __name__ == "__main__":
    args = parser.parse_args()
    obj = instantiate_config(args.conf)
