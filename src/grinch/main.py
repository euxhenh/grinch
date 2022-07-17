import os

import hydra
from hydra.utils import instantiate


def instantiate_config(config_name):
    # context initialization
    conf_abs_path = os.path.abspath('conf')
    with hydra.initialize_config_dir(config_dir=conf_abs_path):
        cfg = hydra.compose(config_name=config_name)
    return instantiate(cfg, _convert_='all')


if __name__ == "__main__":
    obj = instantiate_config('preprocessing')
