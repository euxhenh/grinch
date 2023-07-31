import logging
import time

import matplotlib.pyplot as plt
from pydantic import validate_call

_logger = logging.getLogger(__name__)


def retry(
    retries: int,
    msg: str | None = None,
    sleep: float = 0,
    verbose: bool = True,
    logger: logging.Logger = _logger,
):
    """Returns a decorator that will rerun f n times if msg
    is found in the Exception. If msg is None, will rerun on
    any raise Exception.
    """
    def _decorator(f):
        def _wrapper(*args, **kwargs):
            for i in range(retries + 1):
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    if i < retries and (msg is None or msg in str(e)):
                        if verbose:
                            logger.info(f"{str(e)}. Retry {i + 1}/{retries}.")
                        time.sleep(sleep)
                        continue
                    else:
                        raise e
                break
        return _wrapper
    return _decorator


@validate_call
def plt_interactive(save_path: str | None = None, **kwargs):
    """Returns a decorator that activates plt interactive mode inside function.
    """
    def _decorator(f):
        def _wrapper(self, *args, **kwargs):
            plt.ion()
            f(self, *args, **kwargs)
            plt.ioff()

            if save_path is not None:
                # Set good defaults
                kwargs.setdefault('dpi', 300)
                kwargs.setdefault('bbox_inches', 'tight')
                kwargs.setdefault('transparent', True)
                plt.savefig(save_path, **kwargs)

            plt.clf()
            plt.close()
        return _wrapper
    return _decorator
