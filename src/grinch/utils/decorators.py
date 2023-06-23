import logging
import time

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
