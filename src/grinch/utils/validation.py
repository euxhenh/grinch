import logging
from typing import List

from pydantic import validate_arguments

logger = logging.getLogger(__name__)


def any_not_None(*args):
    """Returns True if any item is not None and False otherwise."""
    return sum(arg is not None for arg in args) > 0


@validate_arguments
def pop_args(args: List[str], **kwargs):
    """Pop any arg in args from kwargs. Meant to be used with estimators
    where some (important) arguments are defined explicitly in the Config,
    while others are deferred inside a kwargs block. We pop duplicate
    arguments and raise a warning to avoid confusion.
    """
    for explicit_key in args:
        if kwargs.pop(explicit_key, None) is not None:
            logger.warning(
                f"Popping '{explicit_key}' from kwargs. If you wish"
                " to overwrite this key, pass it directly in the config."
            )
    return kwargs
