import logging
from typing import Any, Dict, List

from pydantic import validate_arguments

from .exceptions import ProcessorNotDefined

logger = logging.getLogger(__name__)


def any_not_None(*args):
    """Returns True if any item is not None and False otherwise."""
    return sum(arg is not None for arg in args) > 0


@validate_arguments
def pop_args(args: List[str], kwargs: Dict[str, Any]):
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


def check_has_processor(obj):
    """Checks if obj.processor is not None. Raises an error otherwise.
    """
    if obj.processor is None:
        raise ProcessorNotDefined(
            f"Object of type {obj.__class__} does not contain a processor object."
        )
