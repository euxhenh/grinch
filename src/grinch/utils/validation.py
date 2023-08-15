import logging

from .exceptions import ProcessorNotDefined

logger = logging.getLogger(__name__)


def any_not_None(*args):
    """Returns True if any item is not None and False otherwise.
    Examples
    ________
    >>> any_not_None(1, 2, None)
    True
    >>> any_not_None(None, None)
    False
    """
    return sum(arg is not None for arg in args) > 0


def all_None(*args):
    """Returns True if all are None.
    Examples
    ________
    >>> all_None(1, 2, None)
    False
    >>> all_None(None, None)
    True
    """
    return not any_not_None(*args)


def all_not_None(*args):
    """Returns True if all items are not None.

    Examples
    --------
    >>> all_not_None(None, 1, 2)
    False
    >>> all_not_None(5, "bar")
    True
    """
    return sum(arg is None for arg in args) == 0


def only_one_not_None(*args):
    """Returns True if there is exactly one item that is not None.

    Examples
    --------
    >>> only_one_not_None(None, 1, 'bar')
    False
    >>> only_one_not_None(None, 'foo', None)
    True
    """
    return sum(arg is not None for arg in args) == 1


def check_has_processor(obj):
    """Checks if obj.processor is not None. Raises an error otherwise.
    """
    if obj.processor is None:
        raise ProcessorNotDefined(
            f"Object of type {obj.__class__.__name__} does not "
            "contain a processor object."
        )


def validate_axis(axis):
    """Checks if axis is 0, 1, 'obs', or 'var'.
    """
    if axis in ['obs', 0]:
        return 0
    elif axis in ['var', 1]:
        return 1
    raise ValueError(f"Could not interpret axis={axis}.")
