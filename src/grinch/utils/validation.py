def any_not_None(*args):
    """Returns True if any item is not None and False otherwise."""
    for arg in args:
        if arg is not None:
            return True
    return False
