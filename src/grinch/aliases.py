class auto:
    """Initialize a variable with a lowercase string representation
    of its variable name.
    """
    def __set_name__(self, owner, name):
        self.value = name.lower()

    def __get__(self, obj, objtype=None):
        return self.value


class AnnDataKeys:
    N_COUNTS = auto()
    N_GENES = auto()
    N_CELLS = auto()

    LABEL = auto()
    X_EMB = auto()
    X_EMB_2D = auto()


# Create a shorter alias, since this will be used a lot
ADK = AnnDataKeys
