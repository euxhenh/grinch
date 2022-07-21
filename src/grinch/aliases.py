class auto:
    """Initialize a variable with a lowercase string representation
    of its variable name.
    """

    def __set_name__(self, owner, name):
        self.value = name.lower()

    def __get__(self, obj, objtype=None):
        return self.value


class AnnDataKeys:
    class OBS:
        N_COUNTS = auto()
        N_GENES = auto()
        LABEL = auto()

    class VAR:
        N_COUNTS = auto()
        N_CELLS = auto()

    class OBSM:
        X_EMB = auto()
        X_EMB_2D = auto()
        X_PCA = auto()
        X_TRUNCATED_SVD = auto()
        X_UMAP = auto()

    class VARM:
        ...


# Create shorter aliases, since these will be used a lot
ADK = AnnDataKeys
OBS = AnnDataKeys.OBS
VAR = AnnDataKeys.VAR
OBSM = AnnDataKeys.OBSM
VARM = AnnDataKeys.VARM
