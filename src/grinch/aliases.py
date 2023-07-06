class auto:
    """Initialize a variable with a lowercase string representation of its
    name."""

    def __set_name__(self, owner, name):
        self.value = name.lower()

    def __get__(self, obj, objtype=None):
        return self.value


class AnnDataKeys:

    class OBS:
        N_COUNTS = auto()
        N_GENES = auto()
        LABEL = auto()
        KMEANS = auto()
        LEIDEN = auto()
        LOG_REG = auto()

    class VAR:
        N_COUNTS = auto()
        N_CELLS = auto()
        PCOVER_M = auto()
        PCOVER_I = auto()
        VARIANCE = auto()
        ENSEMBL_ID = auto()
        FEATURE_NAME = auto()
        IS_LEAD = auto()
        LEAD_GROUP = auto()

    class OBSM:
        X_EMB = auto()
        X_EMB_2D = auto()
        X_PCA = auto()
        X_TRUNCATED_SVD = auto()
        X_UMAP = auto()

    class VARM:
        LOG_REG_COEF = auto()

    class UNS:
        X_PCA_ = auto()
        X_TRUNCATED_SVD_ = auto()
        KMEANS_ = auto()
        LEIDEN_ = auto()
        LOG_REG_ = auto()
        TTEST = auto()
        KSTEST = auto()
        GSEA_ENRICH = auto()
        GSEA_PRERANK = auto()
        BIMODALTEST = auto()
        PCOVER_ = auto()
        KNN_ = auto()
        N_GENE_ID_TO_NAME_FAILED = auto()

    class OBSP:
        KNN_CONNECTIVITY = auto()
        KNN_DISTANCE = auto()
        UMAP_CONNECTIVITY = auto()
        UMAP_DISTANCE = auto()
        UMAP_AFFINITY = auto()

    class VARP:
        ...


# Create shorter aliases, since these will be used a lot
ADK = AnnDataKeys
OBS = AnnDataKeys.OBS
VAR = AnnDataKeys.VAR
OBSM = AnnDataKeys.OBSM
VARM = AnnDataKeys.VARM
OBSP = AnnDataKeys.OBSP
VARP = AnnDataKeys.VARP
UNS = AnnDataKeys.UNS

ALLOWED_KEYS = ['obs', 'obsm', 'uns', 'var', 'varm', 'layers', 'obsp', 'varp']

GROUP_SEP = '.'
