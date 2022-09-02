import scipy.sparse as sp
from anndata import AnnData


def as_empty(adata: AnnData) -> AnnData:
    # Returns a copy of adata without the data matrix X and without layers
    return AnnData(
        X=sp.csr_matrix(adata.shape),
        obs=adata.obs,
        var=adata.var,
        obsm=adata.obsm,
        varm=adata.varm,
        uns=adata.uns,
        raw=adata.raw,
        dtype=adata.X.dtype,
        filename=adata.filename,
        obsp=adata.obsp,
        varp=adata.varp,
    )
