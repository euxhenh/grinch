import numpy as np
import pytest
from anndata import AnnData
from hydra.utils import instantiate
from omegaconf import OmegaConf

from grinch import UNS, DETestSummary

X = np.array([
    [1, 5, 4, 45, 62],
    [5, 2, 4, 44, 75],
    [5, 2, 5, 44, 75],
    [75, 62, 4, 4, 2],
    [45, 64, 5, 6, 4],
    [45, 64, 4, 6, 4],
], dtype=np.float32)

X_mods = [X]


@pytest.mark.parametrize("X", X_mods)
def test_enrich(X):
    fcfg = OmegaConf.create(
        {
            "_target_": "src.grinch.FilterCondition",
            "key": "pvals",
            "cutoff": 0.05,
        }
    )
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.GSEA.Config",
            "filter_by": [fcfg],
        }
    )
    cfg = instantiate(cfg, _convert_='all')
    gsea = cfg.initialize()
    adata = AnnData(X)
    ts = DETestSummary(pvals=[0.02, 0.5, 1, 0.01, 0.8])
    adata.uns[UNS.TTEST] = ts.df()
    adata.var_names = ['IGKV4-1', 'CD55', 'IGKC', 'PPFIBP1', 'ABHD4']
    gsea(adata)

    results = adata.uns[UNS.GSEA]
    genes = results['Genes'].to_numpy()
    for g in genes:
        gs = g.split(';')
        assert len(gs) > 0
        assert set(gs).issubset(set(['IGKV4-1', 'PPFIBP1']))

    adata = AnnData(X)
    adata.uns[UNS.TTEST] = {}
    adata.uns[UNS.TTEST]['0'] = ts.df()
    adata.var_names = ['IGKV4-1', 'CD55', 'IGKC', 'PPFIBP1', 'ABHD4']
    gsea(adata)

    results = adata.uns[UNS.GSEA]['0']
    genes = results['Genes'].to_numpy()
    for g in genes:
        gs = g.split(';')
        assert len(gs) > 0
        assert set(gs).issubset(set(['IGKV4-1', 'PPFIBP1']))
