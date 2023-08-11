import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from hydra.utils import instantiate
from omegaconf import OmegaConf

from grinch import UNS

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
            "_target_": "src.grinch.pVal_Filter_05",
        }
    )
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.GSEAEnrich.Config",
            "filter_by": fcfg,
        }
    )
    cfg = instantiate(cfg, _convert_='all')
    gsea = cfg.create()
    adata = AnnData(X)
    ts = pd.DataFrame(data={'pvals': [0.02, 0.5, 1, 0.01, 0.8]})
    adata.uns[UNS.TTEST] = ts
    adata.var_names = ['IGKV4-1', 'CD55', 'IGKC', 'PPFIBP1', 'ABHD4']
    gsea(adata)

    results = adata.uns[UNS.GSEA_ENRICH]
    genes = results['Genes'].to_numpy()
    for g in genes:
        gs = g.split(';')
        assert len(gs) > 0
        assert set(gs).issubset(set(['IGKV4-1', 'PPFIBP1']))

    adata = AnnData(X)
    adata.uns[UNS.TTEST] = {}
    adata.uns[UNS.TTEST]['0'] = ts
    adata.var_names = ['IGKV4-1', 'CD55', 'IGKC', 'PPFIBP1', 'ABHD4']
    gsea(adata)

    results = adata.uns[UNS.GSEA_ENRICH]['0']
    genes = results['Genes'].to_numpy()
    for g in genes:
        gs = g.split(';')
        assert len(gs) > 0
        assert set(gs).issubset(set(['IGKV4-1', 'PPFIBP1']))


def test_lead_genes():
    fcfg = OmegaConf.create(
        {
            "_target_": "src.grinch.FDRqVal_Filter_05",
        }
    )
    cfg = OmegaConf.create(
        {
            "_target_": "src.grinch.FindLeadGenes.Config",
            "filter_by": fcfg,
        }
    )
    cfg = instantiate(cfg, _convert_='all')
    find_lead = cfg.create()
    adata = AnnData(np.random.random((1, 8)))
    adata.var_names = list('ABCDEFGH')

    df1 = pd.DataFrame(data={
        'FDR q-val': [0.5, 0.4, 0.01, 0.6],
        'Lead_genes': ['B;G', 'A;C', 'H;G', 'A;D']
    })
    df2 = pd.DataFrame(data={
        'FDR q-val': [0.005, 0.4, 0.001, 0.02],
        'Lead_genes': ['A;C;H', 'C', 'G', 'A;G']
    })

    adata.uns['gsea_prerank'] = {
        "cl1": df1,
        "cl2": df2,
    }

    # Leads are np.unique(H, G, A, C, H, G, A, G) = ACGH
    find_lead(adata)
    all_leads = [True, False, True, False, False, False, True, True]
    lead_groups = ["cl2", "", "cl2", "", "", "", "cl1;cl2", "cl1;cl2"]

    assert (adata.var['is_lead'] == all_leads).all()
    assert (adata.var['lead_group'] == lead_groups).all()
