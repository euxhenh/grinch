"""Useful defaults for importing directly in yaml's.
"""
from functools import partial

from .cond_filter import Filter

pVal_Filter_01 = partial(Filter, key='pvals', le=0.01)
pVal_Filter_05 = partial(Filter, key='pvals', le=0.05)
qVal_Filter_01 = partial(Filter, key='qvals', le=0.01)
qVal_Filter_05 = partial(Filter, key='qvals', le=0.05)
log2fc_Filter_1 = partial(Filter, key='log2fc', ge=1)
log2fc_Filter_m1 = partial(Filter, key='log2fc', le=-1)
log2fc_Filter_2 = partial(Filter, key='log2fc', ge=2)
log2fc_Filter_m2 = partial(Filter, key='log2fc', le=-2)
abs_log2fc_Filter_1 = partial(Filter, key='abs_log2fc', ge=1)
abs_log2fc_Filter_2 = partial(Filter, key='abs_log2fc', ge=2)
# For lead gene discovery in a GSEA prerank test
FDRqVal_Filter_05 = partial(Filter, key='FDR q-val', le=0.05)
FWERpVal_Filter_05 = partial(Filter, key='FWER p-val', le=0.05)

__all__ = [
    'pVal_Filter_01',
    'pVal_Filter_05',
    'qVal_Filter_01',
    'qVal_Filter_05',
    'log2fc_Filter_1',
    'log2fc_Filter_m1',
    'log2fc_Filter_2',
    'log2fc_Filter_m2',
    'abs_log2fc_Filter_1',
    'abs_log2fc_Filter_2',
    'FDRqVal_Filter_05',
    'FWERpVal_Filter_05',
]
