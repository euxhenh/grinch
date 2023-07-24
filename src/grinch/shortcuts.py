"""Useful defaults for importing directly in yaml's.
"""
from functools import partial

from .cond_filter import Filter


pVal_Filter_01 = partial(
    Filter,
    key='pvals',
    cutoff=0.01,
    greater_is_True=False,
    dtype='float',
)

pVal_Filter_05 = partial(
    Filter,
    key='pvals',
    cutoff=0.05,
    greater_is_True=False,
    dtype='float',
)

qVal_Filter_01 = partial(
    Filter,
    key='qvals',
    cutoff=0.01,
    greater_is_True=False,
    dtype='float',
)

qVal_Filter_05 = partial(
    Filter,
    key='qvals',
    cutoff=0.05,
    greater_is_True=False,
    dtype='float',
)

log2fc_Filter_1 = partial(
    Filter,
    key='log2fc',
    cutoff=1,
    greater_is_True=True,
    dtype='float',
)

log2fc_Filter_m1 = partial(
    Filter,
    key='log2fc',
    cutoff=-1,
    greater_is_True=False,
    dtype='float',
)

log2fc_Filter_2 = partial(
    Filter,
    key='log2fc',
    cutoff=2,
    greater_is_True=True,
    dtype='float',
)

log2fc_Filter_m2 = partial(
    Filter,
    key='log2fc',
    cutoff=-2,
    greater_is_True=False,
    dtype='float',
)

abs_log2fc_Filter_1 = partial(
    Filter,
    key='abs_log2fc',
    cutoff=1,
    greater_is_True=True,
    dtype='float',
)

abs_log2fc_Filter_2 = partial(
    Filter,
    key='abs_log2fc',
    cutoff=2,
    greater_is_True=True,
    dtype='float',
)

# For lead gene discovery in a GSEA prerank test
FDRqVal_Filter_05 = partial(
    Filter,
    key='FDR q-val',
    cutoff=0.05,
    greater_is_True=False,
    dtype='float',
)

FWERpVal_Filter_05 = partial(
    Filter,
    key='FWER p-val',
    cutoff=0.05,
    greater_is_True=False,
    dtype='float',
)
