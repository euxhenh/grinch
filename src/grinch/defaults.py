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

abs_log2fc_Filter_1 = partial(
    Filter,
    key='abs_log2fc',
    cutoff=1,
    greater_is_True=True,
    dtype='float',
)
