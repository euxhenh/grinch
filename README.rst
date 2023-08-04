.. -*- mode: rst -*-

|PythonVersion|_ |Codecov|_

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.10-blue
.. _PythonVersion: https://pypi.org/project/scikit-learn/
.. |Codecov| image:: https://codecov.io/gh/euxhenh/grinch/branch/main/graph/badge.svg?token=P8KNCOKJ69
.. _Codecov: https://codecov.io/gh/euxhenh/grinch

**grinch** is a (under development) Python library for single-cell data
analysis aimed at reducing boilerplate code through the use of workflow
'config' files. This encourages reproducibility, minimizes bugs, and allows
the construction of complex pipelines with zero code written.

The library was originally written to prevent overcrowding project
directories with a myriad jupter notebooks and scripts, and replace them
with short and self-explanatory configs. **grinch** supports many steps of
the single-cell analysis pipeline, from normalization and unsupervised
learning, to gene differential expression and gene set enrichment
analysis.

Architecture
____________

**grinch** uses the `AnnData <https://anndata.readthedocs.io/en/latest/>`_
data format to store data matrices and annotations. It relies on `hydra
<https://hydra.cc/docs/intro/>`_ for parsing `yaml` config files and
`pydantic <https://pydantic-docs.helpmanual.io/>`_ for data validation.

Usage
_____

Example configs can be found in the `conf` directory. More docs to come...
