
# GRIDLib

<div align="center">

[![PyPI - Version](https://img.shields.io/pypi/v/gridlib.svg)](https://pypi.python.org/pypi/gridlib)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/gridlib.svg)](https://pypi.python.org/pypi/gridlib)
[![Tests](https://github.com/boydcpeters/gridlib/workflows/tests/badge.svg)](https://github.com/boydcpeters/gridlib/actions?workflow=tests)
[![Codecov](https://codecov.io/gh/boydcpeters/gridlib/branch/main/graph/badge.svg)](https://codecov.io/gh/boydcpeters/gridlib)
[![Read the Docs](https://readthedocs.org/projects/gridlib/badge/)](https://gridlib.readthedocs.io/)
[![PyPI - License](https://img.shields.io/pypi/l/gridlib.svg)](https://pypi.python.org/pypi/gridlib)

[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](https://www.contributor-covenant.org/version/2/0/code_of_conduct/)

</div>

Python package to perform GRID analysis on fluorescence survival time distributions. Genuine Rate IDentification (GRID) analysis can be used to infer the dissociation rates of molecules from fluorescence survival time distributions retrieved with single-molecule tracking [[1]](#1). This package is based on work and research performed during my Bachelor end project in the [Ihor Smal lab](https://smal.ws) under the supervision of Ihor Smal and Maarten W. Paul.

* GitHub repo: <https://github.com/boydcpeters/gridlib.git>
* Documentation: <https://gridlib.readthedocs.io>
* Free software: GNU General Public License v3

## Features

* Simulate fluorescence survival time distributions with user-defined parameters.
* Perform GRID analysis on fluorescence survival time distributions.
* Plot analysis results with matplotlib.
* Perfrom GRID resampling on fluorescence survival time distributions.
* Plot resampling results with matplotlib.
* Load and save fluorescence survival time distributions and analysis and resampling results.

## Quickstart

Install the package:

```bash
pip install gridlib
```

There are a number of example scripts to perform some analyses in the `examples` folder with more
extensive explanations provided in the [documentation](https://gridlib.readthedocs.io/).

## References

The GRID fitting procedure implemented in this package is based on the following paper:

<a id="1">[1]</a>
Reisser, M., Hettich, J., Kuhn, T., Popp, A.P., Große-Berkenbusch, A. and Gebhardt, J.C.M. (2020). Inferring quantity and qualities of superimposed reaction rates from single molecule survival time distributions. Scientific Reports, 10(1). doi:10.1038/s41598-020-58634-y.

BibTex entry for the paper:

```latex
@article{reisser2020inferring,
  title = {Inferring quantity and qualities of superimposed reaction rates from single molecule survival time distributions},
  author = {Reisser, Matthias and Hettich, Johannes and Kuhn, Timo and Popp, Achim P and Gro{\ss}e-Berkenbusch, Andreas and Gebhardt, J Christof M},
  journal = {Scientific reports},
  volume = {10},
  number = {1},
  pages = {1--13},
  year = {2020},
  publisher = {Nature Publishing Group}
}
```

## Citing GRIDLib

To cite this repository:

```latex
@article{gridlib2022github,
  title = {GRIDLib: Python package to perform GRID analysis on fluorescence survival time distributions.},
  author = {Boyd Peters},
  url = {https://github.com/boydcpeters/gridlib},
  version = {0.4.1},
  year={2022},
}
```

In the above BibTex entry, the version number is the current version number and the year corresponds to the
project's open-source release.

## Credits

This package was created with [Cookiecutter][cookiecutter] and the [fedejaure/cookiecutter-modern-pypackage][cookiecutter-modern-pypackage] project template.

[cookiecutter]: https://github.com/cookiecutter/cookiecutter
[cookiecutter-modern-pypackage]: https://github.com/fedejaure/cookiecutter-modern-pypackage
