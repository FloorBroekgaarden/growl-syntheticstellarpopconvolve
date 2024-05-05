# Synthetic Stellar Pop Convolve (SSPC)
![docstring coverage](./badges/docstring_coverage.svg) ![test coverage](./badges/test_coverage.svg)

This repository contains the code and documentation for the synthetic stellar-population convolution code-base `Synthetic Stellar Pop Convolve (SSPC)`. `SSPC` is available in a [Gitlab repository](https://gitlab.com/dhendriks/syntheticstellarpopconvolve) as well as on [Pypi](https://pypi.org/project/syntheticstellarpopconvolve/).

The code can be used to convolve the output of stellar population-synthesis codes with (cosmological) starformation rates.

The code was originally developed by David Hendriks (with invaluable help from Lieke van Son) for the project of [Hendriks et al. 2023 (MNRAS)](https://doi.org/10.1093/mnras/stad2857), where it was used to convolve gravitational-wave merger events from binary systems as well as supernova events from both binary systems and single stars with a cosmological star-formation rate.

## Installation


### Requirements
The Python packages that are required for this code to run are listed in the `requirements.txt`, which automatically gets read out by `setup.py`.

### Installation via PIP:
To install this package via pip:

```
pip install syntheticstellarpopconvolve
```

### Installation from source:
To install the `binary_c-python` from source, which is useful for development versions and customisation, run

```
./install.sh
```

This will install the package, along with all the dependencies, into the current active (virtual) python environment.


## documentation
Documentation is available in the repository and on [readthedocs](https://synthetic-stellar-pop-convolve.readthedocs.io/en/latest/) where we provide [tutorial and example use-case notebooks](https://synthetic-stellar-pop-convolve.readthedocs.io/en/latest/example_notebooks.html)

## Development:
If you want to contribute to the code, then it is recommended that you install the packages in `development_requirements.txt`:

```
pip install -r development_requirements.txt
```

Please do not hesitate to contact us to discuss any contribution. Please see `HOW_TO_CONTRIBUTE`.

Some useful commands to generate documentation and coverage reports are stored in the `commands/` directory.

We use the following naming convention for development and release branches:

```
development/<binary_c-python version>/<binary_c version>
releases/<binary_c-python version>/<binary_c version>
```

### Generating documentation
To build the documentation manually, run

```
./generate_docs.sh
```

from within the `commands/` directory. Note: this requires `pandoc` to be installed on your machine.

### Generating docstring and test coverage report
To generate the unit test and docstring coverage report, run

```
./generate_reports.sh
```

from within the `commands/` directory.
