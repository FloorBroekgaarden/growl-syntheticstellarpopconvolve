# Synthetic Stellar Pop Convolve (SSPC)
![docstring coverage](./badges/docstring_coverage.svg) ![test coverage](./badges/test_coverage.svg)

This repository contains the code and documentation for the synthetic
stellar-population convolution code-base `Synthetic Stellar Pop
Convolve (SSPC)`. `SSPC` is available in a [Gitlab
repository](https://gitlab.com/dhendriks/syntheticstellarpopconvolve)
as well as on
[Pypi](https://pypi.org/project/syntheticstellarpopconvolve/).

The code was originally developed by David Hendriks (with invaluable
help from Lieke van Son) for the project of [Hendriks et al. 2023
(MNRAS)](https://doi.org/10.1093/mnras/stad2857), where it was used to
convolve gravitational-wave merger events from binary systems as well
as supernova events from both binary systems and single stars with a
cosmological star-formation rate.

SSPC can be used to convolve the output of stellar
population-synthesis codes with (cosmological) starformation
rates. It can convolve both event-based (line by line) data, as
well as ensemble-based (nested histogram) data. The user must provide
information about the column/layer that contains the necessary data to
perform the convolution, which is at minimum the delay-time and the
normalized yield. The latter quantity gets multiplied directly with
the appropriate SFR, and as such should already be weighted by
e.g. binary fractions, importance-sampling weights and others.

It is possible to provide additional weights *during* convolution, which can
include things like detection probability that depend on redshift.

list of some current features:

-   convolution of event-based (lines) data
-   convolution of ensemble-based (nested histograms) data
    -   pre- and post-convolution marginalisation of the ensembles
-   convolution of several datasets in sequence with a particular SFR
-   convolution of data with a sequence of SFRs
-   multiprocessed where relevant
-   astropy units
-   rescaling and transforming data
-   additional weighting during convolution (for e.g. detection probability)
-   etc.

In the coming period I will:

-   Continue to clean the code, update the docstrings of the functions, and
    provide type-hinting.
-   Continue to flesh out the tutorial notebooks and add examples and use-cases
    (including how to set up convolution with BinCodex-type data)
-   Add relevant star-formation rate and metallicity-distribution prescriptions
    (including for the MW), and make the part of the code that explains how to
    supply SFR information a bit more clear
-   make the advanced functionality more clear and provide a notebook on this.

Future features I aim to add in the foreseeable future:

-   better functionality for other data-sources/types: Currently event-based data
    requires it being stored in a dataframe but one should be able to provide
    custom-data functions to change this.
-   functionality for pre-convolution data filtering (i.e. performing queries on
    the data): while the input data can be filtered to only contain the relevant
    data, I can imagine that one may want to convolve the data with different
    filters using the same input-data (although that could be done
    post-convolution)
-   better support for spatially-resolved star-formation rates: while
    we currently can already convolve spatially resolved
    star-formation rates by just dividing the space into a grid and
    handling each cell in sequence using the functionality for
    handling multiple SFRs, it probably can be improved by allowing
    extra information to be passed along which can be used in
    e.g. selection functions.
-   ??? (please let me know about desired features)

The code is in late-beta stage, and is largely covered in unit-tests, but its
not entirely finished just yet and likely still contains some bugs that I have
not been able to pick up. Thats where this community can come in! I think this
code could become useful for many pop-synth groups, especially when people
submit git issues for bugs and feature requests!

As I said the code is not fully stable yet but I do invite people to start
having a look at the code-base and try out installing.

If anyone has questions, hit me up on the LISA-UCB slack channel or send me an
email on [mail@davidhendriks.com](mailto:mail@davidhendriks.com).


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
