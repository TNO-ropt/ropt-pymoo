# A pymoo optimizer plugin for ropt
This package installs a plugin for the [ropt](https://github.com/tno-ropt/ropt)
robust optimization package, providing access to algorithms from the pymoo
optimization package.

`ropt-pymoo` is developed by the Netherlands Organisation for Applied Scientific
Research (TNO). All files in this repository are released under the GNU General
Public License v3.0 (a copy is provided in the LICENSE file).

See also the online [`ropt`](https://tno-ropt.github.io/ropt/) and
[`ropt-pymoo`](https://tno-ropt.github.io/ropt-pymoo/) manuals for more
information.


## Dependencies
This code has been tested with Python version 3.11.

The plugin requires the [`pymoo`](https://pymoo.org/) optimizer.


## Installation
From PyPI:
```bash
pip install ropt-pymoo
```


## Development
The `ropt-pymoo` source distribution can be found on
[GitHub](https://github.com/tno-ropt/ropt-pymoo). It uses a standard
`pyproject.toml` file, which contains build information and configuration
settings for various tools. A development environment can be set up with
compatible tools of your choice.

The `ropt-pymoo` package uses [ruff](https://docs.astral.sh/ruff/) (for
formatting and linting), [mypy](https://www.mypy-lang.org/) (for static typing),
and [pytest](https://docs.pytest.org/en/stable/) (for running the test suite).
