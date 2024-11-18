# A pymoo optimizer plugin for ropt
This package installs a plugin for the [ropt](https://github.com/tno-ropt/ropt)
robust optimization package, providing access to algorithms from the pymoo
optimization package.

`ropt-pymoo` is developed by the Netherlands Organisation for Applied Scientific
Research (TNO). All files in this repository are released under the GNU General
Public License v3.0 (a copy is provided in the LICENSE file).


## Dependencies
This code has been tested with Python version 3.11.

The plugin requires the [pymoo](https://pymoo.org/) optimizer.


## Installation
From PyPI:
```bash
pip install ropt-pymoo
```


## Usage
An optimization by ropt using the plugin works mostly as any other optimization
run (see also the [ropt documentation](https://tno-ropt.github.io/ropt/)).
However, there are a few things to consider:

1. Gradients are not used, as `pymoo` does not seem to support passing
   user-defined gradients. Hence, any specifications relating to gradient
   calulcations in ropt are ignored.
2. Some standard optimization parameters that can be specified in the
   optimization section are ignored, specifically:
    - `max_iterations`
    - `tolerance`
3. The initial values of the variables are ignored, since `pymoo` generally does
   not use them. In ropt you still need to specify them, since the size of the
   vector determines the number of variables. Setting it to a vector of zero
   values is fine.
4. Linear and non-linear constraints are both supported. Linear constraints are
   not supported directly, but are internally converted to non-linear
   constraints.
5. The algorithm and its options are specified using a syntax closely following
   the `pymoo` manual. For instance, rather than just giving an algorithm name,
   you have to specify the full qualified name of the corresponding object as
   found in the `pymoo.algorithms` module. For instance to specify the `GA`
   algorithm, use: `soo.nonconvex.ga.GA`.
6. The algorithms itself are entirely configured via the `options` field in the
   optimization section of the ropt configuration object. Also in this case, the
   syntax follows the `pymoo` manual. See the section below for more
   information.

## Configuring an algorithm.
Configuration of any of the `pymoo` algorithms is done via the options field in
the ropt configuration object. For instance, consider this example for starting
a `GA` optimization from the `pymoo` manual, with a penalty constraint added:

```python
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.constraints.as_penalty import ConstraintsAsPenalty

method = GA(
    pop_size=20,
    sampling=IntegerRandomSampling(),
    crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
    mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
    eliminate_duplicates=True,
)

res = minimize(ConstraintsAsPenalty(
    problem, penalty=100.0),
    method,
    termination=('n_gen', 40),
    seed=1234,
)
```

To run the equivalent optimization, we need to specify the method and the
termination in the options field. We also  need to specify the constraints
object, and a seed. To do this the different objects are specified with their
parameters in a nested dictionary that will be parsed into equivalent code. For
this example we need to pass a nested dict, for clarity displayed as yml here:
```yaml
parameters:  # The parameters of the GA object:
    pop_size: 20
    sampling:  # The sampling parameter is an object, specify its full path in pymoo:
    object: operators.sampling.rnd.IntegerRandomSampling
    crossover:  # Also an object:
    object: operators.crossover.sbx.SBX
    parameters:  # Specify the parameters passed to the crossover ojbect:
        prob: 1.0
        eta: 3.0
        vtype: float
        repair:  # A repair object, passed to the crossover object:
        object: operators.repair.rounding.RoundingRepair
    mutation:  # An object:
    object: operators.mutation.pm.PM
    parameters:  # And its parameters:
        prob: 1.0
        eta: 3.0
        vtype: float
        repair:  # A repair object, passed to the mutation object:
        object: operators.repair.rounding.RoundingRepair
    eliminate_duplicates: True
termination:  # Specification of the termination object:
    name: max_gen.MaximumGenerationTermination
    parameters:
        n_max_gen: 10
# Alternative specification for the termination, following pymoo practice:
# "termination": ("n_iter", 30)
constraints:  # Specification of the constraint object:
    name: as_penalty.ConstraintsAsPenalty
    parameters:
        penalty: 100.0
seed: 1234  # The seed that is passed to the minimize function:
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
