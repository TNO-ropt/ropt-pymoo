### Introduction

An optimization by ropt using the plugin works mostly as any other optimization
run. However, there are a few things to consider:

1.  **Gradient Information:** The plugin does not use gradient information, as
    `pymoo` algorithms typically don't support user-defined gradients. Any
    gradient calculation settings in `ropt` will be ignored.
2.  **Ignored `ropt` Parameters:** Some standard `ropt` optimization parameters
    like `max_iterations` and `tolerance` are not used by this plugin and will
    have no effect.
3.  **Initial Values:** `pymoo` generally ignores initial variable values.
    However, you still need to provide an initial value vector in `ropt` simply
    to define the number of variables; a zero vector is sufficient for this
    purpose.
4.  **Constraint Support:** Both linear and non-linear constraints are handled.
    Linear constraints are automatically converted to non-linear constraints
    internally before being passed to `pymoo`.
5.  **Algorithm Specification:** You must specify the `pymoo` algorithm using
    its fully qualified object name as found in the `pymoo.algorithms` module
    (e.g., `soo.nonconvex.ga.GA`), not just a short name.
6.  **Algorithm Configuration via `options`:** The chosen `pymoo` algorithm and
    its specific parameters are configured entirely through the `options` field
    within the `ropt` configuration object. The structure of this `options`
    dictionary directly mirrors the way options are set in `pymoo`, as detailed
    in the configuration section below.

### Configuration

The algorithm to specify is set by the `method` field in `optimization` section
of a `ropt` configuration. Futher configuration of `pymoo` algorithms is
performed via the `options` field. The following example demonstrates the
configuration process for a [Genetic
Algorithm](https://pymoo.org/algorithms/soo/ga.html), derived from the `pymoo`
manual, incorporating a penalty constraint. Here is how this is done in `pymoo`:

```python
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.constraints.as_penalty import ConstraintsAsPenalty
from pymoo.problems import get_problem

problem = get_problem("g1")

method = GA(
    pop_size=20,
    sampling=IntegerRandomSampling(),
    crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
    mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
    eliminate_duplicates=True,
)

res = minimize(
    ConstraintsAsPenalty(problem, penalty=100.0),
    method,
    termination=("n_gen", 40),
    seed=1234,
)
```

To configure the equivalent optimization in `ropt`, first set the `method` field
within the `optimization` section of the `ropt` configuration to the fully
qualified name of the algorithm object in the `pymoo.algorithms` module (e.g.,
`"soo.nonconvex.ga.GA"`).

Next, define the specific algorithm parameters, termination criteria,
constraints handling object, and random seed within the `options` field. This
field accepts a nested dictionary structure. The `ropt-pymoo` plugin parses this
dictionary to instantiate and configure the necessary `pymoo` objects based on
the provided names and parameters.

The general structure within the `options` dictionary is as follows:

- **Algorithm Parameters:** Arguments passed directly to the main algorithm's
  constructor (like `pop_size` for `GA`) are typically nested under a
  top-level `parameters` key.
- **Object Parameters:** When a parameter's value is itself a `pymoo` object
  (e.g., `sampling`, `crossover`, `mutation`), specify it using a nested
  dictionary containing:
    - An `object` key: The fully qualified name of the `pymoo` class (e.g.,
      `"operators.sampling.rnd.IntegerRandomSampling"`).
    - An optional `parameters` key: A dictionary of arguments to pass to
      *that* object's constructor. This can be nested further if those
      arguments are also objects.
- **Termination, Constraints, Seed:** These are typically defined using their
  own top-level keys (`termination`, `constraints`, `seed`) within the
  `options` dictionary, often following the same `object`/`parameters` pattern
  if they require configuration.

For clarity, the configuration corresponding to the preceding Python `GA`
example is shown below in YAML format, demonstrating how simple values and
nested objects are specified.

Within the `options` dictionary, the `parameters` key holds the arguments for
the `GA` object. The keys inside this `parameters` dictionary correspond to the
keyword arguments accepted by the `GA` constructor:

```yaml
parameters:
  pop_size: 20
  sampling:  # Specify objects using their fully qualified names:
    object: operators.sampling.rnd.IntegerRandomSampling
  crossover:
    object: operators.crossover.sbx.SBX
    parameters:  # Specify parameters to an object with a `parameters` field:
      prob: 1.0
      eta: 3.0
      vtype: float
      repair:
        object: operators.repair.rounding.RoundingRepair
  mutation:
    object: operators.mutation.pm.PM
    parameters:
      prob: 1.0
      eta: 3.0
      vtype: float
      repair:
        object: operators.repair.rounding.RoundingRepair
  eliminate_duplicates: True
```

The termination criterion is configured using the `termination` field, either by
using a list of parameters that is passed on to the
[`pymoo.termination.get_termination()`](https://pymoo.org/interface/termination.html)
function:

```yaml
termination: ["n_iter", 30]
```

Alternatively, the termination can be configured using a specific termination
object. Specify this object using its fully qualified name from the
[`pymoo.termination`](https://pymoo.org/interface/termination.html) module:

```yaml
termination:
  name: max_gen.MaximumGenerationTermination
  parameters:
    n_max_gen: 10
```

Constraints defined in the `ropt` configuration are automatically passed to
[`pymoo`](https://pymoo.org/constraints/index.html) for handling. You can
customize how `pymoo` manages these constraints by specifying a particular
constraint handling class, such as
[`ConstraintsAsPenalty`](https://pymoo.org/constraints/as_penalty.html) used in
the example. To configure this, add a `constraints` field to the `options`
dictionary, specifying the fully qualified name of the desired class from the
`pymoo.constraints` module and its parameters:

```yaml
constraints:
  name: as_penalty.ConstraintsAsPenalty
  parameters:
    penalty: 100.0
```

Finally, since `GA` requires random number generation, we specify a seed:

```yaml
seed: 1234
```
