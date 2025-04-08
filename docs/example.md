Here is an example of a simple constrained discrete problem, solved using a genetic algorithm:

```py
from pathlib import Path
import numpy as np
from ropt.evaluator import EvaluatorResult
from ropt.plan import BasicOptimizer
from ruamel import yaml

# For convenience we use a YAML file to store the optimizer options:
options = yaml.YAML(typ="safe", pure=True).load(Path("options.yml"))

CONFIG = {
    "variables": {
        # Ignored, but needed to establish the number of variables:
        "initial_values": 2 * [0.0],
        "lower_bounds": [0.0, 0.0],
        "upper_bounds": [10.0, 10.0],
    },
    "optimizer": {
        "method": "soo.nonconvex.ga.GA",
        "options": options,
    },
    "nonlinear_constraints": {"lower_bounds": [-np.inf], "upper_bounds": [0.0]},
}

def function(variables, _):
    x, y = variables[0, :]
    objectives = np.array(-min(3 * x, y), ndmin=2, dtype=np.float64)
    constraints = np.array(x + y - 10, ndmin=2, dtype=np.float64)
    return EvaluatorResult(objectives=objectives, constraints=constraints)

optimal_result = BasicOptimizer(CONFIG, function).run().results
print(f"Optimal variables: {optimal_result.evaluations.variables}")
print(f"Optimal objective: {optimal_result.functions.weighted_objective}")
```

To run, first create a YAML file called `options.yml` with the following contents:

```yaml
parameters:
  pop_size: 20
  sampling:
    object: operators.sampling.rnd.IntegerRandomSampling
  crossover:
    object: operators.crossover.sbx.SBX
    parameters:
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

termination:
  name: max_gen.MaximumGenerationTermination
  parameters:
    n_max_gen: 10

constraints:
  name: as_penalty.ConstraintsAsPenalty
  parameters:
    penalty: 100.0

seed: 1234
```

Running this will output the following:
```console
$ python example.py
=================================================
n_gen  |  n_eval  |     f_avg     |     f_min    
=================================================
     1 |       18 |  1.740556E+02 | -6.000000E+00
     2 |       38 | -2.300000E+00 | -6.000000E+00
     3 |       58 | -3.600000E+00 | -6.000000E+00
     4 |       78 | -4.400000E+00 | -7.000000E+00
     5 |       98 | -4.450000E+00 | -7.000000E+00
     6 |      118 | -4.500000E+00 | -7.000000E+00
     7 |      138 | -4.600000E+00 | -7.000000E+00
     8 |      158 | -4.600000E+00 | -7.000000E+00
     9 |      178 | -4.600000E+00 | -7.000000E+00
    10 |      198 | -4.600000E+00 | -7.000000E+00
Optimal variables: [3. 7.]
Optimal objective: -7.0
```