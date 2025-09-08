Here is an example of a simple constrained discrete problem, solved using a genetic algorithm:

```py
from typing import Any

import numpy as np
from numpy.typing import NDArray
from ropt.evaluator import EvaluatorContext, EvaluatorResult
from ropt.plan import BasicOptimizer

options = {
    "parameters": {
        "pop_size": 20,
        "sampling": {"object": "operators.sampling.rnd.IntegerRandomSampling"},
        "crossover": {
            "object": "operators.crossover.sbx.SBX",
            "parameters": {
                "prob": 1.0,
                "eta": 3.0,
                "vtype": "float",
                "repair": {"object": "operators.repair.rounding.RoundingRepair"},
            },
        },
        "mutation": {
            "object": "operators.mutation.pm.PM",
            "parameters": {
                "prob": 1.0,
                "eta": 3.0,
                "vtype": "float",
                "repair": {"object": "operators.repair.rounding.RoundingRepair"},
            },
        },
        "eliminate_duplicates": True,
    },
    "termination": {
        "name": "max_gen.MaximumGenerationTermination",
        "parameters": {"n_max_gen": 10},
    },
    "constraints": {
        "name": "as_penalty.ConstraintsAsPenalty",
        "parameters": {"penalty": 100.0},
    },
    "seed": 1234,
}


initial_values = 2 * [0.0]

CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": len(initial_values),
        "lower_bounds": [0.0, 0.0],
        "upper_bounds": [10.0, 10.0],
    },
    "optimizer": {
        "method": "soo.nonconvex.ga.GA",
        "options": options,
    },
    "nonlinear_constraints": {
        "lower_bounds": [-np.inf],
        "upper_bounds": [0.0],
    },
}


def function(variables: NDArray[np.float64], _: EvaluatorContext) -> EvaluatorResult:
    x, y = variables[0, :]
    objectives = np.array(-min(3 * x, y), ndmin=2, dtype=np.float64)
    constraints = np.array(x + y - 10, ndmin=2, dtype=np.float64)
    return EvaluatorResult(objectives=objectives, constraints=constraints)


optimal_result = BasicOptimizer(CONFIG, function).run(initial_values).results
assert optimal_result is not None
assert optimal_result.functions is not None
print(f"  variables: {optimal_result.evaluations.variables}")
print(f"  objective: {optimal_result.functions.weighted_objective}\n")
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
