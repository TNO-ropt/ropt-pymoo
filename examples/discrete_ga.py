from pathlib import Path  # noqa: INP001
from typing import Any, Dict, Tuple

import numpy as np
from numpy.typing import NDArray
from ropt.enums import ConstraintType
from ropt.evaluator import EvaluatorContext, EvaluatorResult
from ropt.results import FunctionResults, Results
from ropt.workflow import BasicOptimizationWorkflow
from ruamel import yaml

# For convenience we use a YAML file to store the optimizer options:
options = yaml.YAML(typ="safe", pure=True).load(Path("discrete_ga.yml"))

CONFIG: Dict[str, Any] = {
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
    "nonlinear_constraints": {
        "types": [ConstraintType.LE],
        "rhs_values": [0.0],
    },
}


def function(variables: NDArray[np.float64], _: EvaluatorContext) -> EvaluatorResult:
    x, y = variables[0, :]
    objectives = np.array(-min(3 * x, y), ndmin=2, dtype=np.float64)
    constraints = np.array(x + y - 10, ndmin=2, dtype=np.float64)
    return EvaluatorResult(objectives=objectives, constraints=constraints)


def report(results: Tuple[Results, ...]) -> None:
    for item in results:
        if isinstance(item, FunctionResults):
            assert item.functions is not None
            print(f"evaluation: {item.result_id}")
            print(f"  variables: {item.evaluations.variables}")
            print(f"  objective: {item.functions.weighted_objective}\n")


def run_optimization(config: Dict[str, Any]) -> None:
    optimal_result = (
        BasicOptimizationWorkflow(config, function).add_callback(report).run().results
    )
    assert optimal_result is not None
    assert optimal_result.functions is not None
    print(f"BEST RESULT: {optimal_result.result_id}")
    print(f"  variables: {optimal_result.evaluations.variables}")
    print(f"  objective: {optimal_result.functions.weighted_objective}\n")


def main() -> None:
    run_optimization(CONFIG)


if __name__ == "__main__":
    main()
