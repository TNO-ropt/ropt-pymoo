from typing import Any, Dict, Tuple  # noqa: INP001

import numpy as np
from numpy.typing import NDArray
from ropt.evaluator import EvaluatorContext, EvaluatorResult
from ropt.results import FunctionResults, Results
from ropt.workflow import BasicWorkflow

CONFIG: Dict[str, Any] = {
    "variables": {
        "initial_values": 2 * [0.0],
        "lower_bounds": [0.0, 0.0],
        "upper_bounds": [2.0, 2.0],
    },
    "optimizer": {
        "method": "soo.nonconvex.nelder.NelderMead",
        "options": {
            "termination": ("n_iter", 30),
        },
    },
}


def rosenbrock(variables: NDArray[np.float64], _: EvaluatorContext) -> EvaluatorResult:
    objectives = np.zeros((variables.shape[0], 1), dtype=np.float64)
    for idx in range(variables.shape[0]):
        x, y = variables[idx, :]
        objectives[idx, 0] = (1.0 - x) ** 2 + 100 * (y - x * x) ** 2
    return EvaluatorResult(objectives=objectives)


def report(results: Tuple[Results, ...]) -> None:
    for item in results:
        if isinstance(item, FunctionResults):
            assert item.functions is not None
            print(f"evaluation: {item.result_id}")
            print(f"  variables: {item.evaluations.variables}")
            print(f"  objective: {item.functions.weighted_objective}\n")


def run_optimization(config: Dict[str, Any]) -> None:
    optimal_result = BasicWorkflow(config, rosenbrock, callback=report).run().results
    assert optimal_result is not None
    assert optimal_result.functions is not None
    print(f"BEST RESULT: {optimal_result.result_id}")
    print(f"  variables: {optimal_result.evaluations.variables}")
    print(f"  objective: {optimal_result.functions.weighted_objective}\n")


def main() -> None:
    run_optimization(CONFIG)


if __name__ == "__main__":
    main()
