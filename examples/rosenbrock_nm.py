"""Rosenbrock example."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from ropt.components.evaluators import EvaluationFunctionContext
from ropt.simple import EvaluateResult, optimize

initial_values = 2 * [0.0]

CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": 2,
        "lower_bounds": [0.0, 0.0],
        "upper_bounds": [2.0, 2.0],
    },
    "backend": {
        "method": "soo.nonconvex.nelder.NelderMead",
        "options": {
            "termination": ("n_iter", 30),
        },
    },
}


def rosenbrock(variables: NDArray[np.float64], _: EvaluationFunctionContext) -> float:
    """Evaluate the function.

    Args:
        variables: The variables to evaluate

    Returns:
        Calculated objectives.
    """
    x, y = variables
    return (1.0 - x) ** 2 + 100 * (y - x * x) ** 2


def report(result: EvaluateResult) -> None:
    """Report results of an evaluation.

    Args:
        result: The result.
    """
    if result.results.functions is not None:
        print(f"  variables: {result.results.evaluations.variables}")
        print(f"  objective: {result.target_objective}\n")


def run_optimization(config: dict[str, Any]) -> None:
    """Run the optimization."""
    result = optimize(config, initial_values, rosenbrock, report=report)
    assert result.variables is not None
    assert result.target_objective is not None
    print(f"  variables: {result.variables}")
    print(f"  objective: {result.target_objective}\n")


def main() -> None:
    """Main function."""
    run_optimization(CONFIG)


if __name__ == "__main__":
    main()
