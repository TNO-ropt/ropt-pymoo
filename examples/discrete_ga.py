"""Discrete optimization example."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from ropt.components.evaluators import (
    EvaluationFunctionContext,
    EvaluationFunctionResult,
)
from ropt.simple import EvaluateResult, optimize

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
    "backend": {
        "method": "soo.nonconvex.ga.GA",
        "options": options,
    },
    "nonlinear_constraints": {
        "lower_bounds": [-np.inf],
        "upper_bounds": [0.0],
    },
}


def function(
    variables: NDArray[np.float64], _: EvaluationFunctionContext
) -> EvaluationFunctionResult:
    """Evaluate the function.

    Args:
        variables: The variables to evaluate

    Returns:
        Calculated objectives and constraints.
    """
    x, y = variables
    objectives = np.array([-min(3 * x, y)], dtype=np.float64)
    constraints = np.array([x + y - 10], dtype=np.float64)
    return EvaluationFunctionResult(objectives=objectives, constraints=constraints)


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
    result = optimize(config, initial_values, function, report=report)
    assert result.variables is not None
    assert result.target_objective is not None
    print(f"  variables: {result.variables}")
    print(f"  objective: {result.target_objective}\n")


def main() -> None:
    """Main function."""
    run_optimization(CONFIG)


if __name__ == "__main__":
    main()
