"""Discrete optimization example."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from ropt.evaluator import EvaluatorContext, EvaluatorResult
from ropt.plan import BasicOptimizer
from ropt.results import FunctionResults, Results

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
    """Evaluate the function.

    Args:
        variables: The variables to evaluate
        context:   Evaluator context

    Returns:
        Calculated objectives and constraints.
    """
    x, y = variables[0, :]
    objectives = np.array(-min(3 * x, y), ndmin=2, dtype=np.float64)
    constraints = np.array(x + y - 10, ndmin=2, dtype=np.float64)
    return EvaluatorResult(objectives=objectives, constraints=constraints)


def report(results: tuple[Results, ...]) -> None:
    """Report results of an evaluation.

    Args:
        results: The results.
    """
    for item in results:
        if isinstance(item, FunctionResults) and item.functions is not None:
            print(f"  variables: {item.evaluations.variables}")
            print(f"  objective: {item.functions.weighted_objective}\n")


def run_optimization(config: dict[str, Any]) -> None:
    """Run the optimization."""
    optimal_result = (
        BasicOptimizer(config, function)
        .set_results_callback(report)
        .run(initial_values)
        .results
    )
    assert optimal_result is not None
    assert optimal_result.functions is not None
    print(f"  variables: {optimal_result.evaluations.variables}")
    print(f"  objective: {optimal_result.functions.weighted_objective}\n")


def main() -> None:
    """Main function."""
    run_optimization(CONFIG)


if __name__ == "__main__":
    main()
