# pylint: disable=protected-access

from typing import Any, Dict, cast

import numpy as np
import pytest
from numpy.typing import NDArray
from ropt.enums import ConstraintType
from ropt.optimization import EnsembleOptimizer


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> Dict[str, Any]:
    return {
        "variables": {
            "initial_values": [0.2, 0.0, 0.1],
        },
        "optimizer": {
            "backend": "pymoo",
            "algorithm": "soo.nonconvex.nelder.NelderMead",
        },
        "objective_functions": {
            "weights": [0.75, 0.25],
        },
    }


@pytest.mark.parametrize("parallel", [False, True])
def test_pymoo_bound_constraints(
    enopt_config: Dict[str, Any], evaluator: Any, parallel: bool
) -> None:
    enopt_config["variables"]["lower_bounds"] = [0.15, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 0.2]
    enopt_config["optimizer"]["parallel"] = parallel
    optimizer = EnsembleOptimizer(evaluator())
    result = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert result is not None
    assert np.allclose(result.evaluations.variables, [0.15, 0.0, 0.2], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
def test_pymoo_termination(
    enopt_config: Dict[str, Any], evaluator: Any, parallel: bool
) -> None:
    enopt_config["variables"]["lower_bounds"] = [0.15, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 0.2]
    enopt_config["optimizer"]["parallel"] = parallel

    enopt_config["optimizer"]["options"] = {
        "termination": {"name": "default.DefaultSingleObjectiveTermination"}
    }
    optimizer = EnsembleOptimizer(evaluator())
    result = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert result is not None
    assert np.allclose(result.evaluations.variables, [0.15, 0.0, 0.2], atol=0.02)

    variables = result.evaluations.variables.copy()

    enopt_config["optimizer"]["options"] = {"termination": {"name": "soo"}}
    optimizer = EnsembleOptimizer(evaluator())
    result = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert result is not None
    assert np.allclose(result.evaluations.variables, variables, atol=0.0, rtol=1e-10)


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("bound_type", [ConstraintType.LE, ConstraintType.GE])
def test_pymoo_ineq_nonlinear_constraints(
    enopt_config: Dict[str, Any],
    bound_type: ConstraintType,
    evaluator: Any,
    parallel: bool,
    test_functions: Any,
) -> None:
    enopt_config["optimizer"]["parallel"] = parallel
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["nonlinear_constraints"] = {
        "rhs_values": 0.4 if bound_type == ConstraintType.LE else -0.4,
        "types": [bound_type],
    }

    weight = 1.0 if bound_type == ConstraintType.LE else -1.0
    test_functions = (
        *test_functions,
        lambda variables: cast(
            NDArray[np.float64], weight * variables[0] + weight * variables[2]
        ),
    )
    optimizer = EnsembleOptimizer(evaluator(test_functions))
    result = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {
                "tracker": {
                    "id": "optimum",
                    "source": "opt",
                    "constraint_tolerance": 1e-4,
                }
            },
        ],
    )
    assert result is not None
    assert np.allclose(result.evaluations.variables, [-0.05, 0.0, 0.45], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
def test_pymoo_eq_nonlinear_constraints(
    enopt_config: Dict[str, Any],
    evaluator: Any,
    parallel: bool,
    test_functions: Any,
) -> None:
    enopt_config["optimizer"]["parallel"] = parallel
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["nonlinear_constraints"] = {
        "rhs_values": 1.0,
        "types": [ConstraintType.EQ],
    }

    test_functions = (
        *test_functions,
        lambda variables: cast(NDArray[np.float64], variables[0] + variables[2]),
    )
    optimizer = EnsembleOptimizer(evaluator(test_functions))
    result = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {
                "tracker": {
                    "id": "optimum",
                    "source": "opt",
                    "constraint_tolerance": 1e-4,
                }
            },
        ],
    )
    assert result is not None
    assert np.allclose(result.evaluations.variables, [0.25, 0.0, 0.75], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
def test_pymoo_le_ge_linear_constraints(
    enopt_config: Dict[str, Any], evaluator: Any, parallel: bool
) -> None:
    enopt_config["optimizer"]["parallel"] = parallel
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [-1, 0, -1]],
        "rhs_values": [0.4, -0.4],
        "types": [ConstraintType.LE, ConstraintType.GE],
    }

    optimizer = EnsembleOptimizer(evaluator())
    result = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {
                "tracker": {
                    "id": "optimum",
                    "source": "opt",
                    "constraint_tolerance": 1e-4,
                }
            },
        ],
    )
    assert result is not None
    assert np.allclose(result.evaluations.variables, [-0.05, 0.0, 0.45], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
def test_pymoo_eq_linear_constraints(
    enopt_config: Dict[str, Any], evaluator: Any, parallel: bool
) -> None:
    enopt_config["optimizer"]["parallel"] = parallel
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [0, 1, 1]],
        "rhs_values": [1.0, 0.75],
        "types": [ConstraintType.EQ, ConstraintType.EQ],
    }

    optimizer = EnsembleOptimizer(evaluator())
    result = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {
                "tracker": {
                    "id": "optimum",
                    "source": "opt",
                    "constraint_tolerance": 1e-4,
                }
            },
        ],
    )
    assert result is not None
    assert np.allclose(result.evaluations.variables, [0.25, 0.0, 0.75], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
def test_pymoo_eq_mixed_constraints(
    enopt_config: Dict[str, Any],
    evaluator: Any,
    parallel: bool,
    test_functions: Any,
) -> None:
    enopt_config["optimizer"]["parallel"] = parallel
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["nonlinear_constraints"] = {
        "rhs_values": 1.0,
        "types": [ConstraintType.EQ],
    }
    enopt_config["linear_constraints"] = {
        "coefficients": [[0, 0, 1]],
        "rhs_values": [0.75],
        "types": [ConstraintType.EQ],
    }

    test_functions = (
        *test_functions,
        lambda variables: cast(NDArray[np.float64], variables[0] + variables[2]),
    )
    optimizer = EnsembleOptimizer(evaluator(test_functions))
    result = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {
                "tracker": {
                    "id": "optimum",
                    "source": "opt",
                    "constraint_tolerance": 1e-4,
                }
            },
        ],
    )
    assert result is not None
    assert np.allclose(result.evaluations.variables, [0.25, 0.0, 0.75], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
def test_pymoo_constraint_handling(
    enopt_config: Dict[str, Any],
    evaluator: Any,
    parallel: bool,
    test_functions: Any,
) -> None:
    enopt_config["optimizer"]["parallel"] = parallel
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["nonlinear_constraints"] = {
        "rhs_values": 0.4,
        "types": [ConstraintType.LE],
    }
    enopt_config["optimizer"]["options"] = {
        "termination": {"name": "default.DefaultSingleObjectiveTermination"},
        "constraints": {
            "name": "as_penalty.ConstraintsAsPenalty",
            "parameters": {"penalty": 1},
        },
    }

    test_functions = (
        *test_functions,
        lambda variables: cast(NDArray[np.float64], variables[0] + variables[2]),
    )

    optimizer = EnsembleOptimizer(evaluator(test_functions))
    result = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {
                "tracker": {
                    "id": "optimum",
                    "source": "opt",
                    "constraint_tolerance": 1e-4,
                }
            },
        ],
    )
    assert result is not None
    assert np.allclose(result.evaluations.variables, [-0.05, 0.0, 0.45], atol=0.02)
