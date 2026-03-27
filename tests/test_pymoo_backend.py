# ruff: noqa: RUF069

import sys
from typing import Any

import numpy as np
import pytest
from ropt.enums import ExitCode
from ropt.workflow import BasicOptimizer

# ruff: noqa: FBT001

initial_values = [0.2, 0.0, 0.1]


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> dict[str, Any]:
    return {
        "variables": {
            "variable_count": len(initial_values),
            "lower_bounds": [-1.0, -1.0, -1.0],
            "upper_bounds": [1.0, 1.0, 1.0],
        },
        "backend": {
            "method": "soo.nonconvex.nelder.NelderMead",
        },
        "objectives": {
            "weights": [0.75, 0.25],
        },
    }


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize(
    "external", ["", pytest.param("external/", marks=pytest.mark.external)]
)
def test_pymoo_bound_constraints(
    enopt_config: dict[str, Any], evaluator: Any, parallel: bool, external: str
) -> None:
    enopt_config["backend"]["method"] = f"{external}soo.nonconvex.nelder.NelderMead"
    enopt_config["variables"]["lower_bounds"] = [0.15, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 0.2]
    enopt_config["backend"]["parallel"] = parallel
    optimizer = BasicOptimizer(enopt_config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [0.15, 0.0, 0.2], atol=0.02
    )


@pytest.mark.parametrize("parallel", [False, True])
def test_pymoo_termination(
    enopt_config: dict[str, Any], evaluator: Any, parallel: bool
) -> None:
    enopt_config["variables"]["lower_bounds"] = [0.15, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 0.2]
    enopt_config["backend"]["parallel"] = parallel

    enopt_config["backend"]["options"] = {
        "termination": {"name": "default.DefaultSingleObjectiveTermination"}
    }
    optimizer1 = BasicOptimizer(enopt_config, evaluator())
    optimizer1.run(initial_values)
    assert optimizer1.results is not None
    assert np.allclose(
        optimizer1.results.evaluations.variables, [0.15, 0.0, 0.2], atol=0.02
    )

    enopt_config["backend"]["options"] = {"termination": {"name": "soo"}}
    optimizer2 = BasicOptimizer(enopt_config, evaluator())
    optimizer2.run(initial_values)
    assert optimizer2.results is not None
    assert np.allclose(
        optimizer2.results.evaluations.variables, [0.15, 0.0, 0.2], atol=0.02
    )
    assert np.allclose(
        optimizer1.results.evaluations.variables,
        optimizer2.results.evaluations.variables,
        atol=0.0,
        rtol=1e-10,
    )


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize(
    ("lower_bounds", "upper_bounds"), [(-np.inf, 0.4), (-0.4, np.inf)]
)
def test_pymoo_ineq_nonlinear_constraints(  # noqa: PLR0917
    enopt_config: dict[str, Any],
    lower_bounds: Any,
    upper_bounds: Any,
    evaluator: Any,
    parallel: bool,
    test_functions: Any,
) -> None:
    enopt_config["backend"]["parallel"] = parallel
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
    }

    weight = 1.0 if upper_bounds == 0.4 else -1.0
    test_functions = (
        *test_functions,
        lambda variables, _: weight * variables[0] + weight * variables[2],
    )
    optimizer = BasicOptimizer(enopt_config, evaluator(test_functions))
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [-0.05, 0.0, 0.45], atol=0.02
    )


@pytest.mark.parametrize("parallel", [False, True])
def test_pymoo_eq_nonlinear_constraints(
    enopt_config: dict[str, Any],
    evaluator: Any,
    parallel: bool,
    test_functions: Any,
) -> None:
    enopt_config["backend"]["parallel"] = parallel
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": 1.0,
        "upper_bounds": 1.0,
    }

    test_functions = (
        *test_functions,
        lambda variables, _: variables[0] + variables[2],
    )
    optimizer = BasicOptimizer(
        enopt_config, evaluator(test_functions), constraint_tolerance=1e-4
    )
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [0.25, 0.0, 0.75], atol=0.02
    )


@pytest.mark.parametrize("parallel", [False, True])
def test_pymoo_ineq_nonlinear_constraints_two_sided(
    enopt_config: Any,
    parallel: bool,
    evaluator: Any,
    test_functions: Any,
) -> None:
    enopt_config["backend"]["parallel"] = parallel
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": [0.0],
        "upper_bounds": [0.3],
    }
    test_functions = (
        *test_functions,
        lambda variables, _: variables[0] + variables[2],
    )

    optimizer = BasicOptimizer(enopt_config, evaluator(test_functions))
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [-0.1, 0.0, 0.4], atol=0.02
    )


@pytest.mark.parametrize("parallel", [False, True])
def test_pymoo_le_ge_linear_constraints(
    enopt_config: dict[str, Any], evaluator: Any, parallel: bool
) -> None:
    enopt_config["backend"]["parallel"] = parallel
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1]],
        "lower_bounds": [-np.inf],
        "upper_bounds": [0.4],
    }

    optimizer = BasicOptimizer(enopt_config, evaluator(), constraint_tolerance=1e-4)
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [-0.05, 0.0, 0.45], atol=0.02
    )


@pytest.mark.parametrize("parallel", [False, True])
def test_pymoo_eq_linear_constraints(
    enopt_config: dict[str, Any], evaluator: Any, parallel: bool
) -> None:
    enopt_config["backend"]["parallel"] = parallel
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [0, 1, 1]],
        "lower_bounds": [1.0, 0.75],
        "upper_bounds": [1.0, 0.75],
    }

    optimizer = BasicOptimizer(enopt_config, evaluator(), constraint_tolerance=1e-4)
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [0.25, 0.0, 0.75], atol=0.02
    )


@pytest.mark.parametrize("parallel", [False, True])
def test_pymoo_le_ge_linear_constraints_two_sided(
    enopt_config: Any, evaluator: Any, parallel: bool
) -> None:
    enopt_config["backend"]["parallel"] = parallel
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [1, 0, 1]],
        "lower_bounds": [-np.inf, 0.0],
        "upper_bounds": [0.3, np.inf],
    }

    optimizer = BasicOptimizer(enopt_config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [-0.1, 0.0, 0.4], atol=0.02
    )

    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1]],
        "lower_bounds": [0.0],
        "upper_bounds": [0.3],
    }

    optimizer = BasicOptimizer(enopt_config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [-0.1, 0.0, 0.4], atol=0.02
    )


@pytest.mark.parametrize("parallel", [False, True])
def test_pymoo_eq_mixed_constraints(
    enopt_config: dict[str, Any],
    evaluator: Any,
    parallel: bool,
    test_functions: Any,
) -> None:
    enopt_config["backend"]["parallel"] = parallel
    enopt_config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": [1.0],
        "upper_bounds": [1.0],
    }
    enopt_config["linear_constraints"] = {
        "coefficients": [[0, 0, 1]],
        "lower_bounds": [0.75],
        "upper_bounds": [0.75],
    }

    test_functions = (
        *test_functions,
        lambda variables, _: variables[0] + variables[2],
    )
    optimizer = BasicOptimizer(
        enopt_config, evaluator(test_functions), constraint_tolerance=1e-4
    )
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [0.25, 0.0, 0.75], atol=0.04
    )


@pytest.mark.skipif(
    sys.version_info[:2] == (3, 13),
    reason="Fails on Python 3.13 on GitHub for unknown reasons",
)
@pytest.mark.parametrize("parallel", [False, True])
def test_pymoo_constraint_handling(
    enopt_config: dict[str, Any],
    evaluator: Any,
    parallel: bool,
    test_functions: Any,
) -> None:
    enopt_config["backend"]["parallel"] = parallel
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": -np.inf,
        "upper_bounds": 0.4,
    }
    enopt_config["backend"]["options"] = {
        "termination": {"name": "default.DefaultSingleObjectiveTermination"},
        "constraints": {
            "name": "as_penalty.ConstraintsAsPenalty",
            "parameters": {"penalty": 1},
        },
    }

    test_functions = (
        *test_functions,
        lambda variables, _: variables[0] + variables[2],
    )

    optimizer = BasicOptimizer(
        enopt_config, evaluator(test_functions), constraint_tolerance=1e-4
    )
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [-0.05, 0.0, 0.45], atol=0.02
    )


def test_pymoo_bound_constraints_with_failure(
    enopt_config: dict[str, Any], evaluator: Any, test_functions: Any
) -> None:
    enopt_config["variables"]["lower_bounds"] = [0.15, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 0.2]
    enopt_config["backend"]["method"] = "soo.nonconvex.de.DE"
    enopt_config["backend"]["parallel"] = True
    enopt_config["optimizer"] = {"max_functions": 1000}
    enopt_config["realizations"] = {"realization_min_success": 0}
    optimizer1 = BasicOptimizer(enopt_config, evaluator(test_functions))
    optimizer1.run(initial_values)
    assert optimizer1.results is not None
    assert np.allclose(
        optimizer1.results.evaluations.variables, [0.15, 0.0, 0.2], atol=0.02
    )

    counter = 0

    def _add_nan(x: Any, _: int) -> Any:
        nonlocal counter
        counter += 1
        if counter == 2:
            counter = 0
            return np.nan
        return test_functions[0](x, 0)

    optimizer2 = BasicOptimizer(enopt_config, evaluator((_add_nan, test_functions[1])))
    optimizer2.run(initial_values)
    assert optimizer2.results is not None
    assert np.allclose(
        optimizer2.results.evaluations.variables, [0.15, 0.0, 0.2], atol=0.02
    )
    assert not np.all(
        np.equal(
            optimizer1.results.evaluations.variables,
            optimizer2.results.evaluations.variables,
        )
    )


def test_pymoo_bound_constraints_no_failure_handling(
    enopt_config: dict[str, Any], evaluator: Any, test_functions: Any
) -> None:
    enopt_config["variables"]["lower_bounds"] = [0.15, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 0.2]
    enopt_config["backend"]["method"] = "soo.nonconvex.nelder.NelderMead"
    enopt_config["backend"]["parallel"] = True
    enopt_config["optimizer"] = {"max_functions": 800}

    optimizer1 = BasicOptimizer(enopt_config, evaluator(test_functions))
    optimizer1.run(initial_values)
    assert optimizer1.results is not None
    assert np.allclose(
        optimizer1.results.evaluations.variables, [0.15, 0.0, 0.2], atol=0.02
    )

    enopt_config["realizations"] = {"realization_min_success": 0}

    counter = 0

    def _add_nan(x: Any, _: int) -> Any:
        nonlocal counter
        counter += 1
        if counter == 2:
            counter = 0
            return np.nan
        return test_functions[0](x, 0)

    optimizer2 = BasicOptimizer(enopt_config, evaluator((_add_nan, test_functions[1])))
    exit_code = optimizer2.run(initial_values)
    assert exit_code == ExitCode.TOO_FEW_REALIZATIONS
    assert optimizer2.results is not None
    assert not np.all(
        np.equal(
            optimizer1.results.evaluations.variables,
            optimizer2.results.evaluations.variables,
        )
    )
