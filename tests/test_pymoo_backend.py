# ruff: file-ignore[float-equality-comparison]

import sys
from typing import Any

import numpy as np
import pytest
from ropt.simple import optimize

# ruff: file-ignore[boolean-type-hint-positional-argument]

initial_values = [0.2, 0.0, 0.1]


@pytest.fixture(name="config")
def config_fixture() -> dict[str, Any]:
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
    config: dict[str, Any], eval_func: Any, parallel: bool, external: str
) -> None:
    config["backend"]["method"] = f"{external}soo.nonconvex.nelder.NelderMead"
    config["variables"]["lower_bounds"] = [0.15, -1.0, -1.0]
    config["variables"]["upper_bounds"] = [1.0, 1.0, 0.2]
    config["backend"]["parallel"] = parallel
    result = optimize(config, initial_values, eval_func())
    assert result.variables is not None
    assert np.allclose(result.variables, [0.15, 0.0, 0.2], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
def test_pymoo_termination(
    config: dict[str, Any], eval_func: Any, parallel: bool
) -> None:
    config["variables"]["lower_bounds"] = [0.15, -1.0, -1.0]
    config["variables"]["upper_bounds"] = [1.0, 1.0, 0.2]
    config["backend"]["parallel"] = parallel

    config["backend"]["options"] = {
        "termination": {"name": "default.DefaultSingleObjectiveTermination"}
    }
    result1 = optimize(config, initial_values, eval_func())
    assert result1.variables is not None
    assert np.allclose(result1.variables, [0.15, 0.0, 0.2], atol=0.02)

    config["backend"]["options"] = {"termination": {"name": "soo"}}
    result2 = optimize(config, initial_values, eval_func())
    assert result2.variables is not None
    assert np.allclose(result2.variables, [0.15, 0.0, 0.2], atol=0.02)
    assert np.allclose(
        result1.variables,
        result2.variables,
        atol=0.0,
        rtol=1e-10,
    )


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize(
    ("lower_bounds", "upper_bounds"), [(-np.inf, 0.4), (-0.4, np.inf)]
)
def test_pymoo_ineq_nonlinear_constraints(  # ruff: ignore[too-many-positional-arguments]
    config: dict[str, Any],
    lower_bounds: Any,
    upper_bounds: Any,
    eval_func: Any,
    parallel: bool,
    test_functions: Any,
) -> None:
    config["backend"]["parallel"] = parallel
    config["nonlinear_constraints"] = {
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
    }

    weight = 1.0 if upper_bounds == 0.4 else -1.0

    def constraint_function(variables: Any, _: Any) -> float:
        return weight * float(variables[0] + variables[2])

    result = optimize(
        config, initial_values, eval_func(test_functions, [constraint_function])
    )
    assert result.variables is not None
    assert np.allclose(result.variables, [-0.05, 0.0, 0.45], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
def test_pymoo_eq_nonlinear_constraints(
    config: dict[str, Any],
    eval_func: Any,
    parallel: bool,
    test_functions: Any,
) -> None:
    config["backend"]["parallel"] = parallel
    config["nonlinear_constraints"] = {
        "lower_bounds": 1.0,
        "upper_bounds": 1.0,
    }

    def constraint_function(variables: Any, _: Any) -> float:
        return float(variables[0] + variables[2])

    result = optimize(
        config,
        initial_values,
        eval_func(test_functions, [constraint_function]),
        constraint_tolerance=1e-4,
    )
    assert result.variables is not None
    assert np.allclose(result.variables, [0.25, 0.0, 0.75], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
def test_pymoo_ineq_nonlinear_constraints_two_sided(
    config: Any,
    parallel: bool,
    eval_func: Any,
    test_functions: Any,
) -> None:
    config["backend"]["parallel"] = parallel
    config["nonlinear_constraints"] = {
        "lower_bounds": [0.0],
        "upper_bounds": [0.3],
    }

    def constraint_function(variables: Any, _: Any) -> float:
        return float(variables[0] + variables[2])

    result = optimize(
        config, initial_values, eval_func(test_functions, [constraint_function])
    )
    assert result.variables is not None
    assert np.allclose(result.variables, [-0.1, 0.0, 0.4], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
def test_pymoo_le_ge_linear_constraints(
    config: dict[str, Any], eval_func: Any, parallel: bool
) -> None:
    config["backend"]["parallel"] = parallel
    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1]],
        "lower_bounds": [-np.inf],
        "upper_bounds": [0.4],
    }

    result = optimize(config, initial_values, eval_func(), constraint_tolerance=1e-4)
    assert result.variables is not None
    assert np.allclose(result.variables, [-0.05, 0.0, 0.45], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
def test_pymoo_eq_linear_constraints(
    config: dict[str, Any], eval_func: Any, parallel: bool
) -> None:
    config["backend"]["parallel"] = parallel
    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [0, 1, 1]],
        "lower_bounds": [1.0, 0.75],
        "upper_bounds": [1.0, 0.75],
    }

    result = optimize(config, initial_values, eval_func(), constraint_tolerance=1e-4)
    assert result.variables is not None
    assert np.allclose(result.variables, [0.25, 0.0, 0.75], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
def test_pymoo_le_ge_linear_constraints_two_sided(
    config: Any, eval_func: Any, parallel: bool
) -> None:
    config["backend"]["parallel"] = parallel
    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [1, 0, 1]],
        "lower_bounds": [-np.inf, 0.0],
        "upper_bounds": [0.3, np.inf],
    }

    result = optimize(config, initial_values, eval_func())
    assert result.variables is not None
    assert np.allclose(result.variables, [-0.1, 0.0, 0.4], atol=0.02)

    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1]],
        "lower_bounds": [0.0],
        "upper_bounds": [0.3],
    }

    result = optimize(config, initial_values, eval_func())
    assert result.variables is not None
    assert np.allclose(result.variables, [-0.1, 0.0, 0.4], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
def test_pymoo_eq_mixed_constraints(
    config: dict[str, Any],
    eval_func: Any,
    parallel: bool,
    test_functions: Any,
) -> None:
    config["backend"]["parallel"] = parallel
    config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    config["nonlinear_constraints"] = {
        "lower_bounds": [1.0],
        "upper_bounds": [1.0],
    }
    config["linear_constraints"] = {
        "coefficients": [[0, 0, 1]],
        "lower_bounds": [0.75],
        "upper_bounds": [0.75],
    }

    def constraint_function(variables: Any, _: Any) -> float:
        return float(variables[0] + variables[2])

    result = optimize(
        config,
        initial_values,
        eval_func(test_functions, [constraint_function]),
        constraint_tolerance=1e-4,
    )
    assert result.variables is not None
    assert np.allclose(result.variables, [0.25, 0.0, 0.75], atol=0.04)


@pytest.mark.skipif(
    sys.version_info[:2] == (3, 13),
    reason="Fails on Python 3.13 on GitHub for unknown reasons",
)
@pytest.mark.parametrize("parallel", [False, True])
def test_pymoo_constraint_handling(
    config: dict[str, Any],
    eval_func: Any,
    parallel: bool,
    test_functions: Any,
) -> None:
    config["backend"]["parallel"] = parallel
    config["nonlinear_constraints"] = {
        "lower_bounds": -np.inf,
        "upper_bounds": 0.4,
    }
    config["backend"]["options"] = {
        "termination": {"name": "default.DefaultSingleObjectiveTermination"},
        "constraints": {
            "name": "as_penalty.ConstraintsAsPenalty",
            "parameters": {"penalty": 1},
        },
    }

    def constraint_function(variables: Any, _: Any) -> float:
        return float(variables[0] + variables[2])

    result = optimize(
        config,
        initial_values,
        eval_func(test_functions, [constraint_function]),
        constraint_tolerance=1e-4,
    )
    assert result.variables is not None
    assert np.allclose(result.variables, [-0.05, 0.0, 0.45], atol=0.02)


def test_pymoo_bound_constraints_with_failure(
    config: dict[str, Any], eval_func: Any, test_functions: Any
) -> None:
    config["variables"]["lower_bounds"] = [0.15, -1.0, -1.0]
    config["variables"]["upper_bounds"] = [1.0, 1.0, 0.2]
    config["backend"]["method"] = "soo.nonconvex.de.DE"
    config["backend"]["parallel"] = True
    config["optimizer"] = {"max_functions": 1000}
    config["realizations"] = {"realization_min_success": 0}
    result1 = optimize(config, initial_values, eval_func(test_functions))
    assert result1.variables is not None
    assert np.allclose(result1.variables, [0.15, 0.0, 0.2], atol=0.02)

    counter = 0

    def _add_nan(x: Any, _: int) -> Any:
        nonlocal counter
        counter += 1
        if counter == 2:
            counter = 0
            return np.nan
        return test_functions[0](x, 0)

    result2 = optimize(config, initial_values, eval_func((_add_nan, test_functions[1])))
    assert result2.variables is not None
    assert np.allclose(result2.variables, [0.15, 0.0, 0.2], atol=0.02)
    assert not np.all(
        np.equal(
            result1.variables,
            result2.variables,
        )
    )
