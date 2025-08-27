from typing import Any

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray
from ropt.config import EnOptConfig
from ropt.enums import EventType, ExitCode
from ropt.plan import BasicOptimizer, Event
from ropt.results import FunctionResults
from ropt.transforms import OptModelTransforms
from ropt.transforms.base import NonLinearConstraintTransform, ObjectiveTransform

initial_values = [0.2, 0.0, 0.1]


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> dict[str, Any]:
    return {
        "variables": {
            "variable_count": len(initial_values),
            "lower_bounds": [-1.0, -1.0, -1.0],
            "upper_bounds": [1.0, 1.0, 1.0],
        },
        "optimizer": {
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
    enopt_config["optimizer"]["method"] = f"{external}soo.nonconvex.nelder.NelderMead"
    enopt_config["variables"]["lower_bounds"] = [0.15, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 0.2]
    enopt_config["optimizer"]["parallel"] = parallel
    variables = BasicOptimizer(enopt_config, evaluator()).run(initial_values).variables
    assert variables is not None
    assert np.allclose(variables, [0.15, 0.0, 0.2], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
def test_pymoo_termination(
    enopt_config: dict[str, Any], evaluator: Any, parallel: bool
) -> None:
    enopt_config["variables"]["lower_bounds"] = [0.15, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 0.2]
    enopt_config["optimizer"]["parallel"] = parallel

    enopt_config["optimizer"]["options"] = {
        "termination": {"name": "default.DefaultSingleObjectiveTermination"}
    }
    variables1 = BasicOptimizer(enopt_config, evaluator()).run(initial_values).variables
    assert variables1 is not None
    assert np.allclose(variables1, [0.15, 0.0, 0.2], atol=0.02)

    enopt_config["optimizer"]["options"] = {"termination": {"name": "soo"}}
    variables2 = BasicOptimizer(enopt_config, evaluator()).run(initial_values).variables
    assert variables2 is not None
    assert np.allclose(variables2, [0.15, 0.0, 0.2], atol=0.02)
    assert np.allclose(variables1, variables2, atol=0.0, rtol=1e-10)


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize(
    ("lower_bounds", "upper_bounds"), [(-np.inf, 0.4), (-0.4, np.inf)]
)
def test_pymoo_ineq_nonlinear_constraints(
    enopt_config: dict[str, Any],
    lower_bounds: Any,
    upper_bounds: Any,
    evaluator: Any,
    parallel: bool,
    test_functions: Any,
) -> None:
    enopt_config["optimizer"]["parallel"] = parallel
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
    }

    weight = 1.0 if upper_bounds == 0.4 else -1.0
    test_functions = (
        *test_functions,
        lambda variables: weight * variables[0] + weight * variables[2],
    )
    variables = (
        BasicOptimizer(enopt_config, evaluator(test_functions))
        .run(initial_values)
        .variables
    )
    assert variables is not None
    assert np.allclose(variables, [-0.05, 0.0, 0.45], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
def test_pymoo_eq_nonlinear_constraints(
    enopt_config: dict[str, Any],
    evaluator: Any,
    parallel: bool,
    test_functions: Any,
) -> None:
    enopt_config["optimizer"]["parallel"] = parallel
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": 1.0,
        "upper_bounds": 1.0,
    }

    test_functions = (
        *test_functions,
        lambda variables: variables[0] + variables[2],
    )
    variables = (
        BasicOptimizer(
            enopt_config, evaluator(test_functions), constraint_tolerance=1e-4
        )
        .run(initial_values)
        .variables
    )
    assert variables is not None
    assert np.allclose(variables, [0.25, 0.0, 0.75], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
def test_pymoo_ineq_nonlinear_constraints_two_sided(
    enopt_config: Any,
    parallel: bool,
    evaluator: Any,
    test_functions: Any,
) -> None:
    enopt_config["optimizer"]["parallel"] = parallel
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": [0.0],
        "upper_bounds": [0.3],
    }
    test_functions = (
        *test_functions,
        lambda variables: variables[0] + variables[2],
    )

    variables = (
        BasicOptimizer(enopt_config, evaluator(test_functions))
        .run(initial_values)
        .variables
    )
    assert variables is not None
    assert np.allclose(variables, [-0.1, 0.0, 0.4], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
def test_pymoo_le_ge_linear_constraints(
    enopt_config: dict[str, Any], evaluator: Any, parallel: bool
) -> None:
    enopt_config["optimizer"]["parallel"] = parallel
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1]],
        "lower_bounds": [-np.inf],
        "upper_bounds": [0.4],
    }

    variables = (
        BasicOptimizer(enopt_config, evaluator(), constraint_tolerance=1e-4)
        .run(initial_values)
        .variables
    )
    assert variables is not None
    assert np.allclose(variables, [-0.05, 0.0, 0.45], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
def test_pymoo_eq_linear_constraints(
    enopt_config: dict[str, Any], evaluator: Any, parallel: bool
) -> None:
    enopt_config["optimizer"]["parallel"] = parallel
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [0, 1, 1]],
        "lower_bounds": [1.0, 0.75],
        "upper_bounds": [1.0, 0.75],
    }

    variables = (
        BasicOptimizer(enopt_config, evaluator(), constraint_tolerance=1e-4)
        .run(initial_values)
        .variables
    )
    assert variables is not None
    assert np.allclose(variables, [0.25, 0.0, 0.75], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
def test_pymoo_le_ge_linear_constraints_two_sided(
    enopt_config: Any, evaluator: Any, parallel: bool
) -> None:
    enopt_config["optimizer"]["parallel"] = parallel
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [1, 0, 1]],
        "lower_bounds": [-np.inf, 0.0],
        "upper_bounds": [0.3, np.inf],
    }

    variables = BasicOptimizer(enopt_config, evaluator()).run(initial_values).variables
    assert variables is not None
    assert np.allclose(variables, [-0.1, 0.0, 0.4], atol=0.02)

    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1]],
        "lower_bounds": [0.0],
        "upper_bounds": [0.3],
    }

    variables = BasicOptimizer(enopt_config, evaluator()).run(initial_values).variables
    assert variables is not None
    assert np.allclose(variables, [-0.1, 0.0, 0.4], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
def test_pymoo_eq_mixed_constraints(
    enopt_config: dict[str, Any],
    evaluator: Any,
    parallel: bool,
    test_functions: Any,
) -> None:
    enopt_config["optimizer"]["parallel"] = parallel
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
        lambda variables: variables[0] + variables[2],
    )
    variables = (
        BasicOptimizer(
            enopt_config, evaluator(test_functions), constraint_tolerance=1e-4
        )
        .run(initial_values)
        .variables
    )
    assert variables is not None
    assert np.allclose(variables, [0.25, 0.0, 0.75], atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
def test_pymoo_constraint_handling(
    enopt_config: dict[str, Any],
    evaluator: Any,
    parallel: bool,
    test_functions: Any,
) -> None:
    enopt_config["optimizer"]["parallel"] = parallel
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": -np.inf,
        "upper_bounds": 0.4,
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
        lambda variables: variables[0] + variables[2],
    )

    variables = (
        BasicOptimizer(
            enopt_config, evaluator(test_functions), constraint_tolerance=1e-4
        )
        .run(initial_values)
        .variables
    )
    assert variables is not None
    assert np.allclose(variables, [-0.05, 0.0, 0.45], atol=0.02)


def test_pymoo_bound_constraints_with_failure(
    enopt_config: dict[str, Any], evaluator: Any, test_functions: Any
) -> None:
    enopt_config["variables"]["lower_bounds"] = [0.15, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 0.2]
    enopt_config["optimizer"]["method"] = "soo.nonconvex.de.DE"
    enopt_config["optimizer"]["parallel"] = True
    enopt_config["optimizer"]["max_functions"] = 800
    enopt_config["realizations"] = {"realization_min_success": 0}
    variables1 = (
        BasicOptimizer(enopt_config, evaluator(test_functions))
        .run(initial_values)
        .variables
    )
    assert variables1 is not None
    assert np.allclose(variables1, [0.15, 0.0, 0.2], atol=0.02)

    counter = 0

    def _add_nan(x: Any) -> Any:
        nonlocal counter
        counter += 1
        if counter == 2:
            counter = 0
            return np.nan
        return test_functions[0](x)

    variables2 = (
        BasicOptimizer(enopt_config, evaluator((_add_nan, test_functions[1])))
        .run(initial_values)
        .variables
    )
    assert variables2 is not None
    assert np.allclose(variables2, [0.15, 0.0, 0.2], atol=0.02)
    assert not np.all(np.equal(variables1, variables2))


def test_pymoo_bound_constraints_no_failure_handling(
    enopt_config: dict[str, Any], evaluator: Any, test_functions: Any
) -> None:
    enopt_config["variables"]["lower_bounds"] = [0.15, -1.0, -1.0]
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 0.2]
    enopt_config["optimizer"]["method"] = "soo.nonconvex.nelder.NelderMead"
    enopt_config["optimizer"]["parallel"] = True
    enopt_config["optimizer"]["max_functions"] = 800

    variables1 = (
        BasicOptimizer(enopt_config, evaluator(test_functions))
        .run(initial_values)
        .variables
    )
    assert variables1 is not None
    assert np.allclose(variables1, [0.15, 0.0, 0.2], atol=0.02)

    enopt_config["realizations"] = {"realization_min_success": 0}

    counter = 0

    def _add_nan(x: Any) -> Any:
        nonlocal counter
        counter += 1
        if counter == 2:
            counter = 0
            return np.nan
        return test_functions[0](x)

    plan = BasicOptimizer(enopt_config, evaluator((_add_nan, test_functions[1])))
    plan.run(initial_values)
    assert plan.exit_code == ExitCode.TOO_FEW_REALIZATIONS
    variables2 = plan.variables
    assert variables2 is not None
    assert not np.all(np.equal(variables1, variables2))


class ObjectiveScaler(ObjectiveTransform):
    def __init__(self, scales: ArrayLike) -> None:
        self._scales = np.asarray(scales, dtype=np.float64)
        self._set = True

    def set_scales(self, scales: ArrayLike) -> None:
        if self._set:
            self._scales = np.asarray(scales, dtype=np.float64)
            self._set = False

    def to_optimizer(self, objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        return objectives / self._scales

    def from_optimizer(self, objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        return objectives * self._scales


@pytest.mark.parametrize("parallel", [False, True])
def test_pymoo_objective_with_scaler(
    enopt_config: Any, evaluator: Any, parallel: bool, test_functions: Any
) -> None:
    enopt_config["optimizer"]["parallel"] = parallel
    enopt_config["optimizer"]["method"] = "soo.nonconvex.nelder.NelderMead"
    results1 = BasicOptimizer(enopt_config, evaluator()).run(initial_values).results
    assert results1 is not None
    assert results1.functions is not None
    variables1 = results1.evaluations.variables
    objectives1 = results1.functions.objectives
    assert np.allclose(variables1, [0.0, 0.0, 0.5], atol=0.02)
    assert np.allclose(objectives1, [0.5, 4.5], atol=0.02)

    def function1(variables: NDArray[np.float64]) -> float:
        return float(test_functions[0](variables))

    def function2(variables: NDArray[np.float64]) -> float:
        return float(test_functions[1](variables))

    init1 = test_functions[1](initial_values)
    transforms = OptModelTransforms(
        objectives=ObjectiveScaler(np.array([init1, init1]))
    )

    checked = False

    def check_value(event: Event) -> None:
        nonlocal checked
        results = event.data.get("results", ())
        for item in results:
            if isinstance(item, FunctionResults) and not checked:
                checked = True
                assert item.functions is not None
                assert item.functions.objectives is not None
                objective = test_functions[1](item.evaluations.variables)
                assert np.allclose(item.functions.objectives[-1], objective / init1)
                transformed = item.transform_from_optimizer(
                    event.data["config"], event.data["transforms"]
                )
                assert transformed.functions is not None
                assert transformed.functions.objectives is not None
                assert np.allclose(transformed.functions.objectives[-1], objective)

    optimizer = BasicOptimizer(
        enopt_config, evaluator([function1, function2]), transforms=transforms
    )
    optimizer._observers.append(  # noqa: SLF001
        (EventType.FINISHED_EVALUATION, check_value)
    )
    results2 = optimizer.run(initial_values).results
    assert results2 is not None
    assert np.allclose(results2.evaluations.variables, variables1, atol=0.02)
    assert results2.functions is not None
    assert np.allclose(objectives1, results2.functions.objectives, atol=0.025)


@pytest.mark.parametrize("parallel", [False, True])
def test_pymoo_objective_with_lazy_scaler(
    enopt_config: Any, evaluator: Any, parallel: bool, test_functions: Any
) -> None:
    enopt_config["optimizer"]["parallel"] = parallel
    enopt_config["optimizer"]["method"] = "soo.nonconvex.nelder.NelderMead"
    results1 = BasicOptimizer(enopt_config, evaluator()).run(initial_values).results
    assert results1 is not None
    assert results1.functions is not None
    variables1 = results1.evaluations.variables
    objectives1 = results1.functions.objectives
    assert np.allclose(variables1, [0.0, 0.0, 0.5], atol=0.02)
    assert np.allclose(objectives1, [0.5, 4.5], atol=0.02)

    objective_transform = ObjectiveScaler(np.array([1.0, 1.0]))
    transforms = OptModelTransforms(objectives=objective_transform)

    def function1(variables: NDArray[np.float64]) -> float:
        objective1 = test_functions[1](variables)
        objective_transform.set_scales([objective1, objective1])
        return float(test_functions[0](variables))

    def function2(variables: NDArray[np.float64]) -> float:
        return float(test_functions[1](variables))

    checked = False

    def check_value(event: Event) -> None:
        nonlocal checked
        results = event.data.get("results", ())
        for item in results:
            if isinstance(item, FunctionResults) and not checked:
                checked = True
                assert item.functions is not None
                assert item.functions.objectives is not None
                assert np.allclose(item.functions.objectives[-1], 1.0)
                transformed = item.transform_from_optimizer(
                    event.data["config"], event.data["transforms"]
                )
                assert transformed.functions is not None
                assert transformed.functions.objectives is not None
                assert np.allclose(
                    transformed.functions.objectives[-1],
                    test_functions[1](item.evaluations.variables),
                )

    optimizer = BasicOptimizer(
        enopt_config, evaluator([function1, function2]), transforms=transforms
    )
    optimizer._observers.append(  # noqa: SLF001
        (EventType.FINISHED_EVALUATION, check_value)
    )
    results2 = optimizer.run(initial_values).results
    assert results2 is not None
    assert np.allclose(results2.evaluations.variables, variables1, atol=0.02)
    assert results2.functions is not None
    assert np.allclose(objectives1, results2.functions.objectives, atol=0.025)


class ConstraintScaler(NonLinearConstraintTransform):
    def __init__(self, scales: ArrayLike) -> None:
        self._scales = np.asarray(scales, dtype=np.float64)
        self._set = True

    def set_scales(self, scales: ArrayLike) -> None:
        if self._set:
            self._scales = np.asarray(scales, dtype=np.float64)
            self._set = False

    def bounds_to_optimizer(
        self, lower_bounds: NDArray[np.float64], upper_bounds: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return lower_bounds / self._scales, upper_bounds / self._scales

    def to_optimizer(self, constraints: NDArray[np.float64]) -> NDArray[np.float64]:
        return constraints / self._scales

    def from_optimizer(self, constraints: NDArray[np.float64]) -> NDArray[np.float64]:
        return constraints * self._scales

    def nonlinear_constraint_diffs_from_optimizer(
        self, lower_diffs: NDArray[np.float64], upper_diffs: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return lower_diffs * self._scales, upper_diffs * self._scales


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize(
    "external", ["", pytest.param("external/", marks=pytest.mark.external)]
)
def test_pymoo_nonlinear_constraint_with_scaler(
    enopt_config: Any,
    evaluator: Any,
    parallel: bool,
    test_functions: Any,
    external: str,
) -> None:
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": 0.0,
        "upper_bounds": 0.4,
    }
    enopt_config["optimizer"]["parallel"] = parallel
    enopt_config["optimizer"]["method"] = f"{external}soo.nonconvex.nelder.NelderMead"

    functions = (
        *test_functions,
        lambda variables: variables[0] + variables[2],
    )

    results1 = (
        BasicOptimizer(enopt_config, evaluator(functions)).run(initial_values).results
    )
    assert results1 is not None
    assert results1.evaluations.variables[[0, 2]].sum() > 0.0 - 1e-5
    assert results1.evaluations.variables[[0, 2]].sum() < 0.4 + 1e-5

    scales = np.array(functions[-1](initial_values), ndmin=1)
    transforms = OptModelTransforms(nonlinear_constraints=ConstraintScaler(scales))
    config = EnOptConfig.model_validate(enopt_config, context=transforms)
    assert config.nonlinear_constraints is not None
    assert config.nonlinear_constraints.upper_bounds == 0.4
    assert transforms.nonlinear_constraints is not None
    bounds = transforms.nonlinear_constraints.bounds_to_optimizer(
        config.nonlinear_constraints.lower_bounds,
        config.nonlinear_constraints.upper_bounds,
    )
    assert bounds is not None
    assert bounds[1] == 0.4 / scales

    check = True

    def check_constraints(event: Event) -> None:
        nonlocal check
        results = event.data.get("results", ())
        for item in results:
            if isinstance(item, FunctionResults) and check:
                check = False
                assert item.functions is not None
                assert item.functions.constraints is not None
                constraints = functions[-1](item.evaluations.variables)
                assert np.allclose(item.functions.constraints, constraints / scales)
                transformed = item.transform_from_optimizer(
                    event.data["config"], event.data["transforms"]
                )
                assert transformed.functions is not None
                assert transformed.functions.constraints is not None
                assert np.allclose(transformed.functions.constraints, constraints)

    optimizer = BasicOptimizer(
        enopt_config, evaluator(functions), transforms=transforms
    )
    optimizer._observers.append(  # noqa: SLF001
        (EventType.FINISHED_EVALUATION, check_constraints)
    )
    results2 = optimizer.run(initial_values).results
    assert results2 is not None
    assert np.allclose(
        results2.evaluations.variables, results1.evaluations.variables, atol=0.02
    )
    assert results1.functions is not None
    assert results2.functions is not None
    assert np.allclose(
        results1.functions.objectives, results2.functions.objectives, atol=0.025
    )


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize(
    "external", ["", pytest.param("external/", marks=pytest.mark.external)]
)
def test_pymoo_nonlinear_constraint_with_lazy_scaler(
    enopt_config: Any,
    evaluator: Any,
    parallel: bool,
    test_functions: Any,
    external: str,
) -> None:
    enopt_config["nonlinear_constraints"] = {
        "lower_bounds": 0.0,
        "upper_bounds": 0.4,
    }
    enopt_config["optimizer"]["parallel"] = parallel
    enopt_config["optimizer"]["method"] = f"{external}soo.nonconvex.nelder.NelderMead"

    functions = (
        *test_functions,
        lambda variables: variables[0] + variables[2],
    )

    results1 = (
        BasicOptimizer(enopt_config, evaluator(functions)).run(initial_values).results
    )
    assert results1 is not None
    assert results1.evaluations.variables[[0, 2]].sum() > 0.0 - 1e-5
    assert results1.evaluations.variables[[0, 2]].sum() < 0.4 + 1e-5

    scaler = ConstraintScaler([1.0])
    transforms = OptModelTransforms(nonlinear_constraints=scaler)

    config = EnOptConfig.model_validate(enopt_config, context=transforms)
    assert config.nonlinear_constraints is not None
    assert config.nonlinear_constraints.upper_bounds == 0.4
    assert transforms.nonlinear_constraints is not None
    bounds = transforms.nonlinear_constraints.bounds_to_optimizer(
        config.nonlinear_constraints.lower_bounds,
        config.nonlinear_constraints.upper_bounds,
    )
    assert bounds is not None
    assert bounds[1] == 0.4

    def constraint_function(variables: NDArray[np.float64]) -> float:
        value = float(variables[0] + variables[2])
        scaler.set_scales(value)
        return value

    functions = (*test_functions, constraint_function)

    check = True

    def check_constraints(event: Event) -> None:
        nonlocal check
        config = event.data["config"]
        results = event.data.get("results", ())
        for item in results:
            if isinstance(item, FunctionResults) and check:
                check = False
                assert config.nonlinear_constraints is not None
                assert transforms.nonlinear_constraints is not None
                _, upper_bounds = transforms.nonlinear_constraints.bounds_to_optimizer(
                    config.nonlinear_constraints.lower_bounds,
                    config.nonlinear_constraints.upper_bounds,
                )
                value = float(
                    item.evaluations.variables[0] + item.evaluations.variables[2]
                )
                assert np.allclose(upper_bounds, 0.4 / value)
                assert item.functions is not None
                assert item.functions.constraints is not None
                assert np.allclose(item.functions.constraints, 1.0)
                transformed = item.transform_from_optimizer(
                    event.data["config"], event.data["transforms"]
                )
                assert transformed.functions is not None
                assert transformed.functions.constraints is not None
                assert np.allclose(transformed.functions.constraints, value)

    optimizer = BasicOptimizer(
        enopt_config, evaluator(functions), transforms=transforms
    )
    optimizer._observers.append(  # noqa: SLF001
        (EventType.FINISHED_EVALUATION, check_constraints)
    )
    results2 = optimizer.run(initial_values).results
    assert results2 is not None
    assert np.allclose(
        results2.evaluations.variables, results1.evaluations.variables, atol=0.02
    )
    assert results1.functions is not None
    assert results2.functions is not None
    assert np.allclose(
        results1.functions.objectives, results2.functions.objectives, atol=0.025
    )
