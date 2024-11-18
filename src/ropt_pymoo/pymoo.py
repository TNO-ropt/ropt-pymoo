"""This module implements the pymoo optimization plugin."""

from __future__ import annotations

import copy
import importlib
import inspect
import os
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Final,
    TextIO,
)

import numpy as np
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from ropt.enums import ConstraintType
from ropt.plugins.optimizer.base import Optimizer, OptimizerCallback, OptimizerPlugin
from ropt.plugins.optimizer.utils import create_output_path, filter_linear_constraints

from ._config import ParametersConfig

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ropt.config.enopt import EnOptConfig

_OUTPUT_FILE: Final = "optimizer_output"

# These algorithms do not allow NaN fucntion values:
_NO_FAILURE_HANDLING: Final = {"NelderMead"}


@dataclass(slots=True)
class _Constraints:
    linear_eq: NDArray[np.bool_] = field(
        default_factory=lambda: np.array([], dtype=np.bool_)
    )
    linear_ineq: NDArray[np.bool_] = field(
        default_factory=lambda: np.array([], dtype=np.bool_)
    )
    nonlinear_eq: NDArray[np.bool_] = field(
        default_factory=lambda: np.array([], dtype=np.bool_)
    )
    nonlinear_ineq: NDArray[np.bool_] = field(
        default_factory=lambda: np.array([], dtype=np.bool_)
    )
    coefficients: NDArray[np.float64] = field(
        default_factory=lambda: np.array([]),
    )
    rhs_values: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    ineq_func: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None
    eq_func: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None


class _Problem(Problem):  # type: ignore[misc]
    def __init__(  # noqa: PLR0913
        self,
        n_var: int,
        lower: NDArray[np.float64],
        upper: NDArray[np.float64],
        function: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        constraints: _Constraints,
        *,
        parallel: bool = True,
    ) -> None:
        n_ieq_constr = np.count_nonzero(constraints.linear_ineq)
        n_ieq_constr += np.count_nonzero(constraints.nonlinear_ineq)
        n_eq_constr = np.count_nonzero(constraints.linear_eq)
        n_eq_constr += np.count_nonzero(constraints.nonlinear_eq)
        super().__init__(
            n_var=n_var,
            n_obj=1,
            n_ieq_constr=n_ieq_constr,
            n_eq_constr=n_eq_constr,
            xl=lower,
            xu=upper,
            elementwise=parallel is False,
        )
        self._function = function
        self._constraints = constraints

    def _evaluate(
        self,
        variables: NDArray[np.float64],
        out: dict[str, Any],
        *_0: Any,  # noqa: ANN401
        **_1: Any,  # noqa: ANN401
    ) -> None:
        variables = variables.astype(np.float64)
        out["F"] = self._function(variables)
        if self._constraints.ineq_func is not None:
            out["G"] = self._constraints.ineq_func(variables)
        if self._constraints.eq_func is not None:
            out["H"] = self._constraints.eq_func(variables)

    def __deepcopy__(self, memo: Any) -> _Problem:  # noqa: ANN401
        # Deep copy does not work on the plain object due to the callbacks.
        return _Problem(
            copy.deepcopy(self.n_var),
            lower=copy.deepcopy(self.xl),
            upper=copy.deepcopy(self.xu),
            function=self._function,
            constraints=self._constraints,
        )


class PyMooOptimizer(Optimizer):
    """Plugin class for optimization via pymoo."""

    def __init__(
        self, config: EnOptConfig, optimizer_callback: OptimizerCallback
    ) -> None:
        """Initialize the optimizer implemented by the pymoo plugin.

        See the [ropt.plugins.optimizer.base.Optimizer][] abstract base class.

        # noqa
        """
        self._config = config
        self._optimizer_callback = optimizer_callback
        options = (
            copy.deepcopy(self._config.optimizer.options)
            if isinstance(self._config.optimizer.options, dict)
            else {}
        )
        _, _, method = self._config.optimizer.method.rpartition("/")
        options["algorithm"] = method
        self._parameters = ParametersConfig.model_validate(options)
        self._bounds = self._get_bounds()
        self._constraints = self._get_constraints()
        self._cached_variables: NDArray[np.float64] | None = None
        self._cached_function: NDArray[np.float64] | None = None
        self._stdout: TextIO

        self._allow_nan = True
        for algorithm in _NO_FAILURE_HANDLING:
            if algorithm in self._parameters.algorithm:
                self._allow_nan = False

    def start(self, initial_values: NDArray[np.float64]) -> None:
        """Start the optimization.

        See the [ropt.plugins.optimizer.base.Optimizer][] abstract base class.

        # noqa
        """
        self._cached_variables = None
        self._cached_function = None

        variable_indices = self._config.variables.indices
        if variable_indices is not None:
            initial_values = initial_values[variable_indices]

        problem = _Problem(
            n_var=initial_values.size,
            lower=self._bounds[0],
            upper=self._bounds[1],
            constraints=self._constraints,
            function=self._calculate_objective,
            parallel=self._config.optimizer.parallel,
        )
        if self._parameters.constraints is not None:
            constraints = self._parameters.get_constraints()
            problem = constraints(problem, **self._parameters.constraints.parameters)

        output_dir = self._config.optimizer.output_dir
        output_file: str | Path
        if output_dir is None:
            output_file = os.devnull
        else:
            output_file = create_output_path(_OUTPUT_FILE, output_dir, suffix=".txt")

        self._stdout = sys.stdout

        with (
            Path(output_file).open("a", encoding="utf-8") as output,
            redirect_stdout(output),
        ):
            minimize(
                problem,
                self._parameters.get_algorithm(),
                termination=self._parameters.get_termination(),
                seed=self._parameters.seed,
                verbose=output_dir is not None,
            )

    @property
    def allow_nan(self) -> bool:
        """Whether NaN is allowed.

        See the [ropt.plugins.optimizer.base.Optimizer][] abstract base class.

        # noqa
        """
        return self._allow_nan

    @property
    def is_parallel(self) -> bool:
        """Whether the current run is parallel.

        See the [ropt.plugins.optimizer.base.Optimizer][] abstract base class.

        # noqa
        """
        return self._config.optimizer.parallel

    def _get_bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        lower_bounds = self._config.variables.lower_bounds
        upper_bounds = self._config.variables.upper_bounds
        variable_indices = self._config.variables.indices
        if variable_indices is not None:
            lower_bounds = lower_bounds[variable_indices]
            upper_bounds = upper_bounds[variable_indices]
        return lower_bounds, upper_bounds

    def _get_constraints(self) -> _Constraints:
        constraints = _Constraints()

        nonlinear_config = self._config.nonlinear_constraints
        if nonlinear_config is not None:
            constraints.nonlinear_eq = nonlinear_config.types == ConstraintType.EQ
            constraints.nonlinear_ineq = nonlinear_config.types != ConstraintType.EQ

        linear_config = self._config.linear_constraints
        if linear_config is not None:
            if self._config.variables.indices is not None:
                linear_config = filter_linear_constraints(
                    linear_config, self._config.variables.indices
                )
            constraints.linear_eq = linear_config.types == ConstraintType.EQ
            constraints.linear_ineq = linear_config.types != ConstraintType.EQ
            constraints.coefficients = linear_config.coefficients.copy()
            constraints.rhs_values = linear_config.rhs_values.copy()
            constraints.coefficients[linear_config.types == ConstraintType.GE] *= -1.0
            constraints.rhs_values[linear_config.types == ConstraintType.GE] *= -1.0

        if np.any(constraints.nonlinear_ineq) or np.any(constraints.linear_ineq):
            constraints.ineq_func = partial(
                self._calculate_constraints,
                nonlinear=constraints.nonlinear_ineq,
                linear=constraints.linear_ineq,
            )
        if np.any(constraints.nonlinear_eq) or np.any(constraints.linear_eq):
            constraints.eq_func = partial(
                self._calculate_constraints,
                nonlinear=constraints.nonlinear_eq,
                linear=constraints.linear_eq,
            )

        return constraints

    def _calculate_objective(
        self, variables: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        functions = self._get_functions(variables)
        if variables.ndim > 1:
            return functions[:, 0]
        return np.array(functions[0])

    def _calculate_constraints(
        self,
        variables: NDArray[np.float64],
        nonlinear: NDArray[np.bool_],
        linear: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        have_nonlinear = np.any(nonlinear)
        have_linear = np.any(linear)
        if have_nonlinear:
            functions = self._get_functions(variables)
            nonlinear_constraints = (
                functions[1:][nonlinear]
                if variables.ndim == 1
                else functions[:, 1:][:, nonlinear]
            )
        if have_linear:
            coeffs = self._constraints.coefficients[linear]
            rhs = self._constraints.rhs_values[linear]
            linear_constraints = (
                np.array(np.matmul(coeffs, variables) - rhs)
                if variables.ndim == 1
                else np.vstack(
                    [
                        np.matmul(coeffs, variables[idx, :]) - rhs
                        for idx in range(variables.shape[0])
                    ],
                )
            )
        if have_nonlinear and have_linear:
            return np.hstack((nonlinear_constraints, linear_constraints))
        if have_nonlinear:
            return nonlinear_constraints
        return linear_constraints

    def _get_functions(self, variables: NDArray[np.float64]) -> NDArray[np.float64]:
        if (
            self._cached_variables is None
            or variables.shape != self._cached_variables.shape
            or not np.allclose(variables, self._cached_variables)
        ):
            self._cached_variables = None
            self._cached_function = None
        if self._cached_function is None:
            self._cached_variables = variables.copy()
            with redirect_stdout(self._stdout):
                function, _ = self._optimizer_callback(
                    variables,
                    return_functions=True,
                    return_gradients=False,
                )
                if self._allow_nan:
                    function = np.where(np.isnan(function), np.inf, function)
            self._cached_function = function.copy()
        return self._cached_function


class PyMooOptimizerPlugin(OptimizerPlugin):
    """Default filter transform plugin class."""

    def create(
        self, config: EnOptConfig, optimizer_callback: OptimizerCallback
    ) -> PyMooOptimizer:
        """Initialize the optimizer plugin.

        See the [ropt.plugins.optimizer.base.OptimizerPlugin][] abstract base class.

        # noqa
        """
        return PyMooOptimizer(config, optimizer_callback)

    def is_supported(self, method: str, *, explicit: bool) -> bool:  # noqa: ARG002
        """Check if a method is supported.

        See the [ropt.plugins.optimizer.base.OptimizerPlugin][] abstract base class.

        # noqa
        """
        module_name, _, class_name = method.rpartition(".")
        if not module_name:
            return False
        full_module_name = f"pymoo.algorithms.{module_name}"
        try:
            module = importlib.import_module(full_module_name)
        except ImportError:
            return False
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__name__ == class_name:
                return True
        return False
