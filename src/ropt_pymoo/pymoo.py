"""This module implements the pymoo optimization plugin."""

from __future__ import annotations

import copy
import importlib
import inspect
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Final, TextIO

import numpy as np
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from ropt.plugins.optimizer.base import Optimizer, OptimizerCallback, OptimizerPlugin
from ropt.plugins.optimizer.utils import NormalizedConstraints, create_output_path

from ._config import ParametersConfig

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ropt.config.enopt import EnOptConfig

_OUTPUT_FILE: Final = "optimizer_output"

# These algorithms do not allow NaN fucntion values:
_NO_FAILURE_HANDLING: Final = {"NelderMead"}


class _Problem(Problem):  # type: ignore[misc]
    def __init__(  # noqa: PLR0913
        self,
        n_var: int,
        lower: NDArray[np.float64],
        upper: NDArray[np.float64],
        function: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        constraints: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        is_eq: list[bool] | None,
        *,
        parallel: bool = True,
    ) -> None:
        self._is_eq: list[bool] | None = None
        self._is_ieq: list[bool] | None = None
        n_eq_constr = 0
        n_ieq_constr = 0

        if is_eq is not None:
            self._is_eq = is_eq
            self._is_ieq = [not item for item in self._is_eq]
            n_eq_constr = sum(self._is_eq)
            n_ieq_constr = sum(self._is_ieq)

        if n_eq_constr == 0:
            self._is_eq = None
        if n_ieq_constr == 0:
            self._is_ieq = None

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
        self._is_eq = is_eq
        self._parallel = parallel

    def _evaluate(
        self,
        variables: NDArray[np.float64],
        out: dict[str, Any],
        *_0: Any,  # noqa: ANN401
        **_1: Any,  # noqa: ANN401
    ) -> None:
        variables = variables.astype(np.float64)
        out["F"] = self._function(variables)
        if self._is_eq is not None or self._is_ieq is not None:
            constraints = self._constraints(variables)
        if self._is_eq is not None:
            out["H"] = constraints[:, self._is_eq]
        if self._is_ieq is not None:
            out["G"] = constraints[:, self._is_ieq]

    def __deepcopy__(self, memo: Any) -> _Problem:  # noqa: ANN401
        # Deep copy does not work on the plain object due to the callbacks.
        return _Problem(
            copy.deepcopy(self.n_var),
            lower=copy.deepcopy(self.xl),
            upper=copy.deepcopy(self.xu),
            function=self._function,
            constraints=self._constraints,
            is_eq=copy.deepcopy(self._is_eq),
            parallel=copy.deepcopy(self._parallel),
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
        self._normalized_constraints = self._init_constraints()
        self._parameters = ParametersConfig.model_validate(options)
        self._bounds = self._get_bounds()
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

        if self._config.variables.mask is not None:
            initial_values = initial_values[self._config.variables.mask]

        problem = _Problem(
            n_var=initial_values.size,
            lower=self._bounds[0],
            upper=self._bounds[1],
            function=self._calculate_objective,
            constraints=self._calculate_constraints,
            is_eq=(
                self._normalized_constraints.is_eq
                if self._normalized_constraints is not None
                else None
            ),
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
        if self._config.variables.mask is not None:
            lower_bounds = lower_bounds[self._config.variables.mask]
            upper_bounds = upper_bounds[self._config.variables.mask]
        return lower_bounds, upper_bounds

    def _init_constraints(self) -> NormalizedConstraints | None:
        lower_bounds = []
        upper_bounds = []
        if self._config.nonlinear_constraints is not None:
            lower_bounds.append(self._config.nonlinear_constraints.lower_bounds)
            upper_bounds.append(self._config.nonlinear_constraints.upper_bounds)
        if self._config.linear_constraints is not None:
            mask = self._config.variables.mask
            if mask is not None:
                offsets = np.matmul(
                    self._config.linear_constraints.coefficients[:, ~mask],
                    self._config.variables.initial_values[~mask],
                )
            else:
                offsets = 0
            lower_bounds.append(self._config.linear_constraints.lower_bounds - offsets)
            upper_bounds.append(self._config.linear_constraints.upper_bounds - offsets)
        if lower_bounds:
            return NormalizedConstraints(
                np.concatenate(lower_bounds), np.concatenate(upper_bounds), flip=True
            )
        return None

    def _calculate_objective(
        self, variables: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        functions = self._get_functions(variables)
        if variables.ndim > 1:
            return functions[:, 0]
        return np.array(functions[0])

    def _calculate_constraints(
        self, variables: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if self._normalized_constraints is None:
            return np.array([])
        if self._normalized_constraints.constraints is None:
            constraints = []
            if self._config.nonlinear_constraints is not None:
                functions = self._get_functions(variables)
                constraints.append(
                    (
                        functions[1:] if variables.ndim == 1 else functions[:, 1:]
                    ).transpose()
                )
            if self._config.linear_constraints is not None:
                coefficients = self._config.linear_constraints.coefficients
                if self._config.variables.mask is not None:
                    coefficients = coefficients[:, self._config.variables.mask]
                constraints.append(np.matmul(coefficients, variables.transpose()))
            if constraints:
                self._normalized_constraints.set_constraints(
                    np.concatenate(constraints, axis=0)
                )
        assert self._normalized_constraints.constraints is not None
        return self._normalized_constraints.constraints.transpose()

    def _get_functions(self, variables: NDArray[np.float64]) -> NDArray[np.float64]:
        if (
            self._cached_variables is None
            or variables.shape != self._cached_variables.shape
            or not np.allclose(variables, self._cached_variables)
        ):
            self._cached_variables = None
            self._cached_function = None
            if self._normalized_constraints is not None:
                self._normalized_constraints.reset()
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
    """Pymoo optimizer plugin class."""

    def create(
        self, config: EnOptConfig, optimizer_callback: OptimizerCallback
    ) -> PyMooOptimizer:
        """Initialize the optimizer plugin.

        See the [ropt.plugins.optimizer.base.OptimizerPlugin][] abstract base class.

        # noqa
        """
        return PyMooOptimizer(config, optimizer_callback)

    def is_supported(self, method: str) -> bool:
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
