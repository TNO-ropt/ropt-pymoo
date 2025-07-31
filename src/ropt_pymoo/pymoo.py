"""This module implements the pymoo optimization plugin."""

from __future__ import annotations

import copy
import importlib
import inspect
from typing import TYPE_CHECKING, Any, Final, TextIO

import numpy as np
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from ropt.plugins.optimizer.base import Optimizer, OptimizerPlugin
from ropt.plugins.optimizer.utils import (
    NormalizedConstraints,
    get_masked_linear_constraints,
)

from .config import ParametersConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray
    from ropt.config import EnOptConfig
    from ropt.optimization import OptimizerCallback

# These algorithms do not allow NaN function values:
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
    """Pymoo optimization backend for ropt.

    This class provides an interface to several optimization algorithms from
    [`pymoo`](https://pymoo.org/), enabling their use within `ropt`.

    To select an optimizer, set the `method` field within the
    [`optimizer`][ropt.config.OptimizerConfig] section of the
    [`EnOptConfig`][ropt.config.EnOptConfig] configuration object to the
    desired algorithm's name. The name should be a fully qualified class name
    within the `pymoo.algorithms` module (e.g., `soo.nonconvex.ga.GA`).

    For algorithm-specific options, use the `options` dictionary within the
    [`optimizer`][ropt.config.OptimizerConfig] section, which will be
    parsed into a [`ParametersConfig`][ropt_pymoo.config.ParametersConfig]
    object.
    """

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
        if method == "default":
            msg = "The pymoo backend does not support a 'default' method"
            raise ValueError(msg)
        self._cached_variables: NDArray[np.float64] | None = None
        self._cached_function: NDArray[np.float64] | None = None
        self._stdout: TextIO

        self._parameters = ParametersConfig.model_validate(options, context=method)

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

        self._normalized_constraints = self._init_constraints(initial_values)
        self._bounds = self._get_bounds()

        problem = _Problem(
            n_var=initial_values[self._config.variables.mask].size,
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

        minimize(
            problem,
            self._parameters.get_algorithm(),
            termination=self._parameters.get_termination(),
            seed=self._parameters.seed,
            verbose=True,
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
        lower_bounds = self._config.variables.lower_bounds[self._config.variables.mask]
        upper_bounds = self._config.variables.upper_bounds[self._config.variables.mask]
        return lower_bounds, upper_bounds

    def _get_constraint_bounds(
        self, nonlinear_bounds: tuple[NDArray[np.float64], NDArray[np.float64]] | None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        bounds = []
        if nonlinear_bounds is not None:
            bounds.append(nonlinear_bounds)
        if self._linear_constraint_bounds is not None:
            bounds.append(self._linear_constraint_bounds)
        if bounds:
            lower_bounds, upper_bounds = zip(*bounds, strict=True)
            return np.concatenate(lower_bounds), np.concatenate(upper_bounds)
        return None

    def _init_constraints(
        self, initial_values: NDArray[np.float64]
    ) -> NormalizedConstraints | None:
        self._lin_coef: NDArray[np.float64] | None = None
        self._linear_constraint_bounds: (
            tuple[NDArray[np.float64], NDArray[np.float64]] | None
        ) = None
        if self._config.linear_constraints is not None:
            self._lin_coef, lin_lower, lin_upper = get_masked_linear_constraints(
                self._config, initial_values
            )
            self._linear_constraint_bounds = (lin_lower, lin_upper)
        nonlinear_bounds = (
            None
            if self._config.nonlinear_constraints is None
            else (
                self._config.nonlinear_constraints.lower_bounds,
                self._config.nonlinear_constraints.upper_bounds,
            )
        )
        if (bounds := self._get_constraint_bounds(nonlinear_bounds)) is not None:
            normalized_constraints = NormalizedConstraints(flip=True)
            normalized_constraints.set_bounds(*bounds)
            return normalized_constraints
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
            if self._lin_coef is not None:
                constraints.append(np.matmul(self._lin_coef, variables.transpose()))
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
            callback_result = self._optimizer_callback(
                variables,
                return_functions=True,
                return_gradients=False,
            )
            function = callback_result.functions
            # The optimizer callback may change non-linear constraint bounds:
            if self._normalized_constraints is not None:
                bounds = self._get_constraint_bounds(
                    callback_result.nonlinear_constraint_bounds
                )
                assert bounds is not None
                self._normalized_constraints.set_bounds(*bounds)
            assert function is not None
            if self._allow_nan:
                function = np.where(np.isnan(function), np.inf, function)
            self._cached_function = function.copy()
        return self._cached_function


class PyMooOptimizerPlugin(OptimizerPlugin):
    """Pymoo optimizer plugin class."""

    @classmethod
    def create(
        cls, config: EnOptConfig, optimizer_callback: OptimizerCallback
    ) -> PyMooOptimizer:
        """Initialize the optimizer plugin.

        See the [ropt.plugins.optimizer.base.OptimizerPlugin][] abstract base class.

        # noqa
        """
        return PyMooOptimizer(config, optimizer_callback)

    @classmethod
    def is_supported(cls, method: str) -> bool:
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
        for _, class_ in inspect.getmembers(module, inspect.isclass):
            if class_.__name__ == class_name:
                return True
        return False

    @classmethod
    def validate_options(
        cls, method: str, options: dict[str, Any] | list[str] | None
    ) -> None:
        """Validate the options of a given method.

        See the [ropt.plugins.optimizer.base.OptimizerPlugin][] abstract base class.

        # noqa
        """
        if options is not None:
            if method == "default":
                msg = "The pymoo backend does not support a 'default' method"
                raise ValueError(msg)
            if not isinstance(options, dict):
                msg = "Pymoo optimizer options must be a dictionary"
                raise ValueError(msg)
            _, _, method = method.rpartition("/")
            ParametersConfig.model_validate(options, context=method)
