"""This module implements the pymoo optimization plugin."""

from __future__ import annotations

import copy
import importlib
import inspect
import logging
from typing import TYPE_CHECKING, Any, ClassVar, TextIO

import numpy as np
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from ropt.backend import Backend
from ropt.backend.utils import (
    get_linear_constraints,
    get_nonlinear_equalities,
    resolve_verbosity,
    split_linear_constraints,
)

from .config import ParametersConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray
    from ropt.config import BackendConfig
    from ropt.context import EnOptContext
    from ropt.core import OptimizerCallback
    from ropt.plugins import MethodSpec

_logger = logging.getLogger("ropt.backend.pymoo")


class _Problem(Problem):  # type: ignore[misc]
    def __init__(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
        self,
        n_var: int,
        lower: NDArray[np.float64],
        upper: NDArray[np.float64],
        function: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        constraints: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        is_eq: NDArray[np.bool_] | None,
        *,
        parallel: bool = True,
    ) -> None:
        self._function = function
        self._constraints = constraints
        self._is_eq: NDArray[np.bool_] | None = None
        self._is_ieq: NDArray[np.bool_] | None = None

        n_eq_constr = 0
        n_ieq_constr = 0

        if is_eq is not None:
            self._is_eq = is_eq
            self._is_ieq = ~is_eq
            n_eq_constr = int(np.sum(self._is_eq))
            n_ieq_constr = int(np.sum(self._is_ieq))

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
        )

        self._parallel = parallel
        self._n_constraints = n_eq_constr + n_ieq_constr

    def __deepcopy__(self, memo: dict[int, Any]) -> _Problem:
        # Pymoo deep-copies the problem, for instance when wrapping it in a meta
        # problem. The callbacks are bound to the live optimizer, which owns
        # locks and other uncopyable state, so they are shared instead of copied.
        copied = self.__class__.__new__(self.__class__)
        memo[id(self)] = copied
        for key, value in self.__dict__.items():
            copied.__dict__[key] = (
                value
                if key in {"_function", "_constraints"}
                else copy.deepcopy(value, memo)
            )
        return copied

    def _evaluate(
        self,
        variables: NDArray[np.float64],
        out: dict[str, Any],
        *_0: Any,  # ruff: ignore[any-type]
        **_1: Any,  # ruff: ignore[any-type]
    ) -> None:
        variables = variables.astype(np.float64)
        if self._parallel:
            functions = self._function(variables)
            if self._is_eq is not None or self._is_ieq is not None:
                constraints = self._constraints(variables)
        else:
            functions = np.zeros(variables.shape[0])
            if self._n_constraints > 0:
                constraints = np.zeros((variables.shape[0], self._n_constraints))
            for idx in range(variables.shape[0]):
                functions[idx] = self._function(variables[idx])
                if self._n_constraints > 0:
                    constraints[idx, :] = self._constraints(variables[idx])
        out["F"] = functions
        if self._is_eq is not None:
            out["H"] = constraints[:, self._is_eq]
        if self._is_ieq is not None:
            out["G"] = constraints[:, self._is_ieq]


def _reports(*, verbose: bool | int | None) -> bool:
    level = resolve_verbosity(verbose=verbose)
    return level is None or level > 0


def _algorithm_exists(method: str) -> bool:
    """Report whether `method` names a pymoo algorithm class.

    A predicate rather than a set: the algorithms available are whatever the
    installed `pymoo` provides, which cannot be enumerated in advance. The name
    is `module.path.ClassName` and is matched against the class name exactly,
    so its casing is significant and must not be folded.

    Args:
        method: The method name, without the `pymoo/` prefix.

    Returns:
        Whether the installed pymoo provides a class of that name.
    """
    module_name, _, class_name = method.rpartition(".")
    if not module_name:
        return False
    full_module_name = f"pymoo.algorithms.{module_name}"
    try:
        module = importlib.import_module(full_module_name)
    except ImportError:
        return False
    return any(
        class_.__name__ == class_name
        for _, class_ in inspect.getmembers(module, inspect.isclass)
    )


class PyMooBackend(Backend):
    """Pymoo optimization backend for ropt.

    This class provides an interface to several optimization algorithms from
    [`pymoo`](https://pymoo.org/), enabling their use within `ropt`.

    !!! note "Optimizer output"
        `pymoo` reports its progress when the `verbose` setting of
        [`BackendConfig`][ropt.config.BackendConfig] asks for it. `pymoo` has no
        reporting levels, so the setting is on or off.

    To select an optimizer, set the `method` field within the
    [`optimizer`][ropt.config.BackendConfig] section of the
    [`EnOptContext`][ropt.context.EnOptContext] configuration object to the
    desired algorithm's name. The name should be a fully qualified class name
    within the `pymoo.algorithms` module (e.g., `soo.nonconvex.ga.GA`).

    For algorithm-specific options, use the `options` dictionary within the
    [`optimizer`][ropt.config.BackendConfig] section, which will be
    parsed into a [`ParametersConfig`][ropt_pymoo.config.ParametersConfig]
    object.
    """

    methods: ClassVar[MethodSpec] = staticmethod(_algorithm_exists)

    def __init__(self, backend_config: BackendConfig) -> None:
        """Initialize the Pymoo optimizer backend.

        Args:
            backend_config: The configuration for the backend, containing the
                            method name and options.

        Raises:
            ValueError: If the method is "default".
        """
        self._config = backend_config
        _, _, method = self._config.method.rpartition("/")
        if method == "default":
            msg = "The pymoo backend does not support a 'default' method"
            raise ValueError(msg)

    def init(
        self, context: EnOptContext, optimizer_callback: OptimizerCallback
    ) -> None:
        """Initialize the optimizer implemented by the pymoo plugin.

        See the [ropt.backend.Backend][] abstract base class.

        # noqa
        """
        self._context = context
        self._optimizer_callback = optimizer_callback
        options = (
            copy.deepcopy(self._config.options)
            if isinstance(self._config.options, dict)
            else {}
        )
        self._cached_variables: NDArray[np.float64] | None = None
        self._cached_function: NDArray[np.float64] | None = None
        self._stdout: TextIO

        _, _, method = self._config.method.rpartition("/")
        self._parameters = ParametersConfig.model_validate(options, context=method)
        _logger.debug("Using PyMoo algorithm: %s", method)

    def start(self, initial_values: NDArray[np.float64]) -> None:
        """Start the optimization.

        See the [ropt.backend.Backend][] abstract base class.

        # noqa
        """
        self._cached_variables = None
        self._cached_function = None

        self._is_eq = self._init_constraints(initial_values)
        self._bounds = self._get_bounds()

        problem = _Problem(
            n_var=initial_values[self._context.variables.mask].size,
            lower=self._bounds[0],
            upper=self._bounds[1],
            function=self._calculate_objective,
            constraints=self._calculate_constraints,
            is_eq=self._is_eq,
            parallel=self._config.parallel,
        )
        if self._parameters.constraints is not None:
            constraints = self._parameters.get_constraints()
            problem = constraints(problem, **self._parameters.constraints.parameters)

        minimize(
            problem,
            self._parameters.get_algorithm(),
            termination=self._parameters.get_termination(),
            seed=self._parameters.seed,
            verbose=_reports(verbose=self._config.verbose),
        )

    @property
    def is_parallel(self) -> bool:
        """Whether the current run is parallel.

        See the [ropt.backend.Backend][] abstract base class.

        # noqa
        """
        return self._config.parallel

    def validate_options(self) -> None:
        """Validate the options of a given method.

        See the [ropt.backend.Backend][] abstract base class.

        # noqa
        """  # ruff: ignore[docstring-missing-exception]
        if self._config.options is not None:
            _, _, method = self._config.method.rpartition("/")
            if not isinstance(self._config.options, dict):
                msg = "Pymoo optimizer options must be a dictionary"
                raise ValueError(msg)
            ParametersConfig.model_validate(self._config.options, context=method)

    def _get_bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        lower_bounds = self._context.variables.lower_bounds[
            self._context.variables.mask
        ]
        upper_bounds = self._context.variables.upper_bounds[
            self._context.variables.mask
        ]
        return lower_bounds, upper_bounds

    def _init_constraints(
        self, initial_values: NDArray[np.float64]
    ) -> NDArray[np.bool_] | None:
        is_eq = get_nonlinear_equalities(self._context)
        self._nonlinear_constraint_count = 0 if is_eq is None else int(is_eq.size)
        self._linear_coefficients: NDArray[np.float64] | None = None
        self._linear_offsets: NDArray[np.float64] | None = None
        if self._context.linear_constraints is not None:
            coefficients, offsets, linear_is_eq = split_linear_constraints(
                *get_linear_constraints(self._context, initial_values)
            )
            self._linear_coefficients = coefficients
            self._linear_offsets = offsets
            is_eq = (
                linear_is_eq if is_eq is None else np.concatenate((is_eq, linear_is_eq))
            )
        return None if is_eq is None or is_eq.size == 0 else is_eq

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
        if self._is_eq is None:
            return np.array([])
        blocks = []
        if self._nonlinear_constraint_count:
            functions = self._get_functions(variables)
            values = functions[1:] if variables.ndim == 1 else functions[:, 1:]
            blocks.append(np.atleast_2d(values).T if variables.ndim == 1 else values.T)
        if self._linear_coefficients is not None:
            assert self._linear_offsets is not None
            points = variables if variables.ndim > 1 else np.expand_dims(variables, 0)
            blocks.append(
                np.matmul(self._linear_coefficients, points.T)
                - self._linear_offsets[:, np.newaxis]
            )
        # Pymoo treats a constraint as satisfied when it is non-positive.
        return -np.concatenate(blocks, axis=0).transpose()

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
            callback_result = self._optimizer_callback(
                variables,
                return_functions=True,
                return_gradients=False,
            )
            function = callback_result.functions
            assert function is not None
            self._cached_function = function.copy()
        return self._cached_function
