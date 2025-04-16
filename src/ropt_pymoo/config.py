"""Configuration classes for pymoo algorithms."""

from __future__ import annotations

import importlib
import inspect
from typing import TYPE_CHECKING, Any, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator
from pymoo.core.algorithm import Algorithm  # noqa: TC002
from pymoo.core.operator import Operator  # noqa: TC002
from pymoo.core.termination import Termination  # noqa: TC002
from pymoo.termination import get_termination

if TYPE_CHECKING:
    from collections.abc import Callable


class _ParametersBaseModel(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        str_min_length=1,
        str_strip_whitespace=True,
        validate_default=True,
        frozen=True,
    )


ParameterValues = bool | int | float | str
"""Types of values that can be passed as parameters to pymoo objects."""


class ObjectConfig(_ParametersBaseModel):
    """Configuration for a `pymoo` object used as a parameter.

    This class defines the configuration for `pymoo` objects (like operators,
    sampling methods, etc.) that are passed as parameters to other `pymoo`
    components (e.g., an algorithm).

    The object itself is identified by its fully qualified class name within the
    `pymoo` package structure. For instance, the `FloatRandomSampling` operator
    is specified as `'operators.sampling.rnd.FloatRandomSampling'`.

    Parameters for the object's constructor are provided in a dictionary. Values
    can be basic types (booleans, integers, floats, strings) or another nested
    `ObjectConfig` instance for complex parameter types.

    Attributes:
        object:     The fully qualified class name of the `pymoo` object.
        parameters: A dictionary of keyword arguments.
    """

    object_: str = Field(alias="object")
    parameters: dict[str, ParameterValues | ObjectConfig] = {}


class TerminationConfig(_ParametersBaseModel):
    """Configuration for `pymoo` termination classes.

    This class defines how the termination object for a `pymoo` optimization
    algorithm is specified.

    The `name` attribute identifies the termination criterion by providing the
    full path to the termination class within the `pymoo.termination` module
    (e.g., `max_gen.MaximumGenerationTermination`).

    The `parameters` attribute holds a dictionary of keyword arguments that will
    be passed to the constructor of the chosen termination criterion.

    For details about termination objects, consult the `pymoo` manual:
    [Termination Criterion](https://pymoo.org/interface/termination.html).

    Note:
        Instead of using a termination object, it is also possible to use a
        tuple of termination conditions. See the
        [`ParametersConfig`][ropt_pymoo.config.ParametersConfig] class for
        details.

    Attributes:
        name:       The fully qualified termination class name.
        parameters: Keyword arguments for the termination criterion's constructor.
    """

    name: str = "soo"
    parameters: dict[str, int | float] = {}


class ConstraintsConfig(_ParametersBaseModel):
    """Configuration for `pymoo` constraint handling methods.

    This class defines how constraint handling is configured for a `pymoo`
    optimization.

    The `name` attribute specifies the constraint handling class using its fully
    qualified name within the `pymoo.constraints` module (e.g.,
    `as_penalty.ConstraintsAsPenalty`).

    The `parameters` attribute holds a dictionary of keyword arguments passed to
    the constructor of the chosen constraint handling class.

    For more details on constraint handling, consult the `pymoo` manual:
    [`Constraint Handling`](https://pymoo.org/constraints/index.html).

    Attributes:
        name: The fully qualified name of the constraint handling class.
        parameters: Keyword arguments for the constraint handling class constructor.
    """

    name: str
    parameters: dict[str, bool | int | float | str] = {}


class ParametersConfig(_ParametersBaseModel):
    """Configuration of a `pymoo` algorithm.

    The general structure of the configuration is as follows:

    - **Algorithm Parameters:** Arguments passed directly to the main
        algorithm's constructor are nested under a top-level `parameters` key.

    - **Object Parameters:** When a parameter's value is itself a `pymoo`
        object, specify it using a nested dictionary, which will be parsed into
        a [`ObjectConfig`][ropt_pymoo.config.ObjectConfig] object, containing:

          - An `object` key: The fully qualified name of the `pymoo` class
            (e.g., `"operators.sampling.rnd.IntegerRandomSampling"`).
          - An optional `parameters` key: A dictionary of arguments to pass to
            *that* object's constructor. This can be nested further if those
            arguments are also objects.

    - **Termination, Constraints, Seed:** These are typically defined using
        their own top-level keys (`termination`, `constraints`, `seed`) within
        the `options` dictionary, following the same `object`/`parameters`
        pattern if they require configuration:

          - A `termination` field is parsed into a
            [`TerminationConfig`][ropt_pymoo.config.TerminationConfig] object if
            it is dictionary, or passed to the
            [`pymoo.termination.get_termination()`](https://pymoo.org/interface/termination.html)
            function if it is a tuple of arguments.
          - A `constraints` field is parsed into a
            [`ConstraintsConfig`][ropt_pymoo.config.ConstraintsConfig] object.
          - A `seed` field is parsed into an integer value.

    Note:
        `constraints`, `termination`, and `seed` are optional:

           - If `constraints` is `None`, the default constraint handling of
             `pymoo` will apply.
           - If `termination` is `None`, the default termination criterion `soo`
             is used.
           - The default value of `seed` is `1`.

    Attributes:
        parameters:  The parameters passed to the algorithm constructor
        constraints: Specification of the constraint handling object to use
        termination: Specification of the termination object to use
        seed:        The seed value for the random number generator
    """

    parameters: dict[str, ParameterValues | ObjectConfig] = {}
    constraints: ConstraintsConfig | None = None
    termination: TerminationConfig | tuple[Any, ...] = TerminationConfig()
    seed: int = 1
    _algorithm: str

    @model_validator(mode="after")
    def _set_algorithm(self, info: ValidationInfo) -> Self:
        assert isinstance(info.context, str)
        self._algorithm = info.context
        return self

    @property
    def algorithm(self) -> str:
        """Get the algorithm name.

        Returns:
            The fully qualified algorithm class name.
        """
        return self._algorithm

    def get_algorithm(self) -> Algorithm:
        """Parse the algorithm config and its parameters.

        Raises:
            ValueError: When the algorithm object is not found.

        Returns:
            A `pymoo` algorithm object.
        """
        parameters = _parse_parameters(self.parameters)
        algorithm = _get_class(
            self._algorithm,
            prefix="pymoo.algorithms",
            keyword="algorithm",
        )
        if algorithm is None:
            msg = f"Algorithm not found: {algorithm}"
            raise ValueError(msg)
        return algorithm(**parameters)

    def get_termination(self) -> Termination | tuple[Any, ...]:
        """Parse the termination config.

        Raises:
            ValueError: When the termination object is not found.

        Returns:
            A `pymoo` termination object.
        """
        if isinstance(self.termination, tuple):
            return self.termination
        module_name, _, name = self.termination.name.rpartition(".")
        if not module_name:
            try:
                return get_termination(name, **self.termination.parameters)
            except Exception as exc:
                msg = f"Failed to run get_termination for: {name}"
                raise ValueError(msg) from exc
        termination_class = _get_class(
            self.termination.name,
            prefix="pymoo.termination",
            keyword="name",
        )
        if termination_class is None:
            msg = f"Termination class not found: {termination_class}"
            raise ValueError(msg)
        return termination_class(**self.termination.parameters)

    def get_constraints(self) -> Any:  # noqa: ANN401
        """Parse the constraints config.

        Raises:
            ValueError: When the constraints object is not found.

        Returns:
            A `pymoo` constraints class.
        """
        if self.constraints is None:
            return None
        constraints_class = _get_class(
            self.constraints.name,
            prefix="pymoo.constraints",
            keyword="name",
        )
        if constraints_class is None:
            msg = f"Constraints class not found: {constraints_class}"
            raise ValueError(msg)
        return constraints_class


def _get_class(name: str, prefix: str, keyword: str) -> Callable[..., Any] | None:
    module_name, _, class_name = name.rpartition(".")
    if not module_name:
        msg = f"Not a properly formatted {keyword} name: {name}"
        raise ValueError(msg)
    full_module_name = prefix + "." + module_name
    try:
        module = importlib.import_module(full_module_name)
    except ImportError as exc:
        msg = f"Invalid value for {keyword}: {full_module_name}"
        raise ValueError(msg) from exc
    return next(
        (
            cls
            for _, cls in (inspect.getmembers(module, inspect.isclass))
            if cls.__name__ == class_name
        ),
        None,
    )


def _parse_parameters(
    parameters: dict[str, ParameterValues | ObjectConfig],
) -> dict[str, ParameterValues | Operator]:
    parsed = {}
    for key, value in parameters.items():
        if isinstance(value, ObjectConfig):
            object_class = _get_class(value.object_, prefix="pymoo", keyword="object")
            if object_class is None:
                msg = f"Object class not found: {object}"
                raise ValueError(msg)
            operator_parameters = _parse_parameters(value.parameters)
            parsed[key] = object_class(**operator_parameters)
        else:
            parsed[key] = value
    return parsed
