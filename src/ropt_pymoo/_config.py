from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field
from pymoo.core.algorithm import Algorithm  # noqa: TC002
from pymoo.core.operator import Operator  # noqa: TC002
from pymoo.core.termination import Termination  # noqa: TC002
from pymoo.termination import get_termination


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


class ObjectConfig(_ParametersBaseModel):
    """Configuration for an operator.

    Configures a pymoo object, which can be used a a parameter. They are
    specified by their fully qualified name within the `pymoo` module. For
    example to specify the `FloatRandomSampling` operator, the operator should
    be specified as 'operators.sampling.rnd.FloatRandomSampling`.

    The parameters are specified as a dict that will be passed as keyword
    arguments to the operator. These may consist of the basic types, such as
    booleans, integers, floats and strings, or as a nested object specification.

    Attributes:
        object:     The fully qualified operator class name.
        parameters: The parameters passed to the operator constructor.
    """

    object_: str = Field(alias="object")
    parameters: dict[str, ParameterValues | ObjectConfig] = {}


class TerminationConfig(_ParametersBaseModel):
    """Configuration of termination conditions.

    Termination objects are used within `pymoo` to determine when an
    optimization algorithm should terminate. They are specified by a string for
    a set of defaults, which are specified here by there fully qualified name
    within the `pymoo.termination` module. For example to specify the
    `MaximumGenerationTermination` termination, the it should be specified here
    as 'max_gen.MaximumGenerationTermination`.

    In addition to the fully qualified name it is also possible to use the short
    name of the termination object, as it is used by the `get_termination`
    function, e.g. `n_eval`.

    For details about termination objects, consult the `pymoo` manual:
    [`Operators`](https://pymoo.org/interface/termination.html)


    Note:
        Instead of using a termination object, it is also possible to use a
        tuple of termination conditions. See the `pymoo` manual for details.

    Attributes:
        name:       The fully qualified termination class name.
        parameters: The parameters passed to the termination constructor.
    """

    name: str = "soo"
    parameters: dict[str, int | float] = {}


class ConstraintsConfig(_ParametersBaseModel):
    """Configuration of constraint handling.

    Constraint objects are used within `pymoo` to determine how constraints
    should be handled. Here they are specified by a fully qualified name within
    the `pymoo.constraints` module. For example to specify the
    `ConstraintsAsPenalty` object, the object should be specified
    here as 'as_penalty.ConstraintsAsPenalty`.

    For details about constraint handling, consult the `pymoo` manual:
    [`Operators`](https://pymoo.org/constraints/index.html)

    Attributes:
        name:       The fully qualified constraint class name.
        parameters: The parameters passed to the constraint constructor.
    """

    name: str
    parameters: dict[str, bool | int | float | str] = {}


class ParametersConfig(_ParametersBaseModel):
    """Configuration of the `pymoo` algorithm.

    Optimization by `pymoo` algorithms requires three objects:

    1. An algorithm object. Here they are specified by a fully qualified name
       within the `pymoo.algorithms` module. For example to specify the `GA`
       algorithm, it  should be specified here as 'soo.nonconvex.ga.GA`. The
       algorithm is initialized by parameters that here are specified by the
       `parameters` field.
    2. A constraints handling object.
    3. A termination object, or a tuple of termination values.
    4. A seed value for the random number generator used by the algorithms.

    Attributes:
        algorithm:   The fully qualified name of the algorithm class
        parameters:  The parameters passed to the algorithm constructor
        constraints: Specification of the constraint handling object to use
        termination: Specification of the termination object to use
        seed:        The seed value for the random number generator
    """

    algorithm: str | Algorithm
    parameters: dict[str, ParameterValues | ObjectConfig] = {}
    constraints: ConstraintsConfig | None = None
    termination: TerminationConfig | tuple[Any, ...] = TerminationConfig()
    seed: int = 1

    def get_algorithm(self) -> Algorithm:
        """Parse the algorithm config and its parameters.

        Raises:
            ValueError: When the algorithm object is not found.

        Returns:
            A `pymoo` algorithm object.
        """
        parameters = _parse_parameters(self.parameters)
        algorithm = _get_class(
            self.algorithm,
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
