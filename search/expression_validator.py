from typing import Callable, FrozenSet, Tuple, get_args, get_origin, Union, Any, Container
from arc_types import Grid
from search.expressions import (
    InputExpr,
    ConstantExpr,
    PrimitiveExpr,
    PrimitiveRefExpr,
    ApplyExpr,
)
from search.function_types import FunctionType
from search.function_inference import (
    infer_compose,
    infer_fork,
    infer_chain,
    infer_lbind,
    infer_rbind,
    infer_matcher,
    infer_power,
    infer_mapply,
    infer_mfilter,
    infer_branch,
)
from search.type_system import is_compatible, get_parameters, get_return_type, _is_union


FUNCTION_COMBINATORS = {
    "compose",
    "fork",
    "chain",
    "lbind",
    "rbind",
    "matcher",
    "power",
    "mapply",
    "mfilter",
    "branch",
}

RETURN_TYPE_SPECIALIZERS = {}


def _infer_partial_binding(
    binding_name,
    function_expression,
    fixed_expression,
    registry,
):
    """Infère le type d'une fonction produite par lbind ou rbind."""
    if not isinstance(function_expression, PrimitiveRefExpr):
        return None

    if not isinstance(fixed_expression, (PrimitiveRefExpr, PrimitiveExpr)):
        return None

    target_name = function_expression.name

    fixed_type = infer_expression_type(fixed_expression, registry)
    if not isinstance(fixed_type, FunctionType):
        return None

    if target_name == "compose" and binding_name == "lbind":
        return FunctionType(
            parameter_types=(Any,),
            return_type=fixed_type.return_type,
        )

    if target_name not in {"lbind", "rbind"}:
        return None

    # On récupère le type réel de la primitive fixée.
    fixed_type = infer_expression_type(fixed_expression, registry)
    if not isinstance(fixed_type, FunctionType):
        return None

    if target_name == "compose" and binding_name == "lbind":
        return FunctionType(
            parameter_types=(Any,),
            return_type=FunctionType(
                parameter_types=(Any,),
                return_type=fixed_type.return_type,
            ),
        )

    # Pour lbind(rbind, f), la fonction résultante reçoit un argument
    # puis retourne la signature obtenue en liant f à rbind.
    if target_name == "rbind" and binding_name == "lbind":
        return FunctionType(
            parameter_types=(Any,),
            return_type=FunctionType(
                parameter_types=fixed_type.parameter_types[:-1],
                return_type=fixed_type.return_type,
                variadic=fixed_type.variadic,
            ),
        )

    # Pour rbind(lbind, f), la fonction résultante reçoit un argument
    # puis retourne la signature obtenue en liant f à lbind.
    if target_name == "lbind" and binding_name == "rbind":
        return FunctionType(
            parameter_types=(Any,),
            return_type=FunctionType(
                parameter_types=fixed_type.parameter_types[1:],
                return_type=fixed_type.return_type,
                variadic=fixed_type.variadic,
            ),
        )

    return None


def specialize_size(argument_types, function_expr, fixed_type=None):
    """Infère le type de retour de size."""
    if len(argument_types) != 1:
        return None

    return int


def specialize_arithmetic(argument_types, function_expr, fixed_type=None):
    """Infère le retour arithmétique à partir des arguments connus."""
    if len(argument_types) != 1:
        return None

    argument_type = argument_types[0]

    argument_options = (
        get_args(argument_type)
        if _is_union(argument_type)
        else (argument_type,)
    )

    fixed_options = (
        get_args(fixed_type)
        if fixed_type is not None and _is_union(fixed_type)
        else (fixed_type,)
    )

    # Une opération sur deux coordonnées renvoie un tuple de coordonnées.
    if (
        all(is_compatible(option, Tuple[int, int]) for option in argument_options)
        or (
            fixed_type is not None
            and all(is_compatible(option, Tuple[int, int]) for option in fixed_options)
        )
    ):
        return Tuple[int, int]

    # Une opération sur deux entiers renvoie un entier.
    if (
        all(is_compatible(option, int) for option in argument_options)
        and fixed_type is not None
        and all(is_compatible(option, int) for option in fixed_options)
    ):
        return int

    # Conserver le comportement pour un argument entier si le type fixé est inconnu.
    if fixed_type is None and all(
        is_compatible(option, int) for option in argument_options
    ):
        return int

    return None


for primitive_name in ("add", "subtract", "multiply", "divide"):
    RETURN_TYPE_SPECIALIZERS[primitive_name] = specialize_arithmetic


RETURN_TYPE_SPECIALIZERS["size"] = specialize_size


def get_fixed_argument_type(function_expr, registry):
    """Retourne le type de l'argument fixé par lbind ou rbind."""
    if not isinstance(function_expr, PrimitiveExpr):
        return None

    if function_expr.name not in {"lbind", "rbind"}:
        return None

    if len(function_expr.arguments) != 2:
        return None

    return infer_expression_type(function_expr.arguments[1], registry)


def infer_specialized_return(
    function_expr,
    function_type,
    argument_types,
    registry,
):
    """Infère le type de retour avec les types des arguments connus."""
    if not isinstance(function_type, FunctionType):
        return Any

    primitive_name = None

    if isinstance(function_expr, PrimitiveRefExpr):
        primitive_name = function_expr.name

    elif (
        isinstance(function_expr, PrimitiveExpr)
        and function_expr.name in {"lbind", "rbind"}
        and function_expr.arguments
        and isinstance(function_expr.arguments[0], PrimitiveRefExpr)
    ):
        primitive_name = function_expr.arguments[0].name

    if primitive_name is not None:
        specializer = RETURN_TYPE_SPECIALIZERS.get(primitive_name)

        if specializer is not None:
            fixed_type = get_fixed_argument_type(function_expr, registry)
            specialized_type = specializer(
                argument_types,
                function_expr,
                fixed_type,
            )

            if specialized_type is not None:
                return specialized_type

    return function_type.return_type


def infer_expression_type(expression, registry):
    """Infère le type d'une expression du DSL."""

    if isinstance(expression, InputExpr):
        if expression.name != "I":
            raise ValueError(f"Entrée inconnue : {expression.name}")
        return Grid

    if isinstance(expression, ConstantExpr):
        constants = registry["constants"]

        if expression.name not in constants:
            raise ValueError(f"Constante inconnue : {expression.name}")

        registered = constants[expression.name]

        if expression.value != registered["value"]:
            raise ValueError(
                f"Valeur incorrecte pour la constante {expression.name}"
            )

        return registered["type"]

    if isinstance(expression, PrimitiveRefExpr):
        primitives = registry["primitives"]

        if expression.name not in primitives:
            raise ValueError(f"Primitive inconnue : {expression.name}")

        primitive_info = primitives[expression.name]
        parameters = get_parameters(primitive_info)

        if any(parameter["type"] is None for parameter in parameters):
            raise TypeError(
                f"Signature incomplète pour la primitive {expression.name}."
            )

        return_type = get_return_type(primitive_info)

        if return_type is None:
            raise TypeError(
                f"Type de retour inconnu pour la primitive {expression.name}."
            )

        return FunctionType(
            parameter_types=tuple(
                parameter["type"] for parameter in parameters
            ),
            return_type=return_type,
        )

    if isinstance(expression, PrimitiveExpr):
        primitives = registry["primitives"]

        if expression.name not in primitives:
            raise ValueError(f"Primitive inconnue : {expression.name}")

        primitive_info = primitives[expression.name]
        parameters = get_parameters(primitive_info)

        if len(expression.arguments) != len(parameters):
            raise ValueError(
                f"{expression.name} attend {len(parameters)} arguments, "
                f"mais en reçoit {len(expression.arguments)}"
            )

        actual_types = [
            infer_expression_type(argument, registry)
            for argument in expression.arguments
        ]

        if expression.name in {"first", "last"}:
            (container_type,) = actual_types

            if _is_union(container_type):
                element_types = []

                for option in get_args(container_type):
                    option_args = get_args(option)

                    if option_args:
                        element_type = (
                            option_args[-1]
                            if expression.name == "last"
                            else option_args[0]
                        )
                        if element_type not in element_types:
                            element_types.append(element_type)

                if element_types:
                    return (
                        element_types[0]
                        if len(element_types) == 1
                        else Union[tuple(element_types)]
                    )

            container_args = get_args(container_type)

            if container_args:
                if len(container_args) == 2 and container_args[1] is Ellipsis:
                    return container_args[0]

                if expression.name == "first":
                    return container_args[0]

                return container_args[-1]

        if expression.name in {"canvas", }:
            return Grid

        if expression.name in {"add", "subtract", "multiply", "divide"}:
            a, b = actual_types

            if (is_compatible(a, Tuple[int, int]) or is_compatible(b, Tuple[int, int])):
                return Tuple[int, int]

            if (is_compatible(a, int) or is_compatible(b, int)):
                return int

        if expression.name in {"hmirror", "vmirror", "dmirror", "cmirror", "upscale", "hupscale",
                               "vupscale", "downscale", "paint", "normalize", "shift",
                               "rot90", "rot180", "rot270", "shoot", "fill", "decrement",
                               "increment", "crement", "underfill", "underpaint"}:
            actual_type = actual_types[0]
            expected_type = parameters[0]["type"]

            if is_compatible(actual_type, expected_type) or is_compatible(expected_type, actual_type):
                return actual_type

        if expression.name == "sfilter":
            container_type, condition_type = actual_types

            if not isinstance(condition_type, FunctionType):
                raise TypeError(
                    "sfilter attend une fonction de condition connue."
                )

            return container_type

        if expression.name in FUNCTION_COMBINATORS:
            result = _infer_function_combinator(
                registry,
                expression.name,
                expression.arguments,
                actual_types,
            )
            return result

        if expression.name == "apply":
            function_expr, container_expr = expression.arguments
            function_type, container_type = actual_types

            if (
                isinstance(function_expr, PrimitiveExpr)
                and function_expr.name in {"lbind", "rbind"}
                and isinstance(function_expr.arguments[0], PrimitiveRefExpr)
                and function_expr.arguments[0].name == "order"
            ):
                return container_type

            if not isinstance(function_type, FunctionType):
                return Any

            container_origin = get_origin(container_type)
            container_args = get_args(container_type)

            if container_type is tuple or container_type is Tuple:
                return Tuple[Any, ...]

            if container_origin is not None and container_args:
                if container_origin is tuple:
                    element_type = (
                        container_args[0]
                        if len(container_args) == 2 and container_args[1] is Ellipsis
                        else container_args[0]
                        if container_args and all(
                            is_compatible(arg_type, container_args[0])
                            and is_compatible(container_args[0], arg_type)
                            for arg_type in container_args
                        )
                        else None
                    )

                    if element_type is not None:
                        #print("APPLY function expression :", repr(function_expr))
                        #print("APPLY function type :", repr(function_type))
                        #print("APPLY container type :", repr(container_type))
                        #print("APPLY element type :", repr(element_type))

                        return_type = infer_specialized_return(
                            function_expr,
                            function_type,
                            (element_type,),
                            registry,
                        )

                        #print("APPLY specialized return :", repr(return_type))
                        return Tuple[return_type, ...]

                if container_origin is frozenset:
                    element_type = container_args[0]

                    return_type = infer_specialized_return(
                        function_expr,
                        function_type,
                        (element_type,),
                            registry,
                    )
                    return FrozenSet[return_type]

        if expression.name == "order":
            container_type = actual_types[0]
            container_args = get_args(container_type)
            container_origin = get_origin(container_type)

            if container_args:
                if container_origin is frozenset:
                    return Tuple[container_args[0], ...]
                if container_origin is tuple:
                    return container_type

        if expression.name == "merge":
            container_args = get_args(actual_types[0])

            if container_args:
                return container_args[0]

        if expression.name == "astuple":
            a_type, b_type = actual_types
            return Tuple[a_type, b_type]

        if expression.name == "pair":
            a_type, b_type = actual_types
            return Tuple[a_type, b_type]

        if expression.name == "prapply":
            function_type, a_type, b_type = actual_types

            if not isinstance(function_type, FunctionType):
                raise TypeError("Le premier argument de prapply doit être une fonction.")

            return_type = function_type.return_type

            if return_type is Any:
                raise TypeError(
                    "Impossible d'inférer le type de retour de la fonction passée à prapply."
                )

            return FrozenSet[return_type]

        if expression.name == "remove":
            value_type, container_type = actual_types
            return container_type

        if expression.name == "insert":
            value_type, container_type = actual_types

            origin = get_origin(container_type)

            if origin is frozenset:
                # Préserver le type des éléments déjà présents dans le conteneur.
                return container_type

            if container_type in (FrozenSet, frozenset):
                return FrozenSet[value_type]

            raise TypeError(
                f"Impossible d'inférer le type de insert pour {container_type!r}"
            )

        if expression.name in {"argmin", "argmax", "valmin", "valmax"}:
            container_type, key_function = actual_types

            if not isinstance(key_function, FunctionType):
                raise TypeError(
                    f"{expression.name} attend une fonction de comparaison connue."
                )

            container_types = (
                get_args(container_type)
                if _is_union(container_type)
                else (container_type,)
            )

            element_types = []
            for candidate in container_types:
                candidate_args = get_args(candidate)
                if candidate_args:
                    element_type = candidate_args[0]
                    if element_type not in element_types:
                        element_types.append(element_type)

            if not element_types:
                raise TypeError(
                    f"Impossible de déterminer le type des éléments de "
                    f"{expression.name}."
                )

            if expression.name in {"argmin", "argmax"}:
                return (
                    element_types[0]
                    if len(element_types) == 1
                    else Union[tuple(element_types)]
                )

            return key_function.return_type

        for actual_type, parameter in zip(actual_types, parameters):
            expected_type = parameter["type"]

            if expected_type is None:
                continue

            if expected_type is Callable:
                if not isinstance(actual_type, FunctionType):
                    raise TypeError(
                        f"Argument {parameter['name']} de {expression.name} "
                        "doit être une fonction de signature connue."
                    )
                continue
            #print("VALIDATOR expected :", repr(expected_type))
            #print("VALIDATOR actual :", repr(actual_type))
            #print("VALIDATOR actual -> expected :", is_compatible(
            #    actual_type, expected_type
            #))
            #print("VALIDATOR expected -> actual :", is_compatible(
            #    expected_type, actual_type
            #))
            if not is_compatible(actual_type, expected_type) and not is_compatible(expected_type, actual_type):
                raise TypeError(
                    f"Argument {parameter['name']} de {expression.name} : "
                    f"type attendu {expected_type}, reçu {actual_type}"
                )

        if expression.name == "difference" or expression.name == "intersection":
            return actual_types[0]

        if expression.name in ("double", "halve") and len(actual_types) == 1:
            argument_type = actual_types[0]

            if argument_type is int:
                return int

            if argument_type == Tuple[int, int]:
                return Tuple[int, int]

        return get_return_type(primitive_info)

    if isinstance(expression, ApplyExpr):
        function_type = infer_expression_type(expression.function, registry)

        argument_types = tuple(
            infer_expression_type(argument, registry)
            for argument in expression.arguments
        )

        if not isinstance(function_type, FunctionType):
            raise TypeError(
                f"L'expression appelée n'a pas de signature de fonction : "
                f"{function_type!r}"
            )

        if len(argument_types) > function_type.arity:
            raise TypeError(
                f"La fonction attend au maximum {function_type.arity} argument(s), "
                f"mais {len(argument_types)} reçu(s)."
            )

        for index, (actual_type, expected_type) in enumerate(
            zip(argument_types, function_type.parameter_types)
        ):
            if (
                not is_compatible(actual_type, expected_type)
                and not is_compatible(expected_type, actual_type)
            ):
                raise TypeError(
                    f"Type incompatible pour l'argument {index + 1} : "
                    f"attendu {expected_type!r}, reçu {actual_type!r}."
                )

        remaining_parameter_types = function_type.parameter_types[len(argument_types):]

        if remaining_parameter_types:
            return FunctionType(
                parameter_types=remaining_parameter_types,
                return_type=function_type.return_type,
            )

        return function_type.return_type

    raise TypeError(
        f"Type d'expression non pris en charge : {type(expression).__name__}"
    )


def _infer_function_combinator(registry, name, arguments, actual_types):
    """Infère la signature produite par un combinateur de fonctions."""
    if name == "compose":
        outer, inner = actual_types
        result = infer_compose(outer, inner)
        outer_expression = arguments[0]

        if isinstance(outer_expression, PrimitiveExpr):
            if outer_expression.name == "rbind":
                bound_expression = outer_expression.arguments[0]

                if (
                    isinstance(bound_expression, PrimitiveRefExpr)
                    and bound_expression.name in {"argmax", "argmin"}
                ):
                    container_type = inner.return_type
                    container_args = get_args(container_type)

                    if container_args:
                        element_type = container_args[0]
                        return FunctionType(
                            parameter_types=result.parameter_types,
                            return_type=element_type,
                            variadic=result.variadic,
                        )

        return result

    if name == "fork":
        outer, a, b = actual_types
        outer_expression = arguments[0]

        if (
            isinstance(outer_expression, PrimitiveRefExpr)
            and outer_expression.name in {"argmin", "argmax"}
        ):
            outer = FunctionType(
                parameter_types=outer.parameter_types,
                return_type=a.return_type,
                variadic=outer.variadic,
            )

        result = infer_fork(outer, a, b)

        if (
            isinstance(outer_expression, PrimitiveRefExpr)
            and outer_expression.name == "sfilter"
        ):
            return FunctionType(
                parameter_types=result.parameter_types,
                return_type=Container,
                variadic=result.variadic,
            )

        if isinstance(outer_expression, PrimitiveRefExpr):
            primitive_name = outer_expression.name

            if primitive_name in {"difference", "intersection", "combine"}:
                return FunctionType(
                    parameter_types=result.parameter_types,
                    return_type=a.return_type,
                    variadic=result.variadic,
                )

            if primitive_name == "apply":
                function_expression = arguments[1]

                if isinstance(function_expression, PrimitiveExpr):
                    if function_expression.name == "lbind":
                        bound_expression = function_expression.arguments[0]

                        if (
                            isinstance(bound_expression, PrimitiveRefExpr)
                            and bound_expression.name == "lbind"
                        ):
                            inner_function = function_expression.arguments[1]

                            if isinstance(inner_function, PrimitiveRefExpr):
                                if inner_function.name == "shift":
                                    return FunctionType(
                                        parameter_types=result.parameter_types,
                                        return_type=b.return_type,
                                        variadic=result.variadic,
                                    )

        return result

    if name == "chain":
        h, g, f = actual_types
        return infer_chain(h, g, f)

    if name == "lbind":
        #print("LBIND expression :", repr(arguments[0]))
        #print("LBIND fixed expression :", repr(arguments[1]))
        #print("LBIND inferred function :", repr(actual_types[0]))
        #print("LBIND inferred fixed type :", repr(actual_types[1]))
        specialized = _infer_partial_binding(
            "lbind",
            arguments[0],
            arguments[1],
            registry,
        )
        if specialized is not None:
            return specialized

        function_type, _ = actual_types
        fixed_type = actual_types[1]
        result = infer_lbind(function_type, fixed_type)

        function_expression = arguments[0]

        if (
            isinstance(function_expression, PrimitiveRefExpr)
            and function_expression.name == "apply"
        ):
            container_origin = get_origin(fixed_type)
            container_args = get_args(fixed_type)

            if container_origin is tuple and container_args:
                return FunctionType(
                    parameter_types=result.parameter_types,
                    return_type=Tuple[Any, ...],
                    variadic=result.variadic,
                )

            if container_origin is frozenset and container_args:
                return FunctionType(
                    parameter_types=result.parameter_types,
                    return_type=FrozenSet[Any],
                    variadic=result.variadic,
                )

        if isinstance(function_expression, PrimitiveRefExpr):
            primitive_name = function_expression.name

            if primitive_name in {"argmax", "argmin"}:
                container_type = function_type.parameter_types[0]
                container_args = get_args(container_type)

                if container_args:
                    return FunctionType(
                        parameter_types=result.parameter_types,
                        return_type=container_args[0],
                        variadic=result.variadic,
                    )

        return result

    if name == "rbind":
        specialized = _infer_partial_binding(
            "rbind",
            arguments[0],
            arguments[1],
            registry,
        )
        if specialized is not None:
            return specialized

        function_type, _ = actual_types
        fixed_type = actual_types[1]
        result = infer_rbind(function_type, fixed_type)

        function_expression = arguments[0]

        if (
            isinstance(function_expression, PrimitiveRefExpr)
            and function_expression.name == "apply"
        ):
            container_origin = get_origin(fixed_type)
            container_args = get_args(fixed_type)

            if container_origin is tuple and container_args:
                return FunctionType(
                    parameter_types=result.parameter_types,
                    return_type=Tuple[Any, ...],
                    variadic=result.variadic,
                )

            if container_origin is frozenset and container_args:
                return FunctionType(
                    parameter_types=result.parameter_types,
                    return_type=FrozenSet[Any],
                    variadic=result.variadic,
                )

        function_expression = arguments[0]

        if isinstance(function_expression, PrimitiveRefExpr):
            primitive_name = function_expression.name

            if primitive_name in {"argmax", "argmin"}:
                container_type = function_type.parameter_types[0]
                container_args = get_args(container_type)

                if container_args:
                    return FunctionType(
                        parameter_types=result.parameter_types,
                        return_type=container_args[0],
                        variadic=result.variadic,
                    )

        return result

    if name == "matcher":
        function_type = actual_types[0]
        target_type = actual_types[1]
        return infer_matcher(function_type, target_type)

    if name == "power":
        function_type = actual_types[0]
        n = arguments[1].value if isinstance(arguments[1], ConstantExpr) else None

        if n is None:
            raise TypeError(
                "L'exposant de power doit être une constante connue."
            )

        return infer_power(function_type, n)

    if name == "mapply":
        function, container = actual_types
        function_expression = arguments[0]

        if (
            isinstance(function_expression, PrimitiveExpr)
            and function_expression.name == "fork"
            and isinstance(function_expression.arguments[0], PrimitiveRefExpr)
            and function_expression.arguments[0].name == "sfilter"
        ):
            container_args = get_args(container)

            if len(container_args) == 2:
                element_container_type = container_args[1]
                element_container_origin = get_origin(element_container_type)
                element_container_args = get_args(element_container_type)

                if (
                    element_container_origin is tuple
                    and len(element_container_args) == 2
                    and element_container_args[1] is Ellipsis
                ):
                    return FrozenSet[element_container_args[0]]

        if isinstance(function_expression, PrimitiveRefExpr):
            if function_expression.name in {"first", "last"}:
                container_args = get_args(container)

                if container_args:
                    element_type = container_args[0]
                    element_args = get_args(element_type)

                    if element_args:
                        result_type = (
                            element_args[-1]
                            if function_expression.name == "last"
                            else element_args[0]
                        )

                        function = FunctionType(
                            parameter_types=function.parameter_types,
                            return_type=result_type,
                            variadic=function.variadic,
                        )

        return infer_mapply(function, container)

    if name == "mfilter":
        container, function = actual_types
        return infer_mfilter(container, function)

    if name == "branch":
        return infer_branch(*actual_types)

    raise ValueError(f"Combinateur inconnu : {name}")


def validate_expression(expression, registry, expected_type=Grid):
    """Vérifie qu'une expression produit un type compatible avec la cible."""
    actual_type = infer_expression_type(expression, registry)

    if isinstance(actual_type, FunctionType):
        raise TypeError(
            "L'expression produit une fonction, mais un résultat "
            f"de type {expected_type} est attendu."
        )

    if not is_compatible(actual_type, expected_type):
        raise TypeError(
            f"Type de résultat attendu : {expected_type}, reçu : {actual_type}"
        )

    return True
