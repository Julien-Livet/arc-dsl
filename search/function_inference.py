from search.function_types import FunctionType
from search.type_system import is_compatible, _is_union
from typing import get_args, Union, Callable, Container, Any
from arc_types import Boolean
from collections.abc import Callable as ABCCallable, Container as ABCContainer


def infer_compose(outer, inner):
    """
    Infère le type de compose(outer, inner).

    compose(outer, inner)(x) équivaut à outer(inner(x)).
    """
    if not isinstance(outer, FunctionType):
        raise TypeError("La fonction outer doit avoir une signature connue.")

    if not isinstance(inner, FunctionType):
        raise TypeError("La fonction inner doit avoir une signature connue.")

    if outer.arity != 1:
        raise TypeError("outer doit accepter exactement un argument.")

    if inner.arity != 1:
        raise TypeError("inner doit accepter exactement un argument.")

    inner_result = inner.return_type
    outer_argument = outer.parameter_types[0]

    if not is_compatible(inner_result, outer_argument) and not is_compatible(outer_argument, inner_result):
        raise TypeError(
            "Types incompatibles dans compose : "
            f"inner retourne {inner_result}, "
            f"mais outer attend {outer_argument}."
        )

    return FunctionType(
        parameter_types=inner.parameter_types,
        return_type=outer.return_type,
    )


def infer_fork(outer, a, b):
    """
    Infère le type de fork(outer, a, b).

    fork(outer, a, b)(x) équivaut à outer(a(x), b(x)).
    """
    functions = {
        "outer": outer,
        "a": a,
        "b": b,
    }

    for name, function in functions.items():
        if not isinstance(function, FunctionType):
            raise TypeError(
                f"La fonction {name} doit avoir une signature connue."
            )

    if outer.arity != 2:
        raise TypeError("outer doit accepter exactement deux arguments.")

    if a.arity != 1:
        raise TypeError("a doit accepter exactement un argument.")

    if b.arity != 1:
        raise TypeError("b doit accepter exactement un argument.")

    a_input = a.parameter_types[0]
    b_input = b.parameter_types[0]

    a_options = get_args(a_input) if _is_union(a_input) else (a_input,)
    b_options = get_args(b_input) if _is_union(b_input) else (b_input,)

    has_compatible_input = any(
        is_compatible(a_option, b_option)
        or is_compatible(b_option, a_option)
        for a_option in a_options
        for b_option in b_options
    )

    if not has_compatible_input:
        raise TypeError(
            "Les fonctions a et b ont des types d'entrée incompatibles."
        )

    if not is_compatible(a.return_type, outer.parameter_types[0]) and not is_compatible(outer.parameter_types[0], a.return_type):
        raise TypeError(
            "Le résultat de a est incompatible avec le premier argument "
            "de outer."
        )

    if outer.parameter_types[1] in (ABCCallable, Callable):
        second_argument_type = b
    else:
        second_argument_type = b.return_type

    if (
        not is_compatible(second_argument_type, outer.parameter_types[1])
        and not is_compatible(outer.parameter_types[1], second_argument_type)
    ):
        raise TypeError(
            "Le résultat de b est incompatible avec le second argument "
            "de outer."
        )

    common_input = a.parameter_types[0]

    if not is_compatible(b.parameter_types[0], common_input):
        if not is_compatible(common_input, b.parameter_types[0]):
            b_options = (
                get_args(b.parameter_types[0])
                if _is_union(b.parameter_types[0])
                else (b.parameter_types[0],)
            )
            a_options = (
                get_args(common_input)
                if _is_union(common_input)
                else (common_input,)
            )

            common_options = tuple(
                option
                for option in a_options
                if any(
                    is_compatible(option, candidate)
                    or is_compatible(candidate, option)
                    for candidate in b_options
                )
            )

            if not common_options:
                raise TypeError(
                    "Les fonctions a et b ont des types d'entrée incompatibles."
                )

            common_input = (
                common_options[0]
                if len(common_options) == 1
                else Union[common_options]
            )

    return FunctionType(
        parameter_types=(common_input,),
        return_type=outer.return_type,
    )


def infer_chain(h, g, f):
    """
    Infère le type de chain(h, g, f).

    chain(h, g, f)(x) équivaut à h(g(f(x))).
    """
    functions = {
        "h": h,
        "g": g,
        "f": f,
    }

    for name, function in functions.items():
        if not isinstance(function, FunctionType):
            raise TypeError(
                f"La fonction {name} doit avoir une signature connue."
            )

    for name, function in functions.items():
        if function.arity != 1:
            raise TypeError(
                f"{name} doit accepter exactement un argument."
            )

    if g.parameter_types[0] in (ABCCallable, Callable):
        if not isinstance(f, FunctionType):
            raise TypeError(
                "L'argument de g doit être une fonction connue."
            )
    elif (
        not is_compatible(f.return_type, g.parameter_types[0])
        and not is_compatible(g.parameter_types[0], f.return_type)
    ):
        raise TypeError(
            "Le résultat de f est incompatible avec l'argument de g."
        )

    if not is_compatible(g.return_type, h.parameter_types[0]) and not is_compatible(h.parameter_types[0], g.return_type):
        raise TypeError(
            "Le résultat de g est incompatible avec l'argument de h."
        )

    return FunctionType(
        parameter_types=f.parameter_types,
        return_type=h.return_type,
    )


def infer_lbind(function, fixed_type):
    """Infère la signature produite par lbind(function, fixed_type)."""
    if not isinstance(function, FunctionType):
        raise TypeError("function doit avoir une signature connue.")

    if function.variadic:
        raise TypeError("lbind ne prend pas en charge une arité inconnue.")

    if function.arity == 2:
        expected_fixed_arity = 1
    elif function.arity == 3:
        expected_fixed_arity = 2
    else:
        expected_fixed_arity = 3

    if function.arity != expected_fixed_arity + 1:
        raise TypeError(
            "lbind ne peut pas inférer cette arité conformément au DSL."
        )

    expected_type = function.parameter_types[0]

    if (
        not is_compatible(fixed_type, expected_type)
        and not is_compatible(expected_type, fixed_type)
    ):
        raise TypeError(
            "Le type de l'argument fixé est incompatible avec "
            "le premier paramètre de function."
        )

    return_type = function.return_type

    return FunctionType(
        parameter_types=function.parameter_types[1:],
        return_type=return_type,
    )


def infer_rbind(function, fixed_type):
    """Infère la signature produite par rbind(function, fixed_type)."""
    if not isinstance(function, FunctionType):
        raise TypeError("function doit avoir une signature connue.")

    if function.variadic:
        raise TypeError("rbind ne prend pas en charge une arité inconnue.")

    if function.arity == 2:
        expected_fixed_arity = 1
    elif function.arity == 3:
        expected_fixed_arity = 2
    else:
        expected_fixed_arity = 3

    if function.arity != expected_fixed_arity + 1:
        raise TypeError(
            "rbind ne peut pas inférer cette arité conformément au DSL."
        )

    expected_type = function.parameter_types[-1]

    if (
        not is_compatible(fixed_type, expected_type)
        and not is_compatible(expected_type, fixed_type)
    ):
        raise TypeError(
            "Le type de l'argument fixé est incompatible avec "
            "le dernier paramètre de function."
        )

    return_type = function.return_type

    return FunctionType(
        parameter_types=function.parameter_types[:-1],
        return_type=return_type,
    )


def infer_matcher(function, target_type):
    """
    Infère le type de matcher(function, target).

    Le résultat est une fonction unaire qui retourne un booléen.
    """
    if not isinstance(function, FunctionType):
        raise TypeError("function doit avoir une signature connue.")

    if function.arity != 1:
        raise TypeError(
            "matcher attend une fonction qui accepte exactement un argument."
        )

    if not is_compatible(function.return_type, target_type):
        if not is_compatible(target_type, function.return_type):
            raise TypeError(
                "Le type de target est incompatible avec le résultat "
                "de function."
            )

    return FunctionType(
        parameter_types=function.parameter_types,
        return_type=Boolean,
    )


def infer_power(function, n):
    """Infère la signature de power(function, n)."""
    if not isinstance(function, FunctionType):
        raise TypeError("function doit avoir une signature connue.")

    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("n doit être un entier.")

    if n < 1:
        raise ValueError("power exige n >= 1.")

    if function.arity != 1:
        raise TypeError(
            "power attend une fonction qui accepte exactement un argument."
        )

    if n == 1:
        return function

    input_type = function.parameter_types[0]
    output_type = function.return_type

    if not is_compatible(output_type, input_type):
        if not is_compatible(input_type, output_type):
            raise TypeError(
                "La fonction n'est pas composable avec elle-même : "
                "son type de retour est incompatible avec son type d'entrée."
            )

    return function


def infer_mapply(function, container_type):
    """Infère le type de retour de mapply(function, container)."""
    from typing import FrozenSet, get_args, get_origin

    if not isinstance(function, FunctionType):
        if function is Callable:
            return FrozenSet[Any]

        raise TypeError("function doit avoir une signature connue.")

    if function.arity != 1:
        raise TypeError("mapply attend une fonction unaire.")

    result_type = function.return_type
 
    result_types = (
        get_args(result_type)
        if _is_union(result_type)
        else (result_type,)
    )

    container_types = []

    for candidate in result_types:
        origin = get_origin(candidate)
        args = get_args(candidate)

        if origin is frozenset and args:
            container_types.append(FrozenSet[args[0]])

        elif origin is tuple and args:
            if len(args) == 2 and args[1] is Ellipsis:
                container_types.append(FrozenSet[args[0]])
            else:
                raise TypeError(
                    "Tuple hétérogène non pris en charge par mapply : "
                    f"{candidate!r}"
                )

        elif candidate in (Container, ABCContainer):
            container_types.append(FrozenSet[Any])

        elif candidate is Any:
            container_types.append(FrozenSet[Any])

        else:
            raise TypeError(
                "Impossible d'inférer le type des éléments produits par "
                f"mapply : branche de retour = {candidate!r}"
            )

    if not container_types:
        raise TypeError(
            f"Impossible d'inférer le type de retour de mapply : {result_type!r}"
        )

    return (
        container_types[0]
        if len(container_types) == 1
        else Union[tuple(container_types)]
    )


def infer_mfilter(container_type, function):
    """Infère le type de retour de mfilter(container, function)."""
    if not isinstance(function, FunctionType):
        raise TypeError("function doit avoir une signature connue.")

    if function.arity != 1:
        raise TypeError("mfilter attend une fonction unaire.")

    container_args = get_args(container_type)
    if not container_args:
        return container_type

    element_type = container_args[0]

    if not is_compatible(element_type, function.parameter_types[0]):
        raise TypeError(
            f"Le type des éléments {element_type} est incompatible "
            f"avec le paramètre {function.parameter_types[0]}."
        )

    return element_type


def infer_branch(condition, function1, function2):
    """Infère le type de retour du combinateur branch."""

    # Les branches sont des valeurs : branch(condition, valeur1, valeur2).
    if not isinstance(function1, FunctionType) and not isinstance(
        function2, FunctionType
    ):
        if is_compatible(function1, function2):
            return function1
        elif is_compatible(function2, function1):
            return function2
        raise TypeError(
            "Les deux branches de branch doivent avoir des types compatibles."
        )

    # Les branches sont des fonctions.
    if not isinstance(function1, FunctionType) or not isinstance(function2, FunctionType):
        raise TypeError(
            "Les deux branches doivent être soit des valeurs, soit des fonctions."
        )

    if function1.arity != function2.arity:
        raise TypeError("Les fonctions de branch doivent avoir la même arité.")

    compatible: bool = True

    for p1, p2 in zip(function1.parameter_types, function2.parameter_types):
        compatible = compatible and (is_compatible(p1, p2) or is_compatible(p2, p1))

    if not compatible:
        raise TypeError(
            "Les fonctions de branch doivent avoir les mêmes paramètres."
        )

    return_type1 = function1.return_type
    return_type2 = function2.return_type

    if is_compatible(return_type1, return_type2):
        common_return_type = return_type2
    elif is_compatible(return_type2, return_type1):
        common_return_type = return_type1
    else:
        common_return_type = Union[return_type1, return_type2]

    return FunctionType(
        parameter_types=function1.parameter_types,
        return_type=common_return_type,
    )
