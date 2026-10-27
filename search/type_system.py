import types
from typing import Any, Callable, FrozenSet, Tuple, Union, get_args, get_origin
from collections.abc import Container as ABCContainer, Callable as ABCCallable

def get_return_type(primitive_info):
    """Retourne le type de retour annoté d'une primitive."""
    return primitive_info["type_hints"].get("return")


def _is_union(type_hint):
    """Reconnaît typing.Union et les unions de la forme X | Y."""
    return (
        get_origin(type_hint) is Union
        or get_origin(type_hint) is types.UnionType
    )


def is_compatible(actual_type, expected_type):
    """
    Vérifie si actual_type peut satisfaire expected_type.

    Gère Any, les unions, les types simples et les conteneurs génériques.
    """
    if actual_type is Any or expected_type is Any:
        return True

    # Une annotation FrozenSet sans paramètres accepte tout FrozenSet typé.
    if expected_type in (FrozenSet, frozenset):
        return get_origin(actual_type) is frozenset

    # Évite de confondre bool et int, malgré l'héritage Python.
    if actual_type is bool and expected_type is int:
        return False

    if actual_type == expected_type:
        return True

    # Une fonction DSL connue satisfait une annotation Callable non précisée.
    from search.function_types import FunctionType

    if isinstance(actual_type, FunctionType):
        if expected_type in (ABCCallable, Callable):
            return True

    # Chaque branche du type réel doit être acceptée par au moins
    # une branche du type attendu.
    if _is_union(actual_type):
        return all(
            any(
                is_compatible(actual_option, expected_option)
                for expected_option in (
                    get_args(expected_type)
                    if _is_union(expected_type)
                    else (expected_type,)
                )
            )
            for actual_option in get_args(actual_type)
        )

    # Un type réel simple doit correspondre à au moins une branche attendue.
    if _is_union(expected_type):
        expected_options = get_args(expected_type)

        # Une correspondance exacte avec une branche suffit.
        if actual_type in expected_options:
            return True

        actual_origin = get_origin(actual_type)
        actual_args = get_args(actual_type)

        # Un tuple variadique d'éléments T peut satisfaire une branche
        # attendue si T est compatible avec cette branche.
        if (
            actual_origin is tuple
            and len(actual_args) == 2
            and actual_args[1] is Ellipsis
        ):
            element_type = actual_args[0]
            return any(
                is_compatible(element_type, option)
                for option in expected_options
            )

        return any(
            is_compatible(actual_type, option)
            for option in expected_options
        )

    # Un FrozenSet sans paramètres peut représenter un FrozenSet typé.
    if actual_type in (FrozenSet, frozenset):
        expected_origin = get_origin(expected_type)
        if expected_origin is frozenset:
            return True
        expected_args = get_args(expected_type)
        if expected_args and all(
            get_origin(member) is frozenset
            for member in expected_args
        ):
            return True

    # Une annotation Tuple sans paramètres accepte tout tuple typé.
    if expected_type in (Tuple, tuple):
        return get_origin(actual_type) is tuple or actual_type is tuple

    # Un conteneur concret peut satisfaire une interface Container.
    expected_origin = get_origin(expected_type)

    if expected_origin is ABCContainer:
        actual_origin = get_origin(actual_type) or actual_type

        try:
            if not issubclass(actual_origin, ABCContainer):
                return False
        except TypeError:
            return False

        expected_args = get_args(expected_type)
        if not expected_args:
            return True

        actual_args = get_args(actual_type)
        if not actual_args:
            return True

        return all(
            is_compatible(actual_arg, expected_arg)
            for actual_arg, expected_arg in zip(actual_args, expected_args)
        )

    actual_origin = get_origin(actual_type)
    expected_origin = get_origin(expected_type)

    # Pour les conteneurs génériques, comparer l'origine et les arguments.
    if actual_origin is not None or expected_origin is not None:
        if actual_origin != expected_origin:
            return False

        actual_args = get_args(actual_type)
        expected_args = get_args(expected_type)

        # Un tuple variadique, Tuple[T, ...], accepte des tuples
        # dont les éléments sont tous compatibles avec T.
        if expected_origin is tuple and len(expected_args) == 2 and expected_args[1] is Ellipsis:
            if actual_origin is not tuple:
                return False

            if len(actual_args) == 2 and actual_args[1] is Ellipsis:
                return is_compatible(actual_args[0], expected_args[0])

            return all(
                is_compatible(arg, expected_args[0])
                for arg in actual_args
            )

        # Un tuple variadique réel peut satisfaire un tuple attendu
        # si son type d'élément convient à chaque position attendue.
        if actual_origin is tuple and len(actual_args) == 2 and actual_args[1] is Ellipsis:
            if expected_origin is not tuple:
                return False

            return all(
                is_compatible(actual_args[0], expected_arg)
                for expected_arg in expected_args
                if expected_arg is not Ellipsis
            )

        if len(actual_args) != len(expected_args):
            return False

        return all(
            is_compatible(actual_arg, expected_arg)
            and is_compatible(expected_arg, actual_arg)
            for actual_arg, expected_arg in zip(actual_args, expected_args)
        )

    try:
        return issubclass(actual_type, expected_type)
    except TypeError:
        return actual_type == expected_type


def get_parameters(primitive_info):
    """Retourne les paramètres d'une primitive avec leurs types."""
    signature = primitive_info["signature"]
    type_hints = primitive_info["type_hints"]

    parameters = []

    for name, parameter in signature.parameters.items():
        if parameter.kind in (
            parameter.VAR_POSITIONAL,
            parameter.VAR_KEYWORD,
        ):
            continue

        parameters.append({
            "name": name,
            "type": type_hints.get(name),
            "required": parameter.default is parameter.empty,
        })

    return parameters
