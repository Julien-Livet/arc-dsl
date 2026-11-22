import inspect
from typing import get_type_hints

import dsl
import constants
from search.generics import GenericSignature, erase, is_generic


def load_generic_signature(function, type_hints):
    """
    Retourne la signature générique d'une primitive, ou None.

    Seuls les paramètres positionnels sont retenus, comme dans
    get_parameters.
    """
    if not any(is_generic(type_hint) for type_hint in type_hints.values()):
        return None

    parameters = [
        parameter
        for parameter in inspect.signature(function).parameters.values()
        if parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
    ]

    return GenericSignature(
        parameter_types=tuple(type_hints.get(parameter.name) for parameter in parameters),
        return_type=type_hints.get("return"),
    )


def load_primitives():
    """
    Recense les primitives définies directement dans dsl.py.

    type_hints contient les annotations effacées (sans variable de type),
    identiques aux annotations non génériques ; generic_signature contient
    la signature générique, ou None.
    """
    primitives = {}

    for name, function in inspect.getmembers(dsl, inspect.isfunction):
        # Ignore les fonctions importées depuis d'autres modules.
        if function.__module__ != dsl.__name__:
            continue

        type_hints = get_type_hints(function)

        primitives[name] = {
            "function": function,
            "signature": inspect.signature(function),
            "type_hints": {
                key: erase(type_hint)
                for key, type_hint in type_hints.items()
            },
            "generic_signature": load_generic_signature(function, type_hints),
        }

    return primitives


def load_constants():
    """Recense les constantes annotées définies dans constants.py."""
    constant_types = get_type_hints(constants)
    result = {}

    for name, value in vars(constants).items():
        if name.startswith("_"):
            continue

        # Ne retient que les variables explicitement annotées.
        if name not in constant_types:
            continue

        result[name] = {
            "value": value,
            "type": constant_types[name],
        }

    return result


def load_registry():
    """Construit le registre complet du DSL."""
    return {
        "primitives": load_primitives(),
        "constants": load_constants(),
    }
