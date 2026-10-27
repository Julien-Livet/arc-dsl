import inspect
from typing import get_type_hints

import dsl
import constants


def load_primitives():
    """Recense les primitives définies directement dans dsl.py."""
    primitives = {}

    for name, function in inspect.getmembers(dsl, inspect.isfunction):
        # Ignore les fonctions importées depuis d'autres modules.
        if function.__module__ != dsl.__name__:
            continue

        primitives[name] = {
            "function": function,
            "signature": inspect.signature(function),
            "type_hints": get_type_hints(function),
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
