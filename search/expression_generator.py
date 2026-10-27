from search.expressions import (
    InputExpr,
    ConstantExpr,
    PrimitiveRefExpr,
)
from search.expression_validator import infer_expression_type
from search.type_system import is_compatible
from typing import Any

def generate_leaves(registry):
    """Génère les expressions feuilles disponibles dans le registre."""

    leaves = [InputExpr()]

    for name, info in registry["constants"].items():
        leaves.append(
            ConstantExpr(
                name=name,
                value=info["value"],
                type_hint=info["type"],
            )
        )

    for name in registry["primitives"]:
        leaves.append(PrimitiveRefExpr(name))

    return leaves


def generate_leaves_for_type(registry, expected_type):
    """Génère les feuilles compatibles avec le type cible."""

    compatible_leaves = []

    for expression in generate_leaves(registry):
        try:
            actual_type = infer_expression_type(expression, registry)
        except (TypeError, ValueError):
            continue

        if is_compatible(actual_type, expected_type):
            compatible_leaves.append(expression)

    return compatible_leaves


def get_primitives_returning_type(registry, expected_type):
    """Retourne les primitives dont le type de retour est compatible."""

    from search.type_system import get_return_type, is_compatible

    candidates = []

    for name, info in registry["primitives"].items():
        return_type = get_return_type(info)

        if return_type is None:
            continue

        if is_compatible(return_type, expected_type):
            candidates.append(name)

    return candidates


def generate_unary_applications(registry, argument_expressions, expected_type):
    """Génère les applications unaires compatibles avec le type cible."""

    from search.expressions import ApplyExpr, PrimitiveRefExpr
    from search.expression_validator import infer_expression_type
    from search.type_system import is_compatible

    candidates = []

    for name, info in registry["primitives"].items():
        parameters = info["signature"].parameters
        type_hints = info["type_hints"]

        if len(parameters) != 1:
            continue

        parameter = next(iter(parameters.values()))
        parameter_type = type_hints.get(parameter.name)
        return_type = type_hints.get("return")

        if parameter_type is None or return_type is None:
            continue

        # Écarter les retours trop imprécis pour garantir le type cible.
        if return_type is Any:
            continue

        if not is_compatible(return_type, expected_type):
            continue

        for argument in argument_expressions:
            try:
                actual_argument_type = infer_expression_type(argument, registry)
            except (TypeError, ValueError):
                continue

            if not is_compatible(actual_argument_type, parameter_type):
                continue

            expression = ApplyExpr(
                function=PrimitiveRefExpr(name),
                arguments=(argument, ),
            )

            try:
                actual_type = infer_expression_type(expression, registry)
            except (TypeError, ValueError):
                continue

            if is_compatible(actual_type, expected_type):
                candidates.append(expression)

    return candidates


def generate_unary_expressions(registry, expected_type, max_depth=2):
    """Génère des expressions unaires en explorant les types intermédiaires."""

    from search.expression_validator import infer_expression_type
    from search.type_system import is_compatible

    if max_depth < 1:
        return []

    current_level = [InputExpr()]
    expressions = []
    type_cache = {}

    def get_expression_type(expression):
        key = id(expression)

        cached = type_cache.get(key)
        if cached is not None:
            cached_expression, cached_type = cached
            if cached_expression is expression:
                return cached_type

        inferred_type = infer_expression_type(expression, registry)
        type_cache[key] = (expression, inferred_type)
        return inferred_type

    for _ in range(max_depth):
        current_level = generate_unary_intermediate_applications(
            registry,
            current_level,
            infer_type=get_expression_type,
        )

        if not current_level:
            break

        for expression in current_level:
            try:
                actual_type = get_expression_type(expression)
            except (TypeError, ValueError):
                continue

            if is_compatible(actual_type, expected_type):
                expressions.append(expression)

    return expressions


def generate_unary_intermediate_applications(registry, argument_expressions, infer_type=None):
    """Génère des applications unaires sans imposer le type final."""

    from search.expressions import ApplyExpr, PrimitiveRefExpr
    from search.expression_validator import infer_expression_type
    from search.type_system import is_compatible

    if infer_type is None:
        infer_type = lambda expression: infer_expression_type(
            expression,
            registry,
        )

    candidates = []

    for name, info in registry["primitives"].items():
        parameters = info["signature"].parameters
        type_hints = info["type_hints"]

        if len(parameters) != 1:
            continue

        parameter = next(iter(parameters.values()))
        parameter_type = type_hints.get(parameter.name)
        return_type = type_hints.get("return")

        if parameter_type is None or return_type is None:
            continue

        for argument in argument_expressions:
            try:
                actual_argument_type = infer_type(argument)
            except (TypeError, ValueError):
                continue

            if not is_compatible(actual_argument_type, parameter_type):
                continue

            expression = ApplyExpr(
                function=PrimitiveRefExpr(name),
                arguments=(argument, ),
            )

            try:
                infer_type(expression)
            except (TypeError, ValueError):
                continue

            candidates.append(expression)

    return candidates


def generate_binary_applications(
    registry,
    first_argument_expressions,
    second_argument_expressions,
    infer_type=None,
):
    """Génère des applications binaires en réutilisant l'inférence des types."""

    from search.expressions import ApplyExpr, PrimitiveRefExpr
    from search.expression_validator import infer_expression_type
    from search.type_system import is_compatible

    if infer_type is None:
        infer_type = lambda expression: infer_expression_type(
            expression,
            registry,
        )

    candidates = []
    compatibility_cache = {}

    def cached_is_compatible(actual_type, expected_type):
        key = (actual_type, expected_type)
        if key not in compatibility_cache:
            compatibility_cache[key] = is_compatible(
                actual_type,
                expected_type,
            )
        return compatibility_cache[key]

    typed_first_arguments = []
    for expression in first_argument_expressions:
        try:
            actual_type = infer_type(expression)
        except (TypeError, ValueError):
            continue
        typed_first_arguments.append((expression, actual_type))

    typed_second_arguments = []
    for expression in second_argument_expressions:
        try:
            actual_type = infer_type(expression)
        except (TypeError, ValueError):
            continue
        typed_second_arguments.append((expression, actual_type))

    for name, info in registry["primitives"].items():
        parameters = info["signature"].parameters
        type_hints = info["type_hints"]

        if len(parameters) != 2:
            continue

        parameter_types = [
            type_hints.get(parameter_name)
            for parameter_name in parameters
        ]

        if any(parameter_type is None for parameter_type in parameter_types):
            continue

        first_type, second_type = parameter_types

        compatible_first_arguments = [
            expression
            for expression, actual_type in typed_first_arguments
            if cached_is_compatible(actual_type, first_type)
        ]

        compatible_second_arguments = [
            expression
            for expression, actual_type in typed_second_arguments
            if cached_is_compatible(actual_type, second_type)
        ]

        for first_argument in compatible_first_arguments:
            for second_argument in compatible_second_arguments:
                expression = ApplyExpr(
                    function=PrimitiveRefExpr(name),
                    arguments=(first_argument, second_argument),
                )

                try:
                    infer_type(expression)
                except (TypeError, ValueError):
                    continue

                candidates.append(expression)

    return list(dict.fromkeys(candidates))


def generate_expressions(registry, expected_type, max_depth=2):
    """Génère des expressions unaires par profondeur."""

    from search.expression_validator import infer_expression_type
    from search.type_system import is_compatible

    if max_depth < 1:
        return []

    type_cache = {}

    def get_expression_type(expression):
        key = id(expression)

        cached = type_cache.get(key)
        if cached is not None:
            cached_expression, cached_type = cached
            if cached_expression is expression:
                return cached_type

        inferred_type = infer_expression_type(expression, registry)
        type_cache[key] = (expression, inferred_type)
        return inferred_type

    current_level = [InputExpr()]
    expressions = []
    intermediate_expressions = []

    for _ in range(max_depth):
        current_level = generate_unary_intermediate_applications(
            registry,
            current_level,
            infer_type=get_expression_type,
        )

        if not current_level:
            break

        for expression in current_level:
            try:
                actual_type = get_expression_type(expression)
            except (TypeError, ValueError):
                continue

            intermediate_expressions.append(expression)

            if is_compatible(actual_type, expected_type):
                expressions.append(expression)

    constants = [
        expression
        for expression in generate_leaves(registry)
        if isinstance(expression, ConstantExpr)
    ]

    # Conserve les applications binaires entre constantes.
    binary_expressions = generate_binary_applications(
        registry,
        constants,
        constants,
        infer_type=get_expression_type,
    )

    # Ajoute les expressions unaires déjà générées comme arguments possibles.
    binary_expressions.extend(
        generate_binary_applications(
            registry,
            intermediate_expressions,
            intermediate_expressions,
            infer_type=get_expression_type,
        )
    )

    for expression in binary_expressions:
        try:
            actual_type = get_expression_type(expression)
        except (TypeError, ValueError):
            continue

        if is_compatible(actual_type, expected_type) or is_compatible(expected_type, actual_type):
            expressions.append(expression)

    return list(dict.fromkeys(expressions))
