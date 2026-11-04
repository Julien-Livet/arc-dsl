"""Génération de programmes DSL guidée par les types."""

from collections.abc import Callable as ABCCallable
from dataclasses import dataclass, field
from typing import Any, Callable, get_origin, get_args, Union
import types
from itertools import product
from search.expression_validator import (
    infer_expression_type,
    is_compatible,
    ApplyExpr,
    InputExpr,
    ConstantExpr,
    PrimitiveExpr,
    PrimitiveRefExpr,
    FUNCTION_COMBINATORS,
)
from search.type_system import get_return_type, get_parameters
from search.function_types import FunctionType
import random


@dataclass
class GenerationState:
    expected_type: Any
    max_depth: int
    stack: list = field(default_factory=list)
    result: Any = None
    finished: bool = False


@dataclass
class PendingExpression:
    name: str
    arguments: list
    parameter_types: tuple
    return_type: Any
    depth: int = 0


@dataclass(frozen=True)
class PrimitiveSymbol:
    name: str
    parameter_types: tuple
    required_parameters: tuple
    return_type: Any


@dataclass(frozen=True)
class Candidate:
    expression: Any
    result_type: Any
    depth: int


@dataclass
class TypeGoal:
    """Représente un emplacement à compléter dans une expression."""

    expected_type: Any
    depth: int = 0
    parent: Any = None
    argument_index: int | None = None


def enumerate_candidates(expected_type, registry):
    """
    Énumère les symboles dont le type de retour est potentiellement
    compatible avec le type attendu.

    Les types génériques sont conservés : ils pourront être affinés
    ultérieurement à partir des arguments et du contexte.
    """
    from search.expression_validator import is_compatible

    candidates = []

    for name, info in registry["primitives"].items():
        signature = info["signature"]
        hints = info.get("type_hints", {})

        result_type = hints.get("return", Any)
        parameters = tuple(signature.parameters.values())

        parameter_types = tuple(
            hints.get(parameter.name, Any)
            for parameter in parameters
        )

        # Les annotations de retour peuvent être génériques ou unions.
        # On conserve ces candidats pour ne pas perdre de possibilités.
        try:
            compatible = (
                is_compatible(result_type, expected_type)
                or is_compatible(expected_type, result_type)
            )
        except (TypeError, ValueError):
            compatible = False

        if compatible or result_type in (Any, object):
            candidates.append(
                Candidate(
                    name=name,
                    kind="primitive",
                    result_type=result_type,
                    parameter_types=parameter_types,
                    parameter_names=tuple(p.name for p in parameters),
                )
            )

    for name, info in registry["constants"].items():
        constant_type = info["type"]

        try:
            compatible = (
                is_compatible(constant_type, expected_type)
                or is_compatible(expected_type, constant_type)
            )
        except (TypeError, ValueError):
            compatible = False

        if compatible:
            candidates.append(
                Candidate(
                    name=name,
                    kind="constant",
                    result_type=constant_type,
                    value=info["value"],
                )
            )

    return sorted(candidates, key=lambda candidate: (candidate.kind, candidate.name))


def infer_candidate_type(expression, registry):
    """Infère le type effectif d'une expression DSL."""
    from search.expression_validator import infer_expression_type

    return infer_expression_type(expression, registry)


@dataclass(frozen=True)
class GeneratedCandidate:
    expression: Any
    result_type: Any
    depth: int


def types_match(actual_type, expected_type):
    """Vérifie la compatibilité dans les deux sens, comme le validateur."""
    try:
        return (
            is_compatible(actual_type, expected_type)
            or is_compatible(expected_type, actual_type)
        )
    except (TypeError, ValueError):
        return False


def enumerate_expressions(
    expected_type,
    registry,
    max_depth=1,
    context=None,
):
    """
    Énumère des expressions compatibles avec expected_type.

    max_depth=0 : entrée et constantes.
    max_depth=1 : ajoute les primitives à un seul niveau.
    Les combinateurs fonctionnels sont temporairement exclus.
    """
    if context is None:
        context = {}

    candidates = []

    # Entrée du programme.
    input_expression = InputExpr(name="I")
    try:
        input_type = infer_expression_type(input_expression, registry)
        if types_match(input_type, expected_type):
            candidates.append(
                GeneratedCandidate(input_expression, input_type, 0)
            )
    except (TypeError, ValueError):
        pass

    # Constantes connues.
    for name, info in registry["constants"].items():
        expression = ConstantExpr(
            name=name,
            value=info["value"],
            type_hint=info["type"],
        )

        try:
            result_type = infer_expression_type(expression, registry)
        except (TypeError, ValueError):
            continue

        if types_match(result_type, expected_type):
            candidates.append(
                GeneratedCandidate(expression, result_type, 0)
            )

    if max_depth <= 0:
        return candidates

    # Primitives ordinaires.
    for name, info in registry["primitives"].items():
        if name in FUNCTION_COMBINATORS:
            continue

        signature = info["signature"]
        parameters = list(signature.parameters.values())

        # Première version : chaque argument doit être une valeur
        # directement disponible dans le contexte ou une constante.
        argument_options = []

        for parameter in parameters:
            parameter_type = info["type_hints"].get(
                parameter.name,
                Any,
            )

            options = []

            for candidate in candidates:
                if types_match(candidate.result_type, parameter_type):
                    options.append(candidate.expression)

            if parameter.default is not parameter.empty:
                # Les valeurs par défaut pourront être prises en charge
                # explicitement dans une étape ultérieure.
                pass

            if not options:
                argument_options = []
                break

            argument_options.append(options)

        if not argument_options:
            continue

        # Pour l'instant, une seule combinaison est testée :
        # la première possibilité de chaque argument.
        arguments = tuple(options[0] for options in argument_options)
        expression = PrimitiveExpr(name=name, arguments=arguments,
                                   return_type=info["type_hints"].get("return", Any),)

        try:
            result_type = infer_expression_type(expression, registry)
        except (TypeError, ValueError):
            continue

        if types_match(result_type, expected_type):
            candidates.append(
                GeneratedCandidate(
                    expression,
                    result_type,
                    max_depth,
                )
            )

    return candidates


def is_terminal_expression(expression):
    return isinstance(
        expression,
        (InputExpr, ConstantExpr, PrimitiveRefExpr),
    )


def infer_candidate(expression, registry, depth=0):
    """Infère le type d'une expression et crée un candidat."""
    try:
        result_type = infer_expression_type(expression, registry)
    except (TypeError, ValueError):
        return None

    return Candidate(
        expression=expression,
        result_type=result_type,
        depth=depth,
    )


def get_terminal_candidates(registry):
    """Construit les candidats terminaux du DSL."""
    candidates = []

    # Grille d'entrée.
    candidate = infer_candidate(InputExpr(), registry)
    if candidate is not None:
        candidates.append(candidate)

    # Constantes.
    for name, info in registry["constants"].items():
        expression = ConstantExpr(
            name=name,
            value=info["value"],
            type_hint=info["type"],
        )

        candidate = infer_candidate(expression, registry)

        if candidate is not None:
            candidates.append(candidate)

    # Références aux primitives : fonctions non appliquées.
    for name in registry["primitives"]:
        expression = PrimitiveRefExpr(name=name)

        candidate = infer_candidate(expression, registry)

        if candidate is not None:
            candidates.append(candidate)

    return candidates


def get_compatible_candidates(
    expected_type,
    available_candidates,
):
    """Retourne les candidats compatibles avec le type attendu."""
    return [
        candidate
        for candidate in available_candidates
        if matches_type(candidate.result_type, expected_type)
    ]


def expand_primitive(name, primitive_info, registry, available_candidates, depth):
    """Construit les expressions possibles pour une primitive."""

    parameters = get_parameters(primitive_info)

    # Une primitive sans argument est directement constructible.
    if not parameters:
        expression = PrimitiveExpr(
            name=name,
            arguments=(),
            return_type=get_return_type(primitive_info),
        )

        candidate = infer_candidate(expression, registry, depth)
        return [candidate] if candidate else []

    argument_options = []

    for parameter in parameters:
        expected_type = parameter["type"]

        options = get_compatible_candidates(
            expected_type,
            available_candidates,
        )

        # Un paramètre optionnel peut être omis.
        if not parameter["required"]:
            argument_options.append(options + [None])
        else:
            argument_options.append(options)

        if not argument_options[-1]:
            return []

    results = []

    for combination in product(*argument_options):
        # Pour l'instant, on ne traite pas encore l'omission
        # d'un argument intermédiaire.
        if any(argument is None for argument in combination):
            continue

        arguments = tuple(
            candidate.expression
            for candidate in combination
        )

        expression = PrimitiveExpr(
            name=name,
            arguments=arguments,
            return_type=get_return_type(primitive_info),
        )

        candidate = infer_candidate(expression, registry, depth)

        if candidate is not None:
            results.append(candidate)

    return results


def expand_catalogue(registry, catalogue, depth):
    """Ajoute au catalogue les primitives constructibles à cette profondeur."""

    new_candidates = []

    for name, primitive_info in registry["primitives"].items():
        candidates = expand_primitive(
            name=name,
            primitive_info=primitive_info,
            registry=registry,
            available_candidates=catalogue,
            depth=depth,
        )

        new_candidates.extend(candidates)

    return new_candidates


def build_catalogue(registry, max_depth):
    """Construit un catalogue d'expressions jusqu'à max_depth."""

    catalogue = get_terminal_candidates(registry)

    for depth in range(1, max_depth + 1):
        new_candidates = expand_catalogue(
            registry,
            catalogue,
            depth,
        )

        catalogue.extend(new_candidates)

    return catalogue


def build_symbol_catalogue(registry):
    symbols = []

    for name, info in registry["primitives"].items():
        parameters = get_parameters(info)

        symbols.append(
            PrimitiveSymbol(
                name=name,
                parameter_types=tuple(
                    parameter["type"]
                    for parameter in parameters
                ),
                required_parameters=tuple(
                    parameter["required"]
                    for parameter in parameters
                ),
                return_type=get_return_type(info),
            )
        )

    return symbols


def matches_type(actual_type, expected_type):
    """Teste la compatibilité entre un type réel et un type attendu."""

    if isinstance(actual_type, FunctionType):
        if isinstance(expected_type, FunctionType):
            return actual_type.is_compatible_with(expected_type)

        return expected_type in (Callable, ABCCallable)

    if isinstance(expected_type, FunctionType):
        return False

    try:
        return is_compatible(actual_type, expected_type)
    except (TypeError, ValueError):
        return False

def build_primitive_reference(name, registry):
    expression = PrimitiveRefExpr(name=name)

    return infer_candidate(expression, registry)


def create_pending_expression(name, registry):
    """Crée une expression incomplète à partir d'une primitive."""
    primitive_info = registry["primitives"][name]

    parameters = get_parameters(primitive_info)

    required_parameters = [
        parameter
        for parameter in parameters
        if parameter["required"]
    ]

    return PendingExpression(
        name=name,
        arguments=[None] * len(required_parameters),
        parameter_types=tuple(
            parameter["type"]
            for parameter in required_parameters
        ),
        return_type=get_return_type(primitive_info),
    )


def get_next_argument_goal(pending):
    """Retourne le type attendu pour le prochain argument."""
    for argument, expected_type in zip(
        pending.arguments,
        pending.parameter_types,
    ):
        if argument is None:
            return expected_type

    return None


def may_return_type(declared_type, expected_type):
    """Indique si le type déclaré peut correspondre au type attendu."""

    if declared_type is None or declared_type is Any:
        return True

    if expected_type is Any:
        return True

    # Une union peut contenir une branche compatible.
    if get_origin(declared_type) in (Union, types.UnionType):
        return any(
            may_return_type(branch, expected_type)
            for branch in get_args(declared_type)
        )

    if get_origin(expected_type) in (Union, types.UnionType):
        return any(
            may_return_type(declared_type, branch)
            for branch in get_args(expected_type)
        )

    # Vérification dans le sens résultat réel -> résultat attendu.
    return is_compatible(declared_type, expected_type)


def get_compatible_primitives(expected_type, registry):
    """Retourne les primitives potentiellement compatibles."""

    candidates = []

    for name, primitive_info in registry["primitives"].items():

        if name in FUNCTION_COMBINATORS:
            if name in {"branch", "mapply", "lbind", "rbind", "fork", "compose", "chain", "mfilter"}:
                declared_return = get_return_type(primitive_info)

                if may_return_type(declared_return, expected_type):
                    candidates.append(name)

            continue

        declared_return = get_return_type(primitive_info)

        if may_return_type(declared_return, expected_type):
            candidates.append(name)

    return candidates


def get_compatible_terminals(expected_type, registry):
    """Retourne les constantes et l'entrée compatibles."""

    candidates = []

    from arc_types import Grid

    input_type = Grid

    if is_compatible(input_type, expected_type):
        candidates.append(InputExpr())

    for name, constant_info in registry["constants"].items():
        constant_type = constant_info["type"]

        if is_compatible(constant_type, expected_type):
            candidates.append(
                ConstantExpr(
                    name=name,
                    value=constant_info["value"],
                    type_hint=constant_type,
                )
            )

    return candidates


def get_candidates(expected_type, registry):
    """Retourne les primitives et terminaux à examiner."""

    return {
        "terminals": get_compatible_terminals(
            expected_type,
            registry,
        ),
        "primitives": get_compatible_primitives(
            expected_type,
            registry,
        ),
    }


def validate_expression(expression, expected_type, registry):
    actual_type = infer_expression_type(expression, registry)

    return is_compatible(actual_type, expected_type)


def generate_expression(
    expected_type,
    registry,
    max_depth=4,
    depth=0,
):
    """Génère récursivement une expression compatible avec un type."""

    # À profondeur maximale, on ne propose que des terminaux.
    terminals = get_compatible_terminals(expected_type, registry)

    if depth >= max_depth:
        if terminals:
            return random.choice(terminals)

        return None

    candidates = get_candidates(expected_type, registry)

    # Les terminaux et les primitives sont réunis.
    choices = (
        [("terminal", candidate)
         for candidate in candidates["terminals"]]
        +
        [("primitive", name)
         for name in candidates["primitives"]]
    )

    random.shuffle(choices)

    for kind, candidate in choices:

        if kind == "terminal":
            return candidate

        pending = create_pending_expression(candidate, registry)

        success = True

        for index, argument_type in enumerate(
            pending.parameter_types
        ):
            if isinstance(argument_type, FunctionType):
                argument = generate_function_reference(
                    argument_type,
                    registry,
                )
            else:
                argument = generate_expression(
                    expected_type=argument_type,
                    registry=registry,
                    max_depth=max_depth,
                    depth=depth + 1,
                )

            if argument is None:
                success = False
                break

            pending.arguments[index] = argument

        if not success:
            continue

        expression = PrimitiveExpr(
            name=pending.name,
            arguments=tuple(pending.arguments),
            return_type=pending.return_type,
        )

        try:
            if validate_expression(expression, expected_type, registry):
                return expression
        except TypeError:
            continue

    return None


def is_function_type(type_hint):
    """Indique si un type attendu correspond à une fonction."""

    if type_hint is FunctionType:
        return True

    if type_hint is Any:
        return False

    if type_hint in (Callable, ABCCallable):
        return True

    origin = get_origin(type_hint)

    return origin in (Callable, ABCCallable)


def generate_function_reference(expected_type, registry):
    """Sélectionne une primitive dont la signature correspond."""

    candidates = []

    for name, primitive_info in registry["primitives"].items():
        parameters = get_parameters(primitive_info)

        if any(not parameter["required"] for parameter in parameters):
            continue

        actual_type = FunctionType(
            parameter_types=tuple(
                parameter["type"] for parameter in parameters
            ),
            return_type=get_return_type(primitive_info),
        )

        if isinstance(expected_type, FunctionType) and actual_type.is_compatible_with(expected_type):
            candidates.append(PrimitiveRefExpr(name=name))

    if not candidates:
        return None

    return random.choice(candidates)


def get_next_symbols(expected_type, registry, depth=0, max_depth=4):
    """Retourne les symboles compatibles avec le prochain objectif."""

    candidates = []

    candidates.extend(
        get_compatible_terminals(expected_type, registry)
    )

    candidates.extend(
        get_compatible_function_references(expected_type, registry)
    )

    if depth < max_depth:
        candidates.extend(
            get_compatible_primitives(expected_type, registry)
        )

    return candidates


def complete_expression(state, symbol, registry):
    """Ajoute un symbole à l'expression en cours."""

    # Cas d'un terminal : l'objectif est satisfait.
    if isinstance(symbol, (InputExpr, ConstantExpr)):
        return symbol

    # Cas d'une primitive : création d'une expression incomplète.
    pending = create_pending_expression(symbol, registry)

    # Une primitive sans argument est immédiatement complète.
    if not pending.parameter_types:
        expression = PrimitiveExpr(
            name=pending.name,
            arguments=(),
            return_type=pending.return_type,
        )

        actual_type = infer_expression_type(expression, registry)

        if is_compatible(actual_type, state.expected_type):
            return expression

        return None

    # Sinon, on initialise les arguments à compléter.
    return GenerationState(
        expected_type=state.expected_type,
        max_depth=state.max_depth,
        depth=state.depth,
        pending=pending,
    )


def get_next_goal(state):
    """Retourne le type attendu du prochain argument."""

    if state.pending is None:
        return None

    for index, argument in enumerate(state.pending.arguments):
        if argument is None:
            return state.pending.parameter_types[index]

    return None


def add_completed_argument(state, argument, registry):
    """Ajoute un argument déjà construit."""

    goal = get_next_goal(state)

    if goal is None:
        raise ValueError("Aucun argument restant.")

    actual_type = infer_expression_type(argument, registry)

    if not is_compatible(actual_type, goal):
        raise TypeError(
            f"Type incompatible : {actual_type}, attendu : {goal}"
        )

    index = state.pending.arguments.index(None)
    state.pending.arguments[index] = argument

    # Tous les arguments sont-ils remplis ?
    if all(arg is not None for arg in state.pending.arguments):
        expression = PrimitiveExpr(
            name=state.pending.name,
            arguments=tuple(state.pending.arguments),
            return_type=state.pending.return_type,
        )

        return expression

    return state


def get_current_goal(state):
    """Retourne le type attendu pour le prochain symbole."""

    if state.finished:
        return None

    if not state.stack:
        return state.expected_type

    pending = state.stack[-1]

    for argument, expected_type in zip(
        pending.arguments,
        pending.parameter_types,
    ):
        if argument is None:
            return expected_type

    return None


def attach_expression(state, expression, registry):
    """Insère une expression terminée dans son parent."""

    while True:

        # Aucun parent : on vient de terminer la racine.
        if not state.stack:
            actual_type = infer_expression_type(expression, registry)

            if (
                isinstance(actual_type, FunctionType)
                and not is_compatible(actual_type, state.expected_type)
            ):
                pending = create_pending_application(expression, actual_type)
                state.stack.append(pending)
                return state

            if not is_compatible(actual_type, state.expected_type):
                raise TypeError(
                    f"Résultat incompatible : {actual_type}, "
                    f"attendu : {state.expected_type}"
                )

            state.result = expression
            state.finished = True
            return state

        parent = state.stack[-1]

        # Trouver le prochain emplacement disponible.
        index = next(
            i for i, argument in enumerate(parent.arguments)
            if argument is None
        )

        expected_type = parent.parameter_types[index]
        actual_type = infer_expression_type(expression, registry)

        if not is_compatible(actual_type, expected_type):
            raise TypeError(
                f"Argument incompatible pour {parent.name} : "
                f"{actual_type}, attendu : {expected_type}"
            )

        parent.arguments[index] = expression

        # Le parent est-il terminé ?
        if any(arg is None for arg in parent.arguments):
            return state

        # Oui : construire l'expression complète.
        state.stack.pop()

        if parent.name == "__apply__":
            expression = ApplyExpr(
                function=parent.arguments[0],
                arguments=tuple(parent.arguments[1:]),
            )
        else:
            expression = PrimitiveExpr(
                name=parent.name,
                arguments=tuple(parent.arguments),
                return_type=parent.return_type,
            )

        # Vérifier son type réel.
        actual_type = infer_expression_type(expression, registry)

        # On continue la remontée.
        if state.stack:
            continue

        if (isinstance(actual_type, FunctionType)):
            actual_type = actual_type.return_type

        if not is_compatible(actual_type, state.expected_type):
            raise TypeError(
                f"Résultat incompatible : {actual_type}, "
                f"attendu : {state.expected_type}"
            )

        state.result = expression
        state.finished = True
        return state


def select_symbol(state, symbol, registry):
    """Ajoute un symbole choisi à la génération."""

    if state.finished:
        raise ValueError("La génération est déjà terminée.")

    # Cas d'un terminal.
    if isinstance(symbol, (InputExpr, ConstantExpr)):
        return attach_expression(state, symbol, registry)

    # Cas d'une référence de fonction : à traiter séparément.
    if isinstance(symbol, PrimitiveRefExpr):
        return attach_expression(state, symbol, registry)

    # Cas d'une primitive.
    if isinstance(symbol, str):
        pending = create_pending_expression(symbol, registry)

        pending = PendingExpression(
            name=pending.name,
            arguments=pending.arguments,
            parameter_types=pending.parameter_types,
            return_type=pending.return_type,
            depth=len(state.stack),
        )

        # Primitive sans argument.
        if not pending.arguments:
            expression = PrimitiveExpr(
                name=pending.name,
                arguments=(),
                return_type=pending.return_type,
            )
            return attach_expression(state, expression, registry)

        state.stack.append(pending)

        return state

    raise TypeError(f"Symbole non pris en charge : {symbol}")


def get_compatible_function_references(expected_type, registry):
    candidates = []

    for name in registry["primitives"]:
        ref = PrimitiveRefExpr(name=name)
        actual_type = infer_expression_type(ref, registry)

        if expected_type is Any:
            compatible = True
        elif isinstance(expected_type, FunctionType):
            compatible = actual_type.is_compatible_with(expected_type)
        else:
            compatible = is_compatible(actual_type, expected_type)

        if compatible:
            candidates.append(ref)

    return candidates


def create_pending_application(function_expression, function_type):
    if not isinstance(function_type, FunctionType):
        raise TypeError("Le résultat n'est pas une fonction.")

    if function_type.variadic:
        raise NotImplementedError(
            "Les applications variadiques ne sont pas encore gérées."
        )

    return PendingExpression(
        name="__apply__",
        arguments=[function_expression] + [None] * function_type.arity,
        parameter_types=(Any,) + function_type.parameter_types,
        return_type=function_type.return_type,
        depth=0,
    )


def walk_expression(expression):
    """Parcourt une expression en ordre préfixe."""

    yield expression

    if isinstance(expression, PrimitiveExpr):
        for argument in expression.arguments:
            yield from walk_expression(argument)

    elif isinstance(expression, ApplyExpr):
        yield from walk_expression(expression.function)

        for argument in expression.arguments:
            yield from walk_expression(argument)


def get_expression_node(expression, index):
    if index < 0:
        raise IndexError(index)

    for i, node in enumerate(walk_expression(expression)):
        if i == index:
            return node

    raise IndexError(index)
