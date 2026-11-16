from typing import Any, FrozenSet, Tuple, get_args, get_origin

import dsl
from arc_types import Cell, Grid, Patch
from search.dsl_registry import load_registry
from search.expression_validator import infer_expression_type
from search.function_types import FunctionType
from search.python_to_dsl import parse_solver
from search.type_system import get_parameters, is_compatible


def infer(body, registry):
    source = "def solve_test(I):\n" + "".join(
        f"    {line}\n" for line in body
    )

    return infer_expression_type(parse_solver(source, registry), registry)


def test_pair_is_a_tuple_of_element_pairs():
    registry = load_registry()

    # pair(a, b) vaut tuple(zip(a, b)).
    result = infer(
        [
            "x1 = order(objects(I, T, T, T), size)",
            "x2 = apply(color, x1)",
            "x3 = apply(center, x1)",
            "O = pair(x2, x3)",
            "return O",
        ],
        registry,
    )

    assert result == Tuple[Cell, ...]


def test_pair_of_unknown_containers_is_a_tuple_of_unknown_pairs():
    registry = load_registry()

    # totuple retourne un Tuple non paramétré : les éléments sont inconnus.
    result = infer(
        [
            "x1 = totuple(objects(I, T, T, T))",
            "O = pair(x1, x1)",
            "return O",
        ],
        registry,
    )

    assert result == Tuple[Tuple[Any, Any], ...]


def test_paint_accepts_a_tuple_of_cells():
    registry = load_registry()

    obj_type = get_parameters(registry["primitives"]["paint"])[1]["type"]

    assert is_compatible(Tuple[Cell, ...], obj_type)

    cells = ((1, (0, 0)), (2, (1, 1)))

    assert dsl.paint(dsl.canvas(0, (2, 2)), cells) == ((1, 0), (0, 2))

    # Tâche 780d0b14 : paint(grille, pair(couleurs, centres)).
    result = infer(
        [
            "x1 = totuple(objects(I, T, T, T))",
            "x2 = pair(apply(color, x1), apply(center, x1))",
            "O = paint(I, x2)",
            "return O",
        ],
        registry,
    )

    assert is_compatible(result, Grid)


def test_fork_apply_maps_the_function_over_the_container():
    registry = load_registry()

    # Tâche 3eda0437 : fork(apply, a, b)(x) vaut apply(a(x), b(x)), soit
    # ici l'ensemble des occurrences décalées (un ensemble de patchs).
    result = infer(
        [
            "x1 = lbind(occurrences, I)",
            "x2 = lbind(lbind, shift)",
            "O = fork(apply, x2, x1)",
            "return O",
        ],
        registry,
    )

    assert isinstance(result, FunctionType)
    assert get_origin(result.return_type) is frozenset
    assert get_args(result.return_type) == (Patch,)
    assert result.return_type == FrozenSet[Patch]
