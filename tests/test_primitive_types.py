from typing import Callable

from arc_types import Grid
from search.dsl_registry import load_registry
from search.type_system import get_return_type, is_compatible


def test_grid_primitives_can_be_identified():
    registry = load_registry()
    primitives = registry["primitives"]

    assert is_compatible(
        get_return_type(primitives["rot90"]),
        Grid,
    )

    assert is_compatible(
        get_return_type(primitives["replace"]),
        Grid,
    )

    assert is_compatible(
        get_return_type(primitives["fill"]),
        Grid,
    )


def test_function_combinators_return_callable():
    registry = load_registry()
    primitives = registry["primitives"]

    assert is_compatible(
        get_return_type(primitives["compose"]),
        Callable,
    )

    assert is_compatible(
        get_return_type(primitives["fork"]),
        Callable,
    )
