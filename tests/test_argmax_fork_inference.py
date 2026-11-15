import inspect
from typing import Any, FrozenSet, Tuple, Union

from arc_types import Grid, Object, Objects
from search.dsl_registry import load_registry
from search.expression_validator import (
    _container_element_type,
    infer_expression_type,
)
from search.expressions import PrimitiveExpr, PrimitiveRefExpr
from search.function_types import FunctionType
from search.python_to_dsl import parse_solver
from search.type_system import is_compatible


def test_element_type_of_parametrized_containers():
    assert _container_element_type(Objects) == Object
    assert _container_element_type(Tuple[int, ...]) is int
    assert _container_element_type(Tuple[int, bool]) == Union[int, bool]
    assert _container_element_type(
        Union[FrozenSet[int], Tuple[bool, ...]]
    ) == Union[int, bool]


def test_element_type_of_unknown_containers_is_any():
    assert _container_element_type(FrozenSet) is Any
    assert _container_element_type(Tuple) is Any
    assert _container_element_type(Any) is Any
    assert _container_element_type(Union[FrozenSet[int], FrozenSet]) is Any


def test_fork_argmax_returns_an_element_of_the_container():
    registry = load_registry()

    # fork(argmax, objects-like, size)(x) vaut argmax(a(x), b(x)) :
    # un élément du conteneur a(x), pas le conteneur lui-même.
    expression = PrimitiveExpr(
        "fork",
        (
            PrimitiveRefExpr("argmax"),
            PrimitiveRefExpr("partition"),
            PrimitiveExpr(
                "lbind",
                (PrimitiveRefExpr("rbind"), PrimitiveRefExpr("colorcount")),
                None,
            ),
        ),
        None,
    )

    result = infer_expression_type(expression, registry)

    assert isinstance(result, FunctionType)
    assert result.return_type == Object


def test_fork_argmax_on_unparametrized_container_returns_any():
    registry = load_registry()

    # product retourne un FrozenSet non paramétré : l'élément est inconnu.
    pairs = PrimitiveExpr(
        "fork",
        (
            PrimitiveRefExpr("product"),
            PrimitiveRefExpr("identity"),
            PrimitiveRefExpr("identity"),
        ),
        None,
    )

    expression = PrimitiveExpr(
        "fork",
        (
            PrimitiveRefExpr("argmax"),
            pairs,
            PrimitiveExpr(
                "lbind",
                (PrimitiveRefExpr("rbind"), PrimitiveRefExpr("colorcount")),
                None,
            ),
        ),
        None,
    )

    result = infer_expression_type(expression, registry)

    assert isinstance(result, FunctionType)
    assert result.return_type is Any


def test_solver_a64e4611_is_typed_as_grid():
    import solvers

    registry = load_registry()
    source = inspect.getsource(solvers.solve_a64e4611)
    expression = parse_solver(source, registry)

    assert is_compatible(infer_expression_type(expression, registry), Grid)
