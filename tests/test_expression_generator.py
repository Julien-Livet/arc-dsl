import pytest

from arc_types import Grid
from search.dsl_registry import load_registry
from search.expression_generator import generate_leaves, generate_leaves_for_type
from search.expressions import (
    InputExpr,
    ConstantExpr,
    PrimitiveRefExpr,
)
from search.type_system import is_compatible


@pytest.fixture
def registry():
    return load_registry()


def test_generate_leaves_includes_input(registry):
    leaves = generate_leaves(registry)

    assert InputExpr() in leaves


def test_generate_leaves_includes_constants(registry):
    leaves = generate_leaves(registry)

    constant_names = {
        expression.name
        for expression in leaves
        if isinstance(expression, ConstantExpr)
    }

    assert constant_names == set(registry["constants"])


def test_generate_leaves_includes_primitive_references(registry):
    leaves = generate_leaves(registry)

    primitive_names = {
        expression.name
        for expression in leaves
        if isinstance(expression, PrimitiveRefExpr)
    }

    assert primitive_names == set(registry["primitives"])


def test_generate_grid_leaves_includes_input(registry):
    leaves = generate_leaves_for_type(registry, Grid)

    assert InputExpr() in leaves


def test_generate_integer_leaves_excludes_input(registry):
    leaves = generate_leaves_for_type(registry, int)

    assert InputExpr() not in leaves

    assert all(
        isinstance(expression, ConstantExpr)
        for expression in leaves
    )


def test_get_primitives_returning_grid_includes_rotations(registry):
    from search.expression_generator import get_primitives_returning_type

    candidates = get_primitives_returning_type(registry, Grid)

    assert "rot90" in candidates
    assert "rot180" in candidates


def test_generate_unary_grid_applications(registry):
    from search.expression_generator import generate_unary_applications
    from search.expression_validator import infer_expression_type
    from search.expressions import ApplyExpr

    candidates = generate_unary_applications(
        registry,
        [InputExpr()],
        Grid,
    )

    assert candidates
    assert all(isinstance(expression, ApplyExpr) for expression in candidates)

    for expression in candidates:
        assert is_compatible(infer_expression_type(expression, registry), Grid)


def test_generate_nested_unary_applications(registry):
    from search.expression_generator import generate_unary_applications
    from search.expression_validator import infer_expression_type

    first_level = generate_unary_applications(
        registry,
        [InputExpr()],
        Grid,
    )

    second_level = generate_unary_applications(
        registry,
        first_level,
        Grid,
    )

    assert second_level

    for expression in second_level:
        assert is_compatible(infer_expression_type(expression, registry), Grid)


def test_generate_unary_expressions_respects_max_depth(registry):
    from search.expression_generator import generate_unary_expressions
    from search.expressions import ApplyExpr

    expressions = generate_unary_expressions(
        registry,
        Grid,
        max_depth=2,
    )

    assert expressions
    assert all(isinstance(expression, ApplyExpr) for expression in expressions)


def test_generate_unary_expressions_with_zero_depth_returns_empty(registry):
    from search.expression_generator import generate_unary_expressions

    assert generate_unary_expressions(registry, Grid, max_depth=0) == []


def test_unary_generator_only_keeps_expressions_matching_target_type(registry):
    from arc_types import Grid, Integer
    from search.expression_generator import generate_unary_applications
    from search.expressions import InputExpr
    from search.expression_validator import infer_expression_type
    from search.type_system import is_compatible

    expressions = generate_unary_applications(
        registry,
        [InputExpr()],
        Integer,
    )

    for expression in expressions:
        actual_type = infer_expression_type(expression, registry)
        assert is_compatible(actual_type, Integer)

    grid_expressions = generate_unary_applications(
        registry,
        [InputExpr()],
        Grid,
    )

    for expression in grid_expressions:
        actual_type = infer_expression_type(expression, registry)
        assert is_compatible(actual_type, Grid)


def test_unary_generator_can_build_expression_through_intermediate_type(registry):
    from arc_types import Grid, Integer
    from search.expression_generator import generate_unary_expressions
    from search.expressions import ApplyExpr, InputExpr, PrimitiveRefExpr
    from search.expression_validator import infer_expression_type
    from search.type_system import is_compatible

    expressions = generate_unary_expressions(
        registry,
        Integer,
        max_depth=2,
    )

    expected_expression = ApplyExpr(
        function=PrimitiveRefExpr("color"),
        arguments=(ApplyExpr(
            function=PrimitiveRefExpr("asobject"),
            arguments=(InputExpr(), ), ),
        ),
    )

    assert expected_expression in expressions
    assert is_compatible(
        infer_expression_type(expected_expression, registry),
        Integer,
    )


def test_validator_infers_type_of_color_after_asobject(registry):
    from arc_types import Integer
    from search.expressions import ApplyExpr, InputExpr, PrimitiveRefExpr
    from search.expression_validator import infer_expression_type
    from search.type_system import is_compatible

    expression = ApplyExpr(
        function=PrimitiveRefExpr("color"),
        arguments=(ApplyExpr(
            function=PrimitiveRefExpr("asobject"),
            arguments=(InputExpr(), ), ),
        ),
    )

    actual_type = infer_expression_type(expression, registry)

    assert is_compatible(actual_type, Integer)


def test_unary_intermediate_applications_have_no_duplicates(registry):
    from search.expression_generator import (
        generate_unary_intermediate_applications,
    )
    from search.expressions import InputExpr

    expressions = generate_unary_intermediate_applications(
        registry,
        [InputExpr()],
    )

    assert len(expressions) == len(set(expressions))


def test_unary_generator_caches_expression_types(registry):
    from collections import Counter
    from unittest.mock import patch

    from arc_types import Integer
    from search.expression_validator import infer_expression_type

    original = infer_expression_type
    counts = Counter()

    def counted_inference(expression, expression_registry):
        counts[expression] += 1
        return original(expression, expression_registry)

    with patch(
        "search.expression_validator.infer_expression_type",
        side_effect=counted_inference,
    ):
        expressions = __import__(
            "search.expression_generator",
            fromlist=["generate_unary_expressions"],
        ).generate_unary_expressions(
            registry,
            Integer,
            max_depth=3,
        )

    assert expressions
    assert counts


def test_generate_binary_applications(registry):
    from search.expression_generator import generate_binary_applications
    from search.expressions import ApplyExpr, ConstantExpr, PrimitiveRefExpr
    from search.expression_validator import infer_expression_type

    zero = ConstantExpr(
        name="ZERO",
        value=0,
        type_hint=int,
    )
    one = ConstantExpr(
        name="ONE",
        value=1,
        type_hint=int,
    )

    expressions = generate_binary_applications(
        registry,
        [zero],
        [one],
    )

    assert expressions

    expected = ApplyExpr(
            function=PrimitiveRefExpr("add"),
            arguments=(zero, one)
    )

    assert expected in expressions

    for expression in expressions:
        infer_expression_type(expression, registry)


def test_generate_binary_applications_checks_argument_types(registry):
    from search.expression_generator import generate_binary_applications
    from search.expressions import InputExpr

    # Une grille ne doit pas être acceptée comme premier argument
    # d'une primitive binaire qui attend un entier.
    expressions = generate_binary_applications(
        registry,
        [InputExpr()],
        [],
    )

    assert all(
        expression.function.function.name != "add"
        for expression in expressions
    )


def test_generate_binary_applications_has_no_duplicates(registry):
    from search.expression_generator import generate_binary_applications
    from search.expressions import ConstantExpr

    zero = ConstantExpr(
        name="ZERO",
        value=0,
        type_hint=int,
    )
    one = ConstantExpr(
        name="ONE",
        value=1,
        type_hint=int,
    )

    expressions = generate_binary_applications(
        registry,
        [zero, zero],
        [one, one],
    )

    assert len(expressions) == len(set(expressions))


def test_generate_binary_applications_with_nested_expression():
    from search.expressions import ApplyExpr, ConstantExpr, InputExpr, PrimitiveRefExpr
    from search.expression_generator import generate_binary_applications
    from search.expression_validator import infer_expression_type

    registry = load_registry()

    nested_expression = ApplyExpr(
        function=PrimitiveRefExpr("color"),
        arguments=(ApplyExpr(
            function=PrimitiveRefExpr("asobject"),
            arguments=(InputExpr(), ), ),
        ),
    )

    one = ConstantExpr(
        name="ONE",
        value=1,
        type_hint=int,
    )

    expressions = generate_binary_applications(
        registry,
        [nested_expression],
        [one],
    )

    expected_expression = ApplyExpr(
            function=PrimitiveRefExpr("add"),
            arguments=(nested_expression, one)
    )

    assert expected_expression in expressions
    assert infer_expression_type(expected_expression, registry) == (
        registry["primitives"]["add"]["type_hints"]["return"]
    )


def test_generate_expressions_returns_compatible_expressions():
    from search.expression_generator import generate_expressions
    from search.expression_validator import infer_expression_type
    from search.type_system import is_compatible

    registry = load_registry()

    expressions = generate_expressions(
        registry,
        expected_type=int,
        max_depth=1,
    )

    assert expressions

    for expression in expressions:
        actual_type = infer_expression_type(expression, registry)
        assert is_compatible(int, actual_type)


def test_generate_expressions_uses_constants():
    from search.expression_generator import generate_expressions
    from search.expressions import ApplyExpr, ConstantExpr, PrimitiveRefExpr

    registry = load_registry()

    expressions = generate_expressions(
        registry,
        expected_type=int,
        max_depth=1,
    )

    one = ConstantExpr(name="ONE", value=1, type_hint=int)
    two = ConstantExpr(name="TWO", value=2, type_hint=int)

    expected_expression = ApplyExpr(
            function=PrimitiveRefExpr("add"),
            arguments=(one, two)
    )

    assert expected_expression in expressions


def test_binary_generation_can_filter_by_return_type():
    from search.expression_generator import get_primitives_returning_type

    registry = load_registry()

    integer_primitives = get_primitives_returning_type(
        registry,
        int,
    )

    boolean_primitives = get_primitives_returning_type(
        registry,
        bool,
    )

    assert "color" in integer_primitives
    assert "greater" in boolean_primitives
    assert "astuple" not in integer_primitives
    assert "hconcat" not in integer_primitives

"""
def test_generate_expressions_uses_intermediate_expressions_for_binary_primitives():
    from search.expression_generator import generate_expressions
    from search.expressions import ApplyExpr, InputExpr, PrimitiveRefExpr
    from search.expression_validator import infer_expression_type
    from arc_types import Grid

    registry = load_registry()

    expressions = generate_expressions(
        registry,
        expected_type=Grid,
        max_depth=2,
    )

    expected_expression = ApplyExpr(
        function=ApplyExpr(
            function=PrimitiveRefExpr("paint"),
            arguments=(ApplyExpr(
                function=PrimitiveRefExpr("identity"),
                arguments=(InputExpr(), ), ),
            ),
        ),
        arguments=(ApplyExpr(
            function=PrimitiveRefExpr("asobject"),
            arguments=(InputExpr(), ), ),
        ),
    )

    assert expected_expression in expressions
    assert infer_expression_type(expected_expression, registry) == Grid
"""

def test_generate_binary_applications_combines_grid_and_object(registry):
    from search.expression_generator import generate_binary_applications
    from search.expressions import ApplyExpr, InputExpr, PrimitiveRefExpr

    grid_expression = ApplyExpr(
        function=PrimitiveRefExpr("identity"),
        arguments=(InputExpr(), ),
    )
    object_expression = ApplyExpr(
        function=PrimitiveRefExpr("asobject"),
        arguments=(InputExpr(), ),
    )

    expressions = generate_binary_applications(
        registry,
        [grid_expression],
        [object_expression],
    )

    expected = ApplyExpr(
            function=PrimitiveRefExpr("paint"),
            arguments=(grid_expression, object_expression)
    )

    assert expected in expressions
