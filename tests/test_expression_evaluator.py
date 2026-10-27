import dsl
import pytest
from arc_types import Grid
from search.expressions import (
    InputExpr,
    ConstantExpr,
    PrimitiveExpr,
    PrimitiveRefExpr,
    ApplyExpr,
)
from search.expression_evaluator import evaluate


def test_evaluate_rotation():
    grid = (
        (1, 2),
        (3, 4),
    )

    expression = PrimitiveExpr(
        "rot90",
        (InputExpr(),),
        Grid,
    )

    result = evaluate(
        expression,
        grid,
        {"rot90": dsl.rot90},
    )

    assert result == dsl.rot90(grid)


def test_evaluate_nested_expression():
    grid = (
        (1, 2),
        (3, 4),
    )

    rotation = PrimitiveExpr(
        "rot90",
        (InputExpr(),),
        Grid,
    )

    expression = PrimitiveExpr(
        "replace",
        (
            rotation,
            ConstantExpr("ZERO", 0, int),
            ConstantExpr("ONE", 1, int),
        ),
        Grid,
    )

    result = evaluate(
        expression,
        grid,
        {
            "rot90": dsl.rot90,
            "replace": dsl.replace,
        },
    )

    assert result == dsl.replace(dsl.rot90(grid), 0, 1)


def test_evaluate_primitive_reference():
    from dsl import rot90
    from search.dsl_registry import load_registry

    registry = load_registry()
    expression = PrimitiveRefExpr("rot90")

    result = evaluate(
        expression,
        ((1, 2), (3, 4)),
        {
            name: info["function"]
            for name, info in registry["primitives"].items()
        },
    )

    assert result is rot90


def test_evaluate_composition_of_primitive_references():
    from dsl import compose, rot90, rot180
    from search.dsl_registry import load_registry
    from search.expressions import (
        InputExpr,
        PrimitiveExpr,
        PrimitiveRefExpr,
    )
    from search.expression_evaluator import evaluate

    registry = load_registry()
    primitives = {
        name: info["function"]
        for name, info in registry["primitives"].items()
    }

    expression = PrimitiveExpr(
        "compose",
        (
            PrimitiveRefExpr("rot90"),
            PrimitiveRefExpr("rot180"),
        ),
        None,
    )

    composed_function = evaluate(expression, ((1, 2), (3, 4)), primitives)

    assert callable(composed_function)

    expected = compose(rot90, rot180)(((1, 2), (3, 4)))

    assert composed_function(((1, 2), (3, 4))) == expected


def test_evaluate_apply_primitive_to_input():
    from search.dsl_registry import load_registry

    registry = load_registry()
    primitives = {
        name: info["function"]
        for name, info in registry["primitives"].items()
    }

    expression = ApplyExpr(
        function=PrimitiveRefExpr("rot90"),
        arguments=(InputExpr(), ),
    )

    input_grid = ((1, 2), (3, 4))
    result = evaluate(expression, input_grid, primitives)

    assert result == primitives["rot90"](input_grid)


def test_evaluate_apply_rejects_non_function():
    expression = ApplyExpr(
        function=ConstantExpr("ZERO", 0, int),
        arguments=(InputExpr(), ),
    )

    with pytest.raises(TypeError, match="n'est pas une fonction"):
        evaluate(expression, ((1, 2),), {})


def test_evaluate_partial_application_of_binary_primitive():
    from search.dsl_registry import load_registry

    registry = load_registry()

    from arc_types import Integer
    from search.expressions import ApplyExpr, ConstantExpr, PrimitiveRefExpr
    from search.expression_evaluator import evaluate

    argument_a = ConstantExpr(
        name="ZERO",
        value=0,
        type_hint=Integer,
    )
    argument_b = ConstantExpr(
        name="ONE",
        value=1,
        type_hint=Integer,
    )

    partial_application = ApplyExpr(
        function=PrimitiveRefExpr("add"),
        arguments=(argument_a, ),
    )
    expression = ApplyExpr(
        function=partial_application,
        arguments=(argument_b, ),
    )

    primitives = {
        name: info["function"]
        for name, info in registry["primitives"].items()
    }

    result = evaluate(expression, None, primitives)

    assert result == 1
