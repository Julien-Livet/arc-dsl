from arc_types import Grid
from search.expressions import InputExpr, ConstantExpr, PrimitiveExpr, PrimitiveRefExpr, ApplyExpr


def test_input_expression():
    expression = InputExpr()

    assert expression.name == "I"


def test_constant_expression():
    expression = ConstantExpr("ZERO", 0, int)

    assert expression.name == "ZERO"
    assert expression.value == 0
    assert expression.type_hint is int


def test_primitive_expression():
    input_expr = InputExpr()
    rotation = PrimitiveExpr("rot90", (input_expr,), Grid)

    assert rotation.name == "rot90"
    assert rotation.arguments == (input_expr,)
    assert rotation.return_type == Grid


def test_apply_expression_stores_function_and_argument():
    function = PrimitiveRefExpr("rot90")
    argument = InputExpr()

    expression = ApplyExpr(
        function=function,
        arguments=(argument, ),
    )

    assert expression.function == function
    assert expression.arguments == (argument, )
