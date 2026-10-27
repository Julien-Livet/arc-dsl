import pytest

from arc_types import Grid
from search.dsl_registry import load_registry
from search.expressions import (
    InputExpr,
    ConstantExpr,
    PrimitiveExpr,
    PrimitiveRefExpr,
    ApplyExpr
)
from search.expression_validator import (
    infer_expression_type,
    validate_expression,
)
from search.type_system import is_compatible


@pytest.fixture
def registry():
    return load_registry()


def test_valid_rotation(registry):
    expression = PrimitiveExpr("rot90", (InputExpr(),), Grid)

    assert infer_expression_type(expression, registry) == Grid
    assert validate_expression(expression, registry)


def test_valid_nested_expression(registry):
    rotation = PrimitiveExpr("rot90", (InputExpr(),), Grid)

    expression = PrimitiveExpr(
        "replace",
        (
            rotation,
            ConstantExpr("ZERO", 0, int),
            ConstantExpr("ONE", 1, int),
        ),
        Grid,
    )

    assert validate_expression(expression, registry)


def test_rejects_wrong_argument_type(registry):
    expression = PrimitiveExpr(
        "rot90",
        (ConstantExpr("ZERO", 0, int),),
        Grid,
    )

    with pytest.raises(TypeError):
        validate_expression(expression, registry)


def test_rejects_wrong_number_of_arguments(registry):
    expression = PrimitiveExpr("rot90", (), Grid)

    with pytest.raises(ValueError):
        validate_expression(expression, registry)


def test_rejects_unknown_constant(registry):
    expression = ConstantExpr("UNKNOWN", 0, int)

    with pytest.raises(ValueError):
        validate_expression(expression, registry)


def test_compose_rejects_non_function_arguments(registry):
    expression = PrimitiveExpr(
        "compose",
        (
            PrimitiveExpr("rot90", (InputExpr(),), Grid),
            PrimitiveExpr("rot180", (InputExpr(),), Grid),
        ),
        Grid,
    )

    with pytest.raises(TypeError, match="signature connue"):
        infer_expression_type(expression, registry)


def test_primitive_reference_has_function_type(registry):
    from search.function_types import FunctionType

    expression = PrimitiveRefExpr("rot90")

    assert infer_expression_type(expression, registry) == FunctionType(
        (Grid,),
        Grid,
    )


def test_binary_primitive_reference_has_two_parameters(registry):
    from search.function_types import FunctionType
    from search.type_system import get_parameters, get_return_type

    expression = PrimitiveRefExpr("add")
    actual_type = infer_expression_type(expression, registry)

    primitive_info = registry["primitives"]["add"]
    parameters = get_parameters(primitive_info)
    expected_type = FunctionType(
        tuple(parameter["type"] for parameter in parameters),
        get_return_type(primitive_info),
    )

    assert actual_type == expected_type


def test_unknown_primitive_reference_is_rejected(registry):
    expression = PrimitiveRefExpr("unknown_primitive")

    with pytest.raises(ValueError, match="Primitive inconnue"):
        infer_expression_type(expression, registry)


def test_compose_of_primitive_references_has_function_type(registry):
    from search.function_types import FunctionType

    expression = PrimitiveExpr(
        "compose",
        (
            PrimitiveRefExpr("rot90"),
            PrimitiveRefExpr("rot180"),
        ),
        FunctionType((Grid,), Grid),
    )

    inferred_type = infer_expression_type(expression, registry)

    assert inferred_type == FunctionType((Grid,), Grid)


def test_apply_primitive_to_input_has_grid_type(registry):
    expression = ApplyExpr(
        function=PrimitiveRefExpr("rot90"),
        arguments=(InputExpr(), ),
    )

    assert infer_expression_type(expression, registry) == Grid
    assert validate_expression(expression, registry)


def test_apply_rejects_non_function(registry):
    expression = ApplyExpr(
        function=ConstantExpr("ZERO", 0, int),
        arguments=(InputExpr(), ),
    )

    with pytest.raises(TypeError, match="signature de fonction"):
        infer_expression_type(expression, registry)


def test_composed_function_application_has_grid_type(registry):
    expression = ApplyExpr(
        function=PrimitiveExpr(
            "compose",
            (
                PrimitiveRefExpr("rot90"),
                PrimitiveRefExpr("rot180"),
            ),
            None,
        ),
        arguments=(InputExpr(), ),
    )

    assert infer_expression_type(expression, registry) == Grid
    assert validate_expression(expression, registry)


def test_identity_application_preserves_argument_type(registry):
    expression = ApplyExpr(
        function=PrimitiveRefExpr("identity"),
        arguments=(InputExpr(), ),
    )

    assert is_compatible(infer_expression_type(expression, registry), Grid)
    assert validate_expression(expression, registry)


def test_binary_primitive_application_preserves_declared_return_type(registry):
    from search.expressions import ConstantExpr, PrimitiveExpr
    from search.type_system import get_return_type

    expression = PrimitiveExpr(
        "add",
        (
            ConstantExpr("ONE", 1, int),
            ConstantExpr("TWO", 2, int),
        ),
        None,
    )

    actual_type = infer_expression_type(expression, registry)
    expected_type = get_return_type(registry["primitives"]["add"])

    assert is_compatible(actual_type, expected_type)


def test_partial_application_of_binary_primitive_returns_function_type(registry):
    from arc_types import Integer
    from search.expressions import ApplyExpr, ConstantExpr, PrimitiveRefExpr
    from search.expression_validator import infer_expression_type
    from search.function_types import FunctionType

    argument = ConstantExpr(
        name="ZERO",
        value=0,
        type_hint=Integer,
    )
    expression = ApplyExpr(
        function=PrimitiveRefExpr("add"),
        arguments=(argument, ),
    )

    actual_type = infer_expression_type(expression, registry)

    assert isinstance(actual_type, FunctionType)
    assert actual_type.arity == 1
    assert actual_type.parameter_types == (
        registry["primitives"]["add"]["type_hints"]["b"],
    )


def test_add_of_integer_constants_infers_integer_type(registry):
    from search.expressions import ApplyExpr, ConstantExpr, PrimitiveRefExpr

    expression = ApplyExpr(
            function=PrimitiveRefExpr("add"),
            arguments=(ConstantExpr("ONE", 1, int), ConstantExpr("TWO", 2, int)),
    )

    assert is_compatible(int, infer_expression_type(expression, registry))
