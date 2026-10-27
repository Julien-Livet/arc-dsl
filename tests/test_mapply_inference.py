from arc_types import Indices
from search.dsl_registry import load_registry
from search.expressions import InputExpr, ConstantExpr, PrimitiveExpr, PrimitiveRefExpr
from search.expression_validator import infer_expression_type


def test_mapply_infers_return_type_from_function():
    registry = load_registry()

    corners = PrimitiveRefExpr("corners")
    objects_info = registry["constants"]

    objects = PrimitiveExpr(
        "objects",
        (
            InputExpr(),
            ConstantExpr("T", objects_info["T"]["value"], objects_info["T"]["type"]),
            ConstantExpr("F", objects_info["F"]["value"], objects_info["F"]["type"]),
            ConstantExpr("T", objects_info["T"]["value"], objects_info["T"]["type"]),
        ),
        None,
    )

    expression = PrimitiveExpr("mapply", (corners, objects), None)

    assert infer_expression_type(expression, registry) == Indices
