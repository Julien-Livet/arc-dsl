from search.expressions import (
    InputExpr,
    ConstantExpr,
    PrimitiveExpr,
    PrimitiveRefExpr,
    ApplyExpr,
)
from functools import partial


def evaluate(expression, input_grid, primitives):
    """Évalue récursivement un arbre d'expression du DSL."""

    if isinstance(expression, InputExpr):
        if expression.name != "I":
            raise ValueError(f"Entrée inconnue : {expression.name}")
        return input_grid

    if isinstance(expression, ConstantExpr):
        return expression.value

    if isinstance(expression, PrimitiveExpr):
        function = primitives.get(expression.name)
        if function is None:
            raise ValueError(f"Primitive inconnue : {expression.name}")

        arguments = [
            evaluate(argument, input_grid, primitives)
            for argument in expression.arguments
        ]

        return function(*arguments)

    if isinstance(expression, PrimitiveRefExpr):
        function = primitives.get(expression.name)
        if function is None:
            raise ValueError(f"Primitive inconnue : {expression.name}")
        return function

    if isinstance(expression, ApplyExpr):
        function = evaluate(expression.function, input_grid, primitives)
        argument = evaluate(expression.arguments[0], input_grid, primitives)

        if not callable(function):
            raise TypeError("L'expression à appliquer n'est pas une fonction.")

        try:
            from search.expression_validator import infer_expression_type
            from search.dsl_registry import load_registry
            from search.function_types import FunctionType

            registry = load_registry()
            function_type = infer_expression_type(expression.function, registry)

            if isinstance(function_type, FunctionType) and function_type.arity > 1:
                return partial(function, argument)

        except (TypeError, ValueError):
            pass

        return function(argument)

    raise TypeError(
        f"Type d'expression non pris en charge : {type(expression).__name__}"
    )
