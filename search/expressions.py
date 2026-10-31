from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InputExpr:
    """Référence à la grille d'entrée."""
    name: str = "I"


@dataclass(frozen=True)
class ConstantExpr:
    """Référence à une constante du DSL."""
    name: str
    value: Any
    type_hint: Any


@dataclass(frozen=True)
class PrimitiveExpr:
    """Application d'une primitive à des expressions."""
    name: str
    arguments: tuple
    return_type: Any


@dataclass(frozen=True)
class PrimitiveRefExpr:
    """Référence à une primitive sans l'appliquer."""
    name: str


@dataclass(frozen=True)
class ApplyExpr:
    """Application d'une fonction à un argument."""
    function: Any
    arguments: tuple


def expression_to_str(expression):
    """Convertit une expression DSL en chaîne Python."""

    if isinstance(expression, InputExpr):
        return expression.name

    if isinstance(expression, ConstantExpr):
        return expression.name

    if isinstance(expression, PrimitiveRefExpr):
        return expression.name

    if isinstance(expression, PrimitiveExpr):
        arguments = ", ".join(
            expression_to_str(argument)
            for argument in expression.arguments
        )
        return f"{expression.name}({arguments})"

    if isinstance(expression, ApplyExpr):
        function = expression_to_str(expression.function)
        arguments = ", ".join(
            expression_to_str(argument)
            for argument in expression.arguments
        )
        return f"{function}({arguments})"

    raise TypeError(
        f"Type d'expression non pris en charge : {type(expression)}"
    )


def evaluate_expression(expression, input_grid, registry):
    source = expression_to_str(expression)

    environment = {
        "I": input_grid,
    }

    environment.update({
        name: info["value"]
        for name, info in registry["constants"].items()
    })

    environment.update({
        name: info["function"]
        for name, info in registry["primitives"].items()
    })

    return eval(source, {"__builtins__": {}}, environment)

