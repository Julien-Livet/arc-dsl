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
