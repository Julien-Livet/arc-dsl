from typing import (
    List,
    Union,
    Tuple,
    Any,
    Container,
    Callable,
    FrozenSet,
    Iterable
)

Boolean = bool
Integer = int
IntegerTuple = Tuple[Integer, Integer]
Numerical = Union[Integer, IntegerTuple]
IntegerSet = Union[FrozenSet[Integer], Tuple[Integer, ...]]
Grid = Tuple[Tuple[Integer, ...], ...]
Cell = Tuple[Integer, IntegerTuple]
Object = Union[FrozenSet[Cell], Tuple[Cell, ...]]
Objects = FrozenSet[Object]
Indices = Union[FrozenSet[IntegerTuple], Tuple[IntegerTuple, ...]]
IndicesSet = FrozenSet[Indices]
Patch = Union[Object, Indices, Tuple[Tuple[int ,int], ...]]
Element = Union[Object, Grid]
Piece = Union[Grid, Patch]
TupleTuple = Tuple[Tuple]
ContainerContainer = Container[Container]


# Signatures génériques : variables de type des primitives de conteneur
# (ItemT, CollT, ...). Elles sont définies dans type_variables.py pour que
# leurs objets restent les mêmes si arc_types.py est réexécuté, comme le fait
# dsl_model/utils.py (load_module).
from type_variables import (
    ItemT,
    OtherItemT,
    ContainerVar,
    ContainerOf,
    CollT,
    OtherCollT,
    InnerCollT,
)
