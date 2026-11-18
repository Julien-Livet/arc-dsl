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
