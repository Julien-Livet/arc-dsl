from dataclasses import dataclass as _dataclass
from typing import (
    List,
    Union,
    Tuple,
    Any,
    Container,
    Callable,
    FrozenSet,
    Iterable,
    TypeVar
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


# Signatures génériques
#
# Les variables ci-dessous relient les types des arguments et du résultat
# d'une primitive, par exemple :
#
#     def remove(value: ItemT, container: CollT[ItemT]) -> CollT[ItemT]
#
# se lit « value est du type des éléments du conteneur, et le résultat est
# un conteneur de même sorte (frozenset ou tuple) et de mêmes éléments ».
#
# Python ne permet pas d'écrire S[T] avec S = TypeVar("S", tuple, frozenset) :
# un TypeVar n'est pas indiçable. Les variables de conteneur sont donc des
# objets ContainerVar, indiçables. Les noms T et S sont évités car T est déjà
# la constante True de constants.py.

ItemT = TypeVar("ItemT")
OtherItemT = TypeVar("OtherItemT")


class ContainerVar:
    """
    Variable de sorte de conteneur dans une signature générique.

    CollT[ItemT] désigne un conteneur d'éléments ItemT dont la sorte
    (frozenset ou tuple) est la même partout où CollT apparaît dans une même
    signature.
    """

    def __init__(self, name, kinds=(frozenset, tuple)):
        self.__name__ = name
        self.kinds = tuple(kinds)

    def __getitem__(self, item_type):
        return ContainerOf(self, item_type)

    def __repr__(self):
        return self.__name__

    # Une variable est désignée par son nom : copies et pickle (processus
    # de travail) retrouvent le même objet.
    def __reduce__(self):
        return self.__name__

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


@_dataclass(frozen=True)
class ContainerOf:
    """Application d'une variable de conteneur à un type d'élément."""

    kind: ContainerVar
    item: Any

    def __repr__(self):
        return f"{self.kind!r}[{self.item!r}]"


CollT = ContainerVar("CollT")
OtherCollT = ContainerVar("OtherCollT")
InnerCollT = ContainerVar("InnerCollT")
