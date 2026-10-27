from arc_types import Grid, IntegerTuple, Cell, Object, Objects, Indices, Patch
from search.type_system import is_compatible
from typing import Any, Container, FrozenSet, Tuple, Union

def test_identical_types_are_compatible():
    assert is_compatible(int, int)
    assert is_compatible(bool, bool)


def test_bool_is_not_integer_for_arc_types():
    assert not is_compatible(bool, int)


def test_any_is_compatible_with_everything():
    assert is_compatible(Any, int)
    assert is_compatible(int, Any)


def test_expected_union_accepts_either_branch():
    expected = Union[int, bool]

    assert is_compatible(int, expected)
    assert is_compatible(bool, expected)
    assert not is_compatible(str, expected)


def test_actual_union_requires_all_branches_to_match():
    actual = Union[int, bool]

    assert is_compatible(actual, Union[int, bool])
    assert not is_compatible(actual, int)


def test_tuple_element_types_must_match():
    assert is_compatible(Tuple[int, int], Tuple[int, int])
    assert not is_compatible(Tuple[int, int], Tuple[str, str])


def test_grid_type():
    assert is_compatible(Grid, Grid)
    assert not is_compatible(Tuple[Tuple[str]], Grid)


def test_frozenset_element_types_must_match():
    assert is_compatible(FrozenSet[int], FrozenSet[int])
    assert not is_compatible(FrozenSet[str], FrozenSet[int])


def test_arc_aliases():
    assert is_compatible(IntegerTuple, IntegerTuple)
    assert is_compatible(Cell, Cell)
    assert is_compatible(Object, Object)
    assert is_compatible(Objects, Objects)
    assert is_compatible(Indices, Indices)


def test_patch_union():
    assert is_compatible(Object, Patch)
    assert is_compatible(Indices, Patch)
    assert not is_compatible(Grid, Patch)


def test_frozenset_of_frozensets_is_compatible_with_nested_container():
    actual = FrozenSet[FrozenSet[Tuple[int, int]]]
    expected = Container[Container]

    assert is_compatible(actual, expected)
