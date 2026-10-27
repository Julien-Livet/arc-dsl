from arc_types import Grid
from search.dsl_registry import load_registry
from search.type_system import get_parameters


def test_replace_parameters():
    registry = load_registry()
    parameters = get_parameters(registry["primitives"]["replace"])

    assert [p["name"] for p in parameters] == [
        "grid",
        "replacee",
        "replacer",
    ]

    assert [p["type"] for p in parameters] == [
        Grid,
        int,
        int,
    ]

    assert all(p["required"] for p in parameters)


def test_rot90_parameters():
    registry = load_registry()
    parameters = get_parameters(registry["primitives"]["rot90"])

    assert len(parameters) == 1
    assert parameters[0]["name"] == "grid"
    assert parameters[0]["type"] == Grid
    assert parameters[0]["required"]
