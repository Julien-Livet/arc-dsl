from search.dsl_registry import load_registry

def test_registry_loads_primitives_and_constants():
    registry = load_registry()

    primitives = registry["primitives"]
    constants = registry["constants"]

    assert primitives
    assert constants

    assert "rot90" in primitives
    assert "replace" in primitives

    assert "ZERO" in constants
    assert "ONE" in constants
    assert "DOWN" in constants

    assert constants["ZERO"]["value"] == 0
    assert constants["ONE"]["value"] == 1
    assert constants["DOWN"]["value"] == (1, 0)
