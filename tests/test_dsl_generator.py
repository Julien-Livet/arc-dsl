import pytest

from arc_types import Grid
from search.dsl_generator import (
    APPLY_NAME,
    GenerationState,
    get_current_goal,
    get_next_symbols,
    select_symbol,
)
from search.dsl_registry import load_registry
from search.expression_validator import infer_expression_type
from search.expressions import PrimitiveRefExpr, expression_to_str
from search.type_system import is_compatible


@pytest.fixture(scope="module")
def registry():
    return load_registry()


def ref(name):
    """Désigne une primitive passée comme valeur, sans l'appeler."""
    return ("ref", name)


def select(state, item, registry):
    """Choisit le prochain symbole parmi les candidats du générateur."""
    symbols = get_next_symbols(
        expected_type=get_current_goal(state),
        registry=registry,
        depth=len(state.stack),
        max_depth=state.max_depth,
    )

    primitives = {symbol for symbol in symbols if isinstance(symbol, str)}
    named = {
        symbol.name: symbol
        for symbol in symbols
        if getattr(symbol, "name", None)
    }

    if isinstance(item, tuple):
        symbol = named[item[1]]
        assert isinstance(symbol, PrimitiveRefExpr)
    elif item in primitives:
        symbol = item
    else:
        symbol = named[item]

    return select_symbol(state, symbol, registry)


def generate(items, registry, max_depth=4):
    state = GenerationState(expected_type=Grid, max_depth=max_depth)

    for item in items:
        assert not state.finished, expression_to_str(state.result)
        state = select(state, item, registry)

    return state


def test_function_valued_branch_is_applied_at_root(registry):
    # Tâche 7b7f7511 : branch(portrait(I), tophalf, lefthalf)(I).
    state = generate(
        ["branch", "portrait", "I", ref("tophalf"), ref("lefthalf")],
        registry,
    )

    # branch(...) est une fonction Grid -> Grid : la génération continue
    # et attend l'argument de cette fonction.
    assert not state.finished
    assert state.stack[-1].name == APPLY_NAME
    assert get_current_goal(state) == Grid

    state = select(state, "I", registry)

    assert state.finished
    assert (
        expression_to_str(state.result)
        == "branch(portrait(I), tophalf, lefthalf)(I)"
    )
    assert is_compatible(infer_expression_type(state.result, registry), Grid)


def test_function_is_kept_as_a_value_where_any_is_expected(registry):
    state = generate(["branch", "portrait", "I", ref("tophalf")], registry)

    pending = state.stack[-1]

    assert pending.name == "branch"
    assert pending.arguments[1] == PrimitiveRefExpr("tophalf")


def test_function_is_applied_in_a_value_slot(registry):
    state = generate(
        ["upscale", "compose", ref("hmirror"), ref("vmirror"), "I", "TWO"],
        registry,
    )

    assert state.finished
    assert (
        expression_to_str(state.result)
        == "upscale(compose(hmirror, vmirror)(I), TWO)"
    )


def test_function_reference_stays_a_value_in_a_callable_slot(registry):
    state = generate(
        ["fill", "I", "ONE", "mapply", ref("neighbors"), "ofcolor", "I", "FIVE"],
        registry,
    )

    assert state.finished
    assert (
        expression_to_str(state.result)
        == "fill(I, ONE, mapply(neighbors, ofcolor(I, FIVE)))"
    )


def test_applied_function_must_return_the_expected_type(registry):
    # branch(portrait(I), height, width)(I) est un entier, pas une grille.
    state = generate(
        ["branch", "portrait", "I", ref("height"), ref("width")],
        registry,
    )

    with pytest.raises(TypeError):
        select(state, "I", registry)


def test_finished_expression_is_never_a_function(registry):
    state = generate(["compose", ref("hmirror"), ref("vmirror")], registry)

    assert not state.finished

    state = select(state, "I", registry)

    assert state.finished
    assert expression_to_str(state.result) == "compose(hmirror, vmirror)(I)"
