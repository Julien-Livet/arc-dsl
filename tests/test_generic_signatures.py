import copy
import pickle
from typing import Any, Callable, Container, FrozenSet, Tuple, Union

import pytest

import dsl
from arc_types import (
    Cell,
    CollT,
    ContainerOf,
    Grid,
    Indices,
    IntegerTuple,
    ItemT,
    Object,
    Objects,
)
from search.dsl_generator import (
    GenerationState,
    get_current_goal,
    get_next_symbols,
    select_symbol,
)
from search.dsl_registry import load_registry
from search.expression_validator import infer_expression_type
from search.function_types import FunctionType
from search.generics import (
    GenericCall,
    erase,
    infer_generic_call,
    merge_item_types,
    substitute,
    unify,
)
from search.python_to_dsl import parse_solver


# Annotations des primitives avant qu'elles ne deviennent génériques. La
# signature effacée doit leur rester identique : le reste du code
# (références de fonctions, combinateurs, catalogue) n'y voit aucune
# différence.
PREVIOUS_ANNOTATIONS = {
    "argmax": ((Container, Callable), Any),
    "argmin": ((Container, Callable), Any),
    "astuple": ((Any, Any), Tuple[Any, Any]),
    "combine": ((Container, Container), Container),
    "contained": ((Any, Container), bool),
    "dedupe": ((Tuple,), Tuple),
    "difference": ((FrozenSet, FrozenSet), FrozenSet),
    "extract": ((Container, Callable), Any),
    "first": ((Container,), Any),
    "initset": ((Any,), FrozenSet),
    "insert": ((Any, FrozenSet), FrozenSet),
    "intersection": ((FrozenSet, FrozenSet), FrozenSet),
    "last": ((Container,), Any),
    "leastcommon": ((Container,), Any),
    "merge": ((Container[Container],), Container),
    "mostcommon": ((Container,), Any),
    "order": ((Container, Callable), Tuple),
    "other": ((Container, Any), Any),
    "remove": ((Any, Container), Container),
    "repeat": ((Any, int), Tuple),
    "sfilter": ((Container, Callable), Container),
    "sizefilter": ((Container, int), FrozenSet),
    "totuple": ((FrozenSet,), Tuple),
}


@pytest.fixture(scope="module")
def registry():
    return load_registry()


def infer(body, registry):
    source = "def solve_test(I):\n" + "".join(
        f"    {line}\n" for line in body
    )

    return infer_expression_type(parse_solver(source, registry), registry)


def infer_value(expression, registry):
    return infer(["O = " + expression, "return O"], registry)


# Signatures

def test_erased_signatures_are_the_previous_annotations(registry):
    for name, (parameter_types, return_type) in PREVIOUS_ANNOTATIONS.items():
        info = registry["primitives"][name]

        assert info["generic_signature"] is not None, name
        assert tuple(
            info["type_hints"][parameter]
            for parameter in info["signature"].parameters
        ) == parameter_types, name
        assert info["type_hints"]["return"] == return_type, name


def test_generic_signature_of_remove(registry):
    signature = registry["primitives"]["remove"]["generic_signature"]

    assert signature.parameter_types == (ItemT, CollT[ItemT])
    assert signature.return_type == CollT[ItemT]


def test_other_primitives_have_no_generic_signature(registry):
    generic = {
        name
        for name, info in registry["primitives"].items()
        if info["generic_signature"] is not None
    }

    assert generic == set(PREVIOUS_ANNOTATIONS)


def test_container_variables_survive_copy_and_pickle():
    pattern = CollT[ItemT]

    assert copy.deepcopy(pattern).kind is CollT
    assert pickle.loads(pickle.dumps(CollT)) is CollT
    assert isinstance(pattern, ContainerOf)


# Unification

def test_unify_binds_the_kind_and_the_items():
    bindings = unify(CollT[ItemT], FrozenSet[int], {})

    assert bindings == {CollT: frozenset({frozenset}), ItemT: int}
    assert substitute(CollT[ItemT], bindings) == FrozenSet[int]


def test_a_grid_is_a_tuple_of_rows():
    bindings = unify(CollT[ItemT], Grid, {})

    assert bindings[ItemT] == Tuple[int, ...]
    assert substitute(CollT[ItemT], bindings) == Grid


def test_a_frozenset_pattern_rejects_a_tuple():
    assert unify(FrozenSet[ItemT], Tuple[int, ...], {}) is None


def test_item_types_must_overlap():
    assert merge_item_types(int, FrozenSet[int]) is None
    assert merge_item_types(Any, Cell) == Cell
    assert merge_item_types(Object, FrozenSet[Cell]) == Object


def test_without_bound_variable_substitute_is_the_erased_type():
    assert substitute(CollT[ItemT], {}) == erase(CollT[ItemT]) == Container


def test_unknown_kind_becomes_an_explicit_union():
    assert substitute(CollT[ItemT], {ItemT: int}) == Union[
        FrozenSet[int],
        Tuple[int, ...],
    ]


def test_incompatible_call_raises(registry):
    signature = registry["primitives"]["remove"]["generic_signature"]

    with pytest.raises(TypeError, match="remove"):
        infer_generic_call("remove", signature, [int, Objects])


# Inférence des appels

def test_remove_keeps_the_container_type(registry):
    assert infer_value(
        "remove(first(objects(I, T, T, T)), objects(I, T, T, T))",
        registry,
    ) == Objects


def test_remove_rejects_a_value_of_another_type(registry):
    # Un entier ne peut pas être un des objets : remove ne retirerait rien.
    with pytest.raises(TypeError, match="remove"):
        infer_value("remove(ZERO, objects(I, T, T, T))", registry)

    with pytest.raises(TypeError, match="remove"):
        infer_value("remove(ZERO, I)", registry)


def test_remove_preserves_each_branch_of_a_union(registry):
    # asobject retourne un Object : frozenset ou tuple de cellules.
    assert infer_value("remove(first(asobject(I)), asobject(I))", registry) == Object


def test_insert_needs_a_frozenset(registry):
    # insert utilise container.union : un tuple n'en a pas.
    with pytest.raises(AttributeError):
        dsl.insert(0, (1, 2))

    with pytest.raises(TypeError, match="insert"):
        infer_value("insert(ZERO, interval(ZERO, TEN, ONE))", registry)

    assert infer_value("insert(ZERO, palette(I))", registry) == FrozenSet[int]


def test_insert_into_a_set_of_functions(registry):
    # Tâche 6aa20dc0 : un ensemble de transformations de même arité.
    result = infer(
        [
            "x1 = initset(identity)",
            "x2 = insert(vmirror, x1)",
            "O = insert(hmirror, x2)",
            "return O",
        ],
        registry,
    )

    (function_type,) = result.__args__

    assert isinstance(function_type, FunctionType)
    assert function_type.arity == 1


def test_first_and_merge_follow_the_items(registry):
    assert infer_value("first(objects(I, T, T, T))", registry) == Object
    assert infer_value("first(asobject(I))", registry) == Cell
    assert infer_value("merge(objects(I, T, T, T))", registry) == FrozenSet[Cell]

    # merge attend un conteneur de conteneurs.
    with pytest.raises(TypeError, match="merge"):
        infer_value("merge(palette(I))", registry)


def test_combine_keeps_the_kind_of_its_first_argument(registry):
    assert infer_value(
        "combine(interval(ZERO, TEN, ONE), palette(I))",
        registry,
    ) == Tuple[int, ...]

    # Des objets et des positions ne se mélangent pas.
    with pytest.raises(TypeError, match="combine"):
        infer_value("combine(objects(I, T, T, T), ofcolor(I, ZERO))", registry)


def test_contained_checks_the_value(registry):
    assert infer_value("contained(ZERO, palette(I))", registry) is bool

    with pytest.raises(TypeError, match="contained"):
        infer_value("contained(ZERO, objects(I, T, T, T))", registry)


def test_collections_of_known_items(registry):
    assert infer_value("order(objects(I, T, T, T), size)", registry) == Tuple[Object, ...]
    assert infer_value("argmax(objects(I, T, T, T), size)", registry) == Object
    assert infer_value("sizefilter(objects(I, T, T, T), TWO)", registry) == Objects
    assert infer_value("totuple(palette(I))", registry) == Tuple[int, ...]
    assert infer_value("astuple(ONE, UNITY)", registry) == Tuple[int, IntegerTuple]
    assert infer_value(
        "difference(asindices(I), ofcolor(I, ZERO))",
        registry,
    ) == FrozenSet[IntegerTuple]


def test_plain_arguments_are_still_checked(registry):
    with pytest.raises(TypeError, match="sfilter"):
        infer_value("sfilter(palette(I), ZERO)", registry)


# Génération

def generate_step(state, symbol, registry):
    """Choisit un symbole parmi les candidats ; un nom de primitive est un appel."""
    candidates = get_next_symbols(
        expected_type=get_current_goal(state),
        registry=registry,
        depth=len(state.stack),
        max_depth=state.max_depth,
    )
    primitives = {candidate for candidate in candidates if isinstance(candidate, str)}
    named = {
        candidate.name: candidate
        for candidate in candidates
        if getattr(candidate, "name", None)
    }

    return select_symbol(
        state,
        symbol if symbol in primitives else named.get(symbol, symbol),
        registry,
    )


def generate(symbols, registry, expected_type=Grid):
    state = GenerationState(expected_type=expected_type, max_depth=4)

    for symbol in symbols:
        state = generate_step(state, symbol, registry)

    return state


def candidate_names(state, registry):
    return {
        getattr(symbol, "name", symbol)
        for symbol in get_next_symbols(
            expected_type=get_current_goal(state),
            registry=registry,
            depth=len(state.stack),
            max_depth=state.max_depth,
        )
    }


def test_remove_slot_follows_the_value(registry):
    state = generate(["fill", "I", "ZERO", "remove", "UNITY"], registry)

    # Le conteneur doit contenir des positions.
    assert get_current_goal(state) == Union[
        FrozenSet[IntegerTuple],
        Tuple[IntegerTuple, ...],
    ]

    names = candidate_names(state, registry)

    assert "ofcolor" in names
    assert "palette" not in names
    assert "objects" not in names


def test_remove_value_follows_the_expected_result(registry):
    state = generate(["fill", "I", "ZERO", "remove"], registry)

    # Le résultat doit être un Patch : la valeur est une cellule ou une
    # position, pas une grille.
    assert get_current_goal(state) == Union[Cell, IntegerTuple]

    names = candidate_names(state, registry)

    assert "UNITY" in names
    assert "I" not in names


def test_first_slot_follows_the_expected_result(registry):
    # canvas attend une couleur : first doit lire un conteneur d'entiers.
    state = generate(["canvas", "first"], registry)

    assert get_current_goal(state) == Union[FrozenSet[int], Tuple[int, ...]]

    names = candidate_names(state, registry)

    assert "palette" in names
    assert "objects" not in names


def test_insert_slot_requires_a_frozenset(registry):
    state = generate(["fill", "I", "ZERO", "insert", "UNITY"], registry)

    assert get_current_goal(state) == FrozenSet[IntegerTuple]
    assert "interval" not in candidate_names(state, registry)


def test_incompatible_argument_is_rejected(registry):
    # objects(...) n'est pas un conteneur d'entiers.
    state = generate(["remove", "ZERO", "objects", "I", "T", "T"], registry, expected_type=Any)

    with pytest.raises(TypeError, match="remove"):
        generate_step(state, "T", registry)



def test_generic_call_is_refined_argument_by_argument(registry):
    call = GenericCall(registry["primitives"]["remove"]["generic_signature"])

    assert call.parameter_type(1) == Container

    call = call.with_argument(0, Cell)

    assert call.parameter_type(1) == Union[FrozenSet[Cell], Tuple[Cell, ...]]
    assert call.with_argument(1, Grid) is None


def test_facts_override_the_expected_result(registry):
    call = GenericCall(registry["primitives"]["remove"]["generic_signature"])
    call = call.expecting(Union[Object, Indices])

    # Borne : des cellules ou des positions.
    assert call.parameter_type(0) == Union[Cell, IntegerTuple]

    # Fait : la valeur est une position.
    call = call.with_argument(0, IntegerTuple)

    assert call.parameter_type(1) == Union[
        FrozenSet[IntegerTuple],
        Tuple[IntegerTuple, ...],
    ]


def test_reloading_arc_types_keeps_the_same_variables(registry):
    # dsl_model/utils.py réexécute arc_types.py (load_module) et remplace le
    # module dans sys.modules. Les variables doivent rester les mêmes objets :
    # sinon dsl.py et search/generics.py, importés avant, ne reconnaissent
    # plus les motifs construits après coup.
    import importlib.util
    import sys

    previous = sys.modules["arc_types"]
    spec = importlib.util.spec_from_file_location("arc_types", previous.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["arc_types"] = module

    try:
        assert module.ItemT is ItemT
        assert module.CollT is CollT
        assert module.ContainerOf is ContainerOf

        signature = registry["primitives"]["remove"]["generic_signature"]

        assert signature.parameter_types == (module.ItemT, module.CollT[module.ItemT])
        assert unify(module.CollT[module.ItemT], FrozenSet[int], {}) == {
            CollT: frozenset({frozenset}),
            ItemT: int,
        }
    finally:
        sys.modules["arc_types"] = previous
