"""
Signatures génériques des primitives du DSL.

Une signature générique relie les types des arguments et du résultat d'une
primitive par des variables (voir arc_types.py) :

    def remove(value: ItemT, container: CollT[ItemT]) -> CollT[ItemT]

- ItemT, OtherItemT sont des typing.TypeVar : types d'éléments ;
- CollT, OtherCollT, InnerCollT sont des ContainerVar : sortes de conteneur
  (frozenset ou tuple), identiques partout où la variable apparaît.

Ce module fournit :

- erase : la signature « effacée », sans variable, telle que la voit le
  reste du code (ItemT -> Any, CollT[ItemT] -> Container, ...). Elle est
  identique aux annotations non génériques d'avant ;
- unify : lie les variables d'un motif au type réel d'un argument ;
- substitute : remplace les variables liées dans un motif ;
- infer_generic_call : type du résultat d'un appel, ou TypeError si les
  arguments ne peuvent pas satisfaire la signature ;
- GenericCall : appel en cours de génération, dont les emplacements se
  précisent à mesure que les arguments sont connus.
"""

from collections.abc import Callable as ABCCallable
from collections.abc import Container as ABCContainer
from dataclasses import dataclass, field
from itertools import product
from typing import (
    Any,
    Callable,
    Container,
    FrozenSet,
    Tuple,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from arc_types import ContainerOf, ContainerVar
from search.type_system import _is_union, is_compatible

KINDS = (frozenset, tuple)

# Au-delà, les unions des arguments ne sont plus examinées branche par branche.
MAX_COMBINATIONS = 256


@dataclass(frozen=True)
class GenericSignature:
    """Signature d'une primitive dont certains types sont des variables."""

    parameter_types: tuple
    return_type: Any


def is_generic(type_hint):
    """Indique si un type contient une variable de type ou de conteneur."""
    if isinstance(type_hint, (TypeVar, ContainerOf)):
        return True

    return any(
        is_generic(argument)
        for argument in get_args(type_hint)
        if argument is not Ellipsis and not isinstance(argument, list)
    )


def _is_variadic_tuple(arguments):
    return len(arguments) == 2 and arguments[1] is Ellipsis


def _make_container(kind, item_type):
    """Construit FrozenSet[item] ou Tuple[item, ...] (non paramétré si Any)."""
    if kind is frozenset:
        return FrozenSet if item_type is Any else FrozenSet[item_type]

    return Tuple if item_type is Any else Tuple[item_type, ...]


def _union(types_):
    """Union de types, absorbée par Any."""
    types_ = tuple(types_)

    if not types_ or any(type_ is Any for type_ in types_):
        return Any

    unique = []

    for type_ in types_:
        if type_ not in unique:
            unique.append(type_)

    return unique[0] if len(unique) == 1 else Union[tuple(unique)]


def erase(type_hint):
    """
    Retire les variables d'un type.

    Le résultat est l'annotation que portait la primitive avant d'être
    générique : le reste du code (références de fonctions, inférence des
    combinateurs, catalogue du générateur) continue de voir ces types.
    """
    if isinstance(type_hint, TypeVar):
        return Any

    if isinstance(type_hint, ContainerOf):
        item_type = erase(type_hint.item)
        return Container if item_type is Any else Container[item_type]

    if not is_generic(type_hint):
        return type_hint

    origin = get_origin(type_hint)
    arguments = get_args(type_hint)

    if origin is frozenset:
        return _make_container(frozenset, erase(arguments[0]))

    if origin is tuple:
        if _is_variadic_tuple(arguments):
            return _make_container(tuple, erase(arguments[0]))

        return Tuple[tuple(erase(argument) for argument in arguments)]

    if _is_union(type_hint):
        return Union[tuple(erase(argument) for argument in arguments)]

    raise NotImplementedError(f"Type générique non pris en charge : {type_hint!r}")


def _variables(type_hint):
    """Variables de type et de conteneur présentes dans un type."""
    if isinstance(type_hint, TypeVar):
        return {type_hint}

    if isinstance(type_hint, ContainerOf):
        return {type_hint.kind} | _variables(type_hint.item)

    return set().union(
        *(
            _variables(argument)
            for argument in get_args(type_hint)
            if argument is not Ellipsis and not isinstance(argument, list)
        )
    )


def substitute(type_hint, bindings):
    """
    Remplace dans un type les variables liées.

    Sans aucune variable liée (ou liée à Any), le résultat est la signature
    effacée. Sinon, un conteneur de sorte inconnue devient l'union explicite
    FrozenSet[...] | Tuple[..., ...], que is_compatible sait comparer.
    """
    if not any(
        bindings.get(variable, Any) is not Any
        for variable in _variables(type_hint)
    ):
        return erase(type_hint)

    if isinstance(type_hint, TypeVar):
        return bindings.get(type_hint, Any)

    if isinstance(type_hint, ContainerOf):
        item_type = substitute(type_hint.item, bindings)
        kinds = bindings.get(type_hint.kind, frozenset(type_hint.kind.kinds))

        if item_type is Any and kinds == frozenset(KINDS):
            return Container

        return _union(
            _make_container(kind, item_type)
            for kind in KINDS
            if kind in kinds
        )

    origin = get_origin(type_hint)
    arguments = get_args(type_hint)

    if origin is frozenset:
        return _make_container(frozenset, substitute(arguments[0], bindings))

    if origin is tuple:
        if _is_variadic_tuple(arguments):
            return _make_container(tuple, substitute(arguments[0], bindings))

        return Tuple[tuple(substitute(argument, bindings) for argument in arguments)]

    if _is_union(type_hint):
        return _union(substitute(argument, bindings) for argument in arguments)

    raise NotImplementedError(f"Type générique non pris en charge : {type_hint!r}")


def container_parts(actual_type):
    """
    Décompose un type en (sorte, type des éléments) possibles.

    Retourne une liste vide si le type ne peut pas être un frozenset ou un
    tuple.
    """
    if actual_type is Any:
        return [(kind, Any) for kind in KINDS]

    if _is_union(actual_type):
        return [
            part
            for option in get_args(actual_type)
            for part in container_parts(option)
        ]

    if actual_type in (FrozenSet, frozenset):
        return [(frozenset, Any)]

    if actual_type in (Tuple, tuple):
        return [(tuple, Any)]

    origin = get_origin(actual_type)
    arguments = get_args(actual_type)

    if origin is frozenset:
        return [(frozenset, arguments[0] if arguments else Any)]

    if origin is tuple:
        if not arguments:
            return [(tuple, Any)]

        if _is_variadic_tuple(arguments):
            return [(tuple, arguments[0])]

        # Tuple de longueur fixe : ses éléments peuvent être de types variés.
        return [(tuple, _union(arguments))]

    if actual_type in (Container, ABCContainer) or origin in (Container, ABCContainer):
        item_type = arguments[0] if arguments else Any
        return [(kind, item_type) for kind in KINDS]

    return []


def merge_item_types(current, new):
    """
    Réunit deux types liés à une même variable d'élément.

    Les deux types doivent avoir au moins une valeur possible en commun ;
    le résultat est leur union. Retourne None s'ils sont incompatibles.
    """
    if current is Any:
        return new

    if new is Any or new == current:
        return current

    from search.function_types import FunctionType

    if isinstance(current, FunctionType) or isinstance(new, FunctionType):
        return _merge_function_types(current, new)

    if is_compatible(new, current) or is_compatible(current, new):
        return _union((current, new))

    return None


def _merge_function_types(current, new):
    """
    Réunit deux fonctions rangées dans un même conteneur.

    Exemple : insert(vmirror, initset(identity)). Les fonctions doivent avoir
    la même arité ; le résultat accepte et produit l'union de leurs types.
    """
    from search.function_types import FunctionType

    if current in (Callable, ABCCallable) or new in (Callable, ABCCallable):
        return Callable

    if not (isinstance(current, FunctionType) and isinstance(new, FunctionType)):
        return None

    if (
        current.variadic != new.variadic
        or len(current.parameter_types) != len(new.parameter_types)
    ):
        return None

    return FunctionType(
        parameter_types=tuple(
            _union((current_type, new_type))
            for current_type, new_type in zip(
                current.parameter_types,
                new.parameter_types,
            )
        ),
        return_type=_union((current.return_type, new.return_type)),
        variadic=current.variadic,
    )


def _bind(bindings, variable, value):
    new_bindings = dict(bindings)
    new_bindings[variable] = value
    return new_bindings


def _accepts(expected_type, actual_type):
    """Contrôle habituel d'un argument non générique."""
    from search.function_types import FunctionType

    if expected_type in (Callable, ABCCallable):
        return isinstance(actual_type, FunctionType)

    if isinstance(actual_type, FunctionType):
        return False

    return is_compatible(actual_type, expected_type)


def _unify_container(kinds, item_pattern, actual_type, bindings, variable=None):
    if variable is not None:
        kinds = bindings.get(variable, frozenset(variable.kinds))

    parts = [
        (kind, item_type)
        for kind, item_type in container_parts(actual_type)
        if kind in kinds
    ]

    if not parts:
        return None

    matched_kinds = frozenset(kind for kind, _ in parts)

    # Une variable n'est liée que si sa sorte est restreinte : sinon elle
    # garde la forme de la signature effacée (Container).
    if variable is not None and matched_kinds != frozenset(variable.kinds):
        bindings = _bind(bindings, variable, matched_kinds)

    return unify(
        item_pattern,
        _union(item_type for _, item_type in parts),
        bindings,
    )


def _unify_fixed_tuple(patterns, actual_type, bindings):
    if actual_type is Any or actual_type in (Tuple, tuple):
        item_types = [Any] * len(patterns)
    elif _is_union(actual_type):
        return join_bindings(
            [
                result
                for option in get_args(actual_type)
                for result in [_unify_fixed_tuple(patterns, option, bindings)]
                if result is not None
            ],
            bindings,
        )
    elif get_origin(actual_type) is tuple:
        arguments = get_args(actual_type)

        if _is_variadic_tuple(arguments):
            item_types = [arguments[0]] * len(patterns)
        elif len(arguments) == len(patterns):
            item_types = list(arguments)
        else:
            return None
    else:
        return None

    for pattern, item_type in zip(patterns, item_types):
        bindings = unify(pattern, item_type, bindings)

        if bindings is None:
            return None

    return bindings


def unify(pattern, actual_type, bindings):
    """
    Lie les variables de pattern pour qu'il accepte actual_type.

    Retourne les nouvelles liaisons (un nouveau dictionnaire), ou None si
    aucune liaison ne convient.
    """
    if isinstance(pattern, TypeVar):
        if pattern not in bindings:
            return _bind(bindings, pattern, actual_type)

        merged = merge_item_types(bindings[pattern], actual_type)

        return None if merged is None else _bind(bindings, pattern, merged)

    if isinstance(pattern, ContainerOf):
        return _unify_container(
            None,
            pattern.item,
            actual_type,
            bindings,
            variable=pattern.kind,
        )

    if not is_generic(pattern):
        return bindings if _accepts(pattern, actual_type) else None

    origin = get_origin(pattern)
    arguments = get_args(pattern)

    if origin is frozenset:
        return _unify_container({frozenset}, arguments[0], actual_type, bindings)

    if origin is tuple:
        if _is_variadic_tuple(arguments):
            return _unify_container({tuple}, arguments[0], actual_type, bindings)

        return _unify_fixed_tuple(arguments, actual_type, bindings)

    if _is_union(pattern):
        for option in arguments:
            result = unify(option, actual_type, bindings)

            if result is not None:
                return result

        return None

    raise NotImplementedError(f"Type générique non pris en charge : {pattern!r}")


def join_bindings(alternatives, base):
    """
    Réunit des liaisons obtenues pour plusieurs branches possibles.

    Une variable liée dans toutes les branches reçoit l'union de ses
    valeurs ; une variable absente d'une branche reste libre.
    """
    if not alternatives:
        return None

    if len(alternatives) == 1:
        return alternatives[0]

    joined = dict(base)
    variables = set().union(*(alternative.keys() for alternative in alternatives))

    for variable in variables:
        if any(variable not in alternative for alternative in alternatives):
            joined.pop(variable, None)
            continue

        values = [alternative[variable] for alternative in alternatives]

        if isinstance(variable, ContainerVar):
            kinds = frozenset().union(*values)

            if kinds == frozenset(variable.kinds):
                joined.pop(variable, None)
            else:
                joined[variable] = kinds
        else:
            joined[variable] = _union(values)

    return joined


def unify_any(pattern, actual_type, bindings):
    """
    Comme unify, mais pour un type attendu dont chaque branche est possible.

    Les liaisons obtenues pour les branches d'une union sont réunies au
    lieu de retenir la première : le résultat ne restreint jamais les
    arguments plus que nécessaire.
    """
    alternatives = [
        result
        for option in _branches(actual_type)
        for result in [unify(pattern, option, bindings)]
        if result is not None
    ]

    return join_bindings(alternatives, bindings)


def _functions_as_callable(bindings):
    from search.function_types import FunctionType

    if not any(isinstance(value, FunctionType) for value in bindings.values()):
        return bindings

    return {
        variable: Callable if isinstance(value, FunctionType) else value
        for variable, value in bindings.items()
    }


def _branches(type_hint):
    return list(get_args(type_hint)) if _is_union(type_hint) else [type_hint]


def infer_generic_call(name, signature, actual_types):
    """
    Infère le type du résultat d'un appel à une primitive générique.

    Chaque argument dont le type est une union est examiné branche par
    branche : un appel est valide si au moins une combinaison de branches
    satisfait la signature, et son résultat est l'union des résultats des
    combinaisons valides.
    """
    generic_positions = [
        index
        for index, pattern in enumerate(signature.parameter_types)
        if is_generic(pattern)
    ]

    choices = [
        _branches(actual_types[index]) if index in generic_positions
        else [actual_types[index]]
        for index in range(len(actual_types))
    ]

    combinations = 1

    for options in choices:
        combinations *= len(options)

    if combinations > MAX_COMBINATIONS:
        choices = [[actual_type] for actual_type in actual_types]

    results = []

    for combination in product(*choices):
        bindings = {}

        for index in generic_positions:
            bindings = unify(
                signature.parameter_types[index],
                combination[index],
                bindings,
            )

            if bindings is None:
                break

        if bindings is not None:
            results.append(substitute(signature.return_type, bindings))

    if not results:
        received = ", ".join(repr(actual_type) for actual_type in actual_types)
        raise TypeError(
            f"Arguments incompatibles pour {name} : reçu ({received}), "
            f"signature {signature.parameter_types!r} -> "
            f"{signature.return_type!r}"
        )

    return _union(results)


@dataclass(frozen=True)
class GenericCall:
    """
    Appel générique en cours de génération.

    bindings contient les faits : les liaisons tirées des arguments déjà
    rattachés. bounds contient la borne tirée du type attendu pour le
    résultat. Pour un argument, un fait l'emporte sur la borne : dans
    fill(I, ZERO, remove(UNITY, ...)), le conteneur doit contenir des
    positions, et non « des cellules ou des positions » comme le dirait le
    seul type Patch.
    """

    signature: GenericSignature
    bindings: dict = field(default_factory=dict)
    bounds: dict = field(default_factory=dict)

    def expecting(self, expected_type):
        """
        Tient compte du type attendu pour le résultat.

        Si le résultat ne peut pas satisfaire ce type, aucune borne n'est
        ajoutée : la vérification finale du résultat s'en chargera.
        """
        if expected_type is None or expected_type is Any:
            return self

        bounds = unify_any(self.signature.return_type, expected_type, {})

        if bounds is None:
            return self

        return GenericCall(self.signature, self.bindings, bounds)

    def _known(self):
        known = dict(self.bounds)

        for variable, value in self.bindings.items():
            if value is not Any:
                known[variable] = value

        return _functions_as_callable(known)

    def parameter_type(self, index):
        """
        Type attendu pour un argument, compte tenu des faits et de la borne.

        Les fonctions liées à une variable y sont remplacées par Callable :
        leur compatibilité est vérifiée par l'unification, pas par
        is_compatible.
        """
        return substitute(self.signature.parameter_types[index], self._known())

    def with_argument(self, index, actual_type):
        """Faits après rattachement d'un argument, ou None s'il est incompatible."""
        pattern = self.signature.parameter_types[index]

        if not is_generic(pattern):
            return self

        bindings = unify(pattern, actual_type, self.bindings)

        if bindings is None:
            return None

        return GenericCall(self.signature, bindings, self.bounds)

    @property
    def return_type(self):
        return substitute(self.signature.return_type, self._known())
