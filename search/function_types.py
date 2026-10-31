from dataclasses import dataclass
from typing import Any

from search.type_system import is_compatible


@dataclass(frozen=True)
class FunctionType:
    """Signature de type d'une fonction du DSL."""

    parameter_types: tuple
    return_type: Any
    variadic: bool = False

    @property
    def arity(self):
        """Nombre d'arguments, ou None si l'arité est variable."""
        if self.variadic:
            return None
        return len(self.parameter_types)

    def accepts_arity(self, count):
        """Indique si la fonction accepte ce nombre d'arguments."""
        if self.variadic:
            return count >= len(self.parameter_types)
        return count == len(self.parameter_types)

    def is_compatible_with(self, expected):
        """
        Vérifie si cette signature peut satisfaire une signature attendue.

        La compatibilité est volontairement prudente pour les FunctionType :
        les arguments doivent être compatibles dans les deux sens, et le
        résultat doit satisfaire le type de retour attendu.
        """
        from typing import Callable, get_origin
        from collections.abc import Callable as ABCCallable

        # Callable non paramétré : aucune contrainte sur la signature.
        if expected in (Callable, ABCCallable):
            return True

        # Callable paramétré : à traiter comme une signature attendue.
        if get_origin(expected) in (Callable, ABCCallable):
            args = expected.__args__

            # Callable[..., R] : arité et types des paramètres libres.
            if args[0] is Ellipsis:
                return is_compatible(self.return_type, args[1])

            # Callable[[T1, T2, ...], R]
            parameter_types, return_type = args
            expected_function = FunctionType(
                parameter_types=tuple(parameter_types),
                return_type=return_type,
            )
            return self.is_compatible_with(expected_function)

        if not isinstance(expected, FunctionType):
            return False

        if self.variadic != expected.variadic:
            return False

        if len(self.parameter_types) != len(expected.parameter_types):
            return False

        parameters_compatible = all(
            is_compatible(actual, wanted)
            and is_compatible(wanted, actual)
            for actual, wanted in zip(
                self.parameter_types,
                expected.parameter_types,
            )
        )

        return (
            parameters_compatible
            and is_compatible(self.return_type, expected.return_type)
        )
