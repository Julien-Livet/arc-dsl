from arc_types import Grid
from search.function_types import FunctionType


def test_unary_function_signature():
    signature = FunctionType((Grid,), Grid)

    assert signature.arity == 1
    assert signature.accepts_arity(1)
    assert not signature.accepts_arity(2)


def test_binary_function_signature():
    signature = FunctionType((Grid, int), Grid)

    assert signature.arity == 2
    assert signature.accepts_arity(2)
    assert not signature.accepts_arity(1)


def test_variadic_function_signature():
    signature = FunctionType((Grid,), Grid, variadic=True)

    assert signature.arity is None
    assert signature.accepts_arity(1)
    assert signature.accepts_arity(3)


def test_compatible_signatures():
    first = FunctionType((Grid,), Grid)
    second = FunctionType((Grid,), Grid)

    assert first.is_compatible_with(second)


def test_incompatible_argument_types():
    first = FunctionType((int,), Grid)
    second = FunctionType((Grid,), Grid)

    assert not first.is_compatible_with(second)


def test_incompatible_return_types():
    first = FunctionType((Grid,), int)
    second = FunctionType((Grid,), Grid)

    assert not first.is_compatible_with(second)
