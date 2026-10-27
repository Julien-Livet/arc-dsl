import pytest

from arc_types import Boolean, Grid
from search.function_types import FunctionType
from search.function_inference import infer_compose, infer_fork, infer_chain, infer_lbind, infer_rbind, infer_matcher, infer_power


def test_infer_compose_grid_to_grid():
    outer = FunctionType((Grid,), Grid)
    inner = FunctionType((Grid,), Grid)

    result = infer_compose(outer, inner)

    assert result == FunctionType((Grid,), Grid)


def test_infer_compose_rejects_incompatible_types():
    outer = FunctionType((int,), Grid)
    inner = FunctionType((Grid,), Grid)

    with pytest.raises(TypeError, match="Types incompatibles"):
        infer_compose(outer, inner)


def test_infer_compose_rejects_binary_outer():
    outer = FunctionType((Grid, int), Grid)
    inner = FunctionType((Grid,), Grid)

    with pytest.raises(TypeError, match="outer doit accepter"):
        infer_compose(outer, inner)


def test_infer_compose_rejects_binary_inner():
    outer = FunctionType((Grid,), Grid)
    inner = FunctionType((Grid, int), Grid)

    with pytest.raises(TypeError, match="inner doit accepter"):
        infer_compose(outer, inner)


def test_infer_fork_grid_to_grid():
    outer = FunctionType((int, int), Grid)
    a = FunctionType((Grid,), int)
    b = FunctionType((Grid,), int)

    result = infer_fork(outer, a, b)

    assert result == FunctionType((Grid,), Grid)


def test_infer_fork_rejects_wrong_outer_arity():
    outer = FunctionType((int,), Grid)
    a = FunctionType((Grid,), int)
    b = FunctionType((Grid,), int)

    with pytest.raises(TypeError, match="outer doit accepter"):
        infer_fork(outer, a, b)


def test_infer_fork_rejects_incompatible_result():
    outer = FunctionType((Grid, int), Grid)
    a = FunctionType((int,), int)
    b = FunctionType((int,), int)

    with pytest.raises(TypeError, match="résultat de a"):
        infer_fork(outer, a, b)


def test_infer_fork_rejects_incompatible_inputs():
    outer = FunctionType((int, int), Grid)
    a = FunctionType((Grid,), int)
    b = FunctionType((int,), int)

    with pytest.raises(TypeError, match="types d'entrée incompatibles"):
        infer_fork(outer, a, b)


def test_infer_chain_grid_to_grid():
    h = FunctionType((int,), Grid)
    g = FunctionType((int,), int)
    f = FunctionType((Grid,), int)

    result = infer_chain(h, g, f)

    assert result == FunctionType((Grid,), Grid)


def test_infer_chain_rejects_incompatible_f_to_g():
    h = FunctionType((int,), Grid)
    g = FunctionType((Grid,), int)
    f = FunctionType((Grid,), int)

    with pytest.raises(TypeError, match="résultat de f"):
        infer_chain(h, g, f)


def test_infer_chain_rejects_incompatible_g_to_h():
    h = FunctionType((Grid,), Grid)
    g = FunctionType((int,), int)
    f = FunctionType((Grid,), int)

    with pytest.raises(TypeError, match="résultat de g"):
        infer_chain(h, g, f)


def test_infer_chain_rejects_binary_function():
    h = FunctionType((int,), Grid)
    g = FunctionType((int,), int)
    f = FunctionType((Grid, int), int)

    with pytest.raises(TypeError, match="f doit accepter"):
        infer_chain(h, g, f)


def test_infer_lbind_binary_function():
    function = FunctionType((int, Grid), Grid)

    result = infer_lbind(function, int)

    assert result == FunctionType((Grid,), Grid)


def test_infer_rbind_binary_function():
    function = FunctionType((Grid, int), Grid)

    result = infer_rbind(function, int)

    assert result == FunctionType((Grid,), Grid)


def test_infer_lbind_ternary_function():
    function = FunctionType((int, Grid, int), Grid)

    result = infer_lbind(function, int)

    assert result == FunctionType((Grid, int), Grid)


def test_infer_rbind_ternary_function():
    function = FunctionType((Grid, int, Grid), Grid)

    result = infer_rbind(function, Grid)

    assert result == FunctionType((Grid, int), Grid)


def test_infer_lbind_rejects_incompatible_fixed_type():
    function = FunctionType((Grid, int), Grid)

    with pytest.raises(TypeError, match="argument fixé"):
        infer_lbind(function, int)


def test_infer_rbind_rejects_incompatible_fixed_type():
    function = FunctionType((int, Grid), Grid)

    with pytest.raises(TypeError, match="argument fixé"):
        infer_rbind(function, int)


def test_infer_matcher_returns_boolean_function():
    function = FunctionType((Grid,), int)

    result = infer_matcher(function, int)

    assert result == FunctionType((Grid,), Boolean)


def test_infer_matcher_rejects_binary_function():
    function = FunctionType((Grid, int), int)

    with pytest.raises(TypeError, match="exactement un argument"):
        infer_matcher(function, int)


def test_infer_matcher_rejects_incompatible_target():
    function = FunctionType((Grid,), int)

    with pytest.raises(TypeError, match="incompatible"):
        infer_matcher(function, Grid)


def test_infer_power_one_returns_original_signature():
    function = FunctionType((Grid,), int)

    assert infer_power(function, 1) == function


def test_infer_power_grid_to_grid():
    function = FunctionType((Grid,), Grid)

    assert infer_power(function, 3) == function


def test_infer_power_rejects_incompatible_types():
    function = FunctionType((Grid,), int)

    with pytest.raises(TypeError, match="composable"):
        infer_power(function, 2)


def test_infer_power_rejects_zero():
    function = FunctionType((Grid,), Grid)

    with pytest.raises(ValueError, match="n >= 1"):
        infer_power(function, 0)


def test_infer_power_rejects_binary_function():
    function = FunctionType((Grid, int), Grid)

    with pytest.raises(TypeError, match="exactement un argument"):
        infer_power(function, 2)
