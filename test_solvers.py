import json
import solvers
import inspect
from search.type_system import is_compatible
from search.dsl_registry import load_registry
from search.python_to_dsl import parse_solver
from search.expression_validator import infer_expression_type
from search.expressions import expression_to_str, PrimitiveRefExpr, ApplyExpr
from search.dsl_generator import (
    get_next_symbols,
    GenerationState,
    get_current_goal,
    get_current_depth,
    select_symbol,
    walk_expression,
    APPLY,
)
from arc_types import Grid


def programDepth(program: str) -> int:
    count: int = 0
    n: int = 0

    for c in program:
        if (c == "("):
            n += 1
        elif (c == ")"):
            n -= 1

        count = max(count, n)

    return count


def hodelTasks() -> list[str]:
    with open("solvers.py", "r") as f:
        lines = f.read().split("\n")

    lines = filter(lambda x: x.startswith("def solve_"), lines)
    tasks: list = []

    for line in lines:
        task = line.removeprefix("def solve_")
        tasks.append(task[:task.index("(")])

    return tasks


def loadFolder(folder: str) -> dict[str, str]:    
    with open(f"../../ARC-AGI-2/data/{folder}.txt", "r") as f:
        taskIds = f.read().split("\n")
    
    data: dict = {}

    for id_ in taskIds:
        with open(f"../../ARC-AGI-2/data/{folder}/{id_}.json") as f:
            data[id_] = json.load(f)
    
    return data


def test_grids() -> None:
    data: dict = loadFolder("training")

    tasks = hodelTasks()

    for task in tasks:
        if (not task in data):
            print(f"{task} task skipped")

            continue

        f = getattr(solvers, f"solve_{task}")

        train = data[task]["train"]

        for pair in train:
            I, O = pair["input"], pair["output"]

            I = tuple(map(tuple, I))
            O = tuple(map(tuple, O))

            assert(O == f(I))


def test_types() -> None:
    registry = load_registry()
    errors: dict = {}
    tasks = hodelTasks()
    count: int = 0

    for task in tasks:
        solver = getattr(solvers, f"solve_{task}")

        if not callable(solver):
            continue

        try:
            source = inspect.getsource(solver)
            expression = parse_solver(source, registry)
            result_type = infer_expression_type(expression, registry)

            assert is_compatible(result_type, Grid) or is_compatible(Grid, result_type), (
                f"{task}: type de retour attendu Grid, obtenu {result_type}"
            )

            count += 1
        except (TypeError, AssertionError) as exc:
            errors[task] = f"{task}: {type(exc).__name__}: {exc}"

    print(f"{count}/{len(tasks)} passed tasks")

    assert not errors, "\n".join(errors.values())


def reconstructExpression(expression, registry, maxDepth: int):
    """Reconstruit une expression symbole par symbole avec le générateur."""
    state = GenerationState(expected_type=Grid, max_depth=maxDepth)

    for node in walk_expression(expression):
        if (state.finished):
            raise AssertionError(
                f"Génération terminée trop tôt : {expression_to_str(state.result)}"
            )

        goal = get_current_goal(state)

        symbols = get_next_symbols(
            expected_type=goal,
            registry=registry,
            depth=get_current_depth(state),
            max_depth=state.max_depth,
        )

        if not symbols:
            raise RuntimeError(
                f"Aucun symbole disponible pour le type {goal}"
            )

        # Une application f(...)(x) est annoncée par le symbole APPLY.
        if type(node) is ApplyExpr:
            if APPLY not in symbols:
                raise AssertionError(
                    f"Symbole {APPLY.name} absent des candidats : {symbols}"
                )

            state = select_symbol(state, APPLY, registry)

            continue

        primitiveSymbols = {
            symbol for symbol in symbols
            if isinstance(symbol, str)
        }

        namedSymbols = {
            symbol.name: symbol
            for symbol in symbols
            if getattr(symbol, "name", None)
        }

        element = node.name

        if type(node) is PrimitiveRefExpr:
            if element in namedSymbols:
                selected_symbol = namedSymbols[element]
            elif element in primitiveSymbols:
                selected_symbol = element
            else:
                raise AssertionError(
                    f"Symbole {element} absent des candidats : {symbols}"
                )
        else:
            if element in primitiveSymbols:
                selected_symbol = element
            elif element in namedSymbols:
                selected_symbol = namedSymbols[element]
            else:
                raise AssertionError(
                    f"Symbole {element} absent des candidats : {symbols}"
                )

        state = select_symbol(
            state,
            selected_symbol,
            registry,
        )

    if (not state.finished):
        raise AssertionError("Génération inachevée")

    return state.result


def test_expressions() -> None:
    registry = load_registry()
    errors: dict = {}
    tasks = hodelTasks()
    count: int = 0

    for task in tasks:
        solver = getattr(solvers, f"solve_{task}")

        if not callable(solver):
            continue

        source = inspect.getsource(solver)
        expression = parse_solver(source, registry)
        program = expression_to_str(expression)

        try:
            result = reconstructExpression(expression, registry, programDepth(program))
            reconstructed = expression_to_str(result)

            assert reconstructed == program, (
                f"programme reconstruit différent : {reconstructed}"
            )

            count += 1
        except (AssertionError, RuntimeError, TypeError) as exc:
            errors[task] = f"{task}: {type(exc).__name__}: {exc}"

    print(f"{count}/{len(tasks)} passed tasks")

    assert not errors, "\n".join(errors.values())


def dsl_dataset() -> None:
    data: dict = loadFolder("training")
    registry = load_registry()
    tasks = hodelTasks()
    programs: list = []

    for task in tasks:
        if (not task in data):
            print(f"{task} task skipped")

            continue

        solver = getattr(solvers, f"solve_{task}")
        source = inspect.getsource(solver)
        expression = parse_solver(source, registry)
        programs.append(expression_to_str(expression))

    with open("dsl_dataset.txt", "w") as f:
        f.write("\n".join(programs))
