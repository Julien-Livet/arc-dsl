import json
import solvers
import inspect
from search.type_system import is_compatible
from search.dsl_registry import load_registry
from search.python_to_dsl import parse_solver
from search.expression_validator import infer_expression_type
from arc_types import Grid


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
    errors = []
    tasks = hodelTasks()
    ignoredTasks: set = {"a64e4611", }
    count: int = 0

    for i, task in enumerate(tasks): #TODO: remove enumerate
        #task = "a64e4611"
        #"""
        if (task in ignoredTasks):
            print(f"{task} task ignored")

            continue
        #"""
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

        except Exception as exc:
            print(f"{i}/{len(tasks)}")
            print(source)
            errors.append(f"{task}: {type(exc).__name__}: {exc}")
            raise exc

    print(f"{count}/{len(tasks)} passed tasks")

    assert not errors, "\n".join(errors)
