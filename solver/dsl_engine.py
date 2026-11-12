from bisect import bisect_left
import inspect
import json
from multiprocessing import Pool
import numpy as np
from search.dsl_registry import load_registry
from search.expressions import expression_to_str, evaluate_expression
from search.python_to_dsl import parse_solver
import solvers


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


def size_cost(x: np.ndarray, y: np.ndarray) -> float:
    if (not x.shape):
        return np.linalg.norm(np.array(y.shape, dtype = float))

    try:
        return np.linalg.norm(np.array(x.shape, dtype = float) - np.array(y.shape, dtype = float))
    except Exception:
        return float("inf")


def value_cost(x: np.ndarray, y: np.ndarray) -> float:
    if (x.shape == y.shape):
        return np.linalg.norm(x - y)

    try:
        return abs(sum(sum(x)) - sum(sum(y)))
    except Exception:
        return float("inf")


def pixel_overlap_cost(x: np.ndarray, y: np.ndarray) -> float:
    if (x.shape != y.shape):
        return np.prod(x.shape) + np.prod(y.shape)

    total = x.size
    matches = np.sum(x == y)

    return 1.0 - (matches / total)


def bounding_box(arr: np.ndarray) -> tuple[float, ...]:
    coords = np.argwhere(arr != 0)
    
    if (coords.size == 0):
        return None

    y_min, x_min = coords.min(axis = 0)
    y_max, x_max = coords.max(axis = 0)

    return (y_min, x_min, y_max, x_max)


def bounding_box_cost(x: np.ndarray, y: np.ndarray) -> float:
    box_x = bounding_box(x)
    box_y = bounding_box(y)

    if (box_x is None and box_y is None):
        return 0.0

    if (box_x is None or box_y is None):
        return 1.0

    diff = np.linalg.norm(np.array(box_x, dtype = float) - np.array(box_y, dtype = float))
    norm = np.linalg.norm(np.array(x.shape, dtype = float) + np.array(y.shape, dtype = float))

    return diff / (norm + 1e-8)


def arc_heuristic(x: np.ndarray, y: np.ndarray) -> float:
    return size_cost(x, y) + bounding_box_cost(x, y) + pixel_overlap_cost(x, y) + value_cost(x, y)


def compute(registry, inputs: tuple, outputs: tuple, expression) -> tuple[float, str]:
    cost: float = 0.0

    try:
        for input, output in zip(inputs, outputs):
            o = np.array(evaluate_expression(expression, input, registry))
            cost += arc_heuristic(o, np.array(output))
    except:
        return None

    return (cost, expression_to_str(expression))


class Engine:
    def __init__(self):
        data: dict = loadFolder("training")
        self.registry = load_registry()
        tasks = hodelTasks()

        self.expressions: list = []
        self._grids: list = []
        self._outputs: list = []

        for task in tasks:
            if (not task in data):
                continue

            solver = getattr(solvers, f"solve_{task}")

            if not callable(solver):
                continue

            source = inspect.getsource(solver)
            self.expressions.append(parse_solver(source, self.registry))

            train = data[task]["train"]

            grids: list = []
            outputs: list = []

            for pair in train:
                I, O = pair["input"], pair["output"]
                grids.append(tuple(map(tuple, I)))
                outputs.append(tuple(map(tuple, O)))

            self._grids.append(grids)
            self._outputs.append(outputs)


    def count(self) -> int:
        return len(self.expressions)


    def orderedIndexes(self) -> list[int]:
        iPrograms: list = sorted([(i, expression_to_str(e)) for i, e in enumerate(self.expressions)],
                                 key = lambda x: (len(x[1]), x[1]))

        return [i for i, _ in iPrograms]


    def program(self, i: int) -> str:
        return expression_to_str(self.expressions[i])


    def grids(self, i: int) -> tuple:
        return self._grids[i]


    def outputs(self, i: int) -> tuple:
        return self._outputs[i]


    def dfIdentity(self, index: int) -> list[tuple[int, ...]]:
        inputs = self.grids(index)
        outputs = self.outputs(index)

        results: list = []

        for input, output in zip(inputs, outputs):
            i = np.array(input)
            o = np.array(output)
            results.append((arc_heuristic(i, o),
                            size_cost(i, o),
                            bounding_box_cost(i, o),
                            pixel_overlap_cost(i, o),
                            value_cost(i, o)))

        return results


    def dfIdentityPairs(self, inputs: tuple, outputs: tuple) -> list[tuple[int, ...]]:
        results: list = []

        for input, output in zip(inputs, outputs):
            i = np.array(input)
            o = np.array(output)
            results.append((arc_heuristic(i, o),
                            size_cost(i, o),
                            bounding_box_cost(i, o),
                            pixel_overlap_cost(i, o),
                            value_cost(i, o)))

        return results


    def dfExpressionPairs(self, inputs: tuple, outputs: tuple, expression) -> list[tuple[int, ...]]:
        results: list = []

        for input, output in zip(inputs, outputs):
            oTmp = np.array(evaluate_expression(expression, input, self.registry))
            o = np.array(output)
            results.append((arc_heuristic(oTmp, o),
                            size_cost(oTmp, o),
                            bounding_box_cost(oTmp, o),
                            pixel_overlap_cost(oTmp, o),
                            value_cost(oTmp, o)))

        return results


    def dfExpression(self, index: int, expression) -> list[tuple[int, ...]]:
        inputs = self.grids(index)
        outputs = self.outputs(index)

        results: list = []

        for input, output in zip(inputs, outputs):
            oTmp = np.array(evaluate_expression(expression, input, self.registry))
            o = np.array(output)
            results.append((arc_heuristic(oTmp, o),
                            size_cost(oTmp, o),
                            bounding_box_cost(oTmp, o),
                            pixel_overlap_cost(oTmp, o),
                            value_cost(oTmp, o)))

        return results


    def expressionOutputs(self, inputs: tuple, expression) -> list[tuple[int, ...]]:
        results: list = []

        for input in inputs:
            results.append(evaluate_expression(expression, input, self.registry))

        return results


    def trajectory(self, i: int) -> list[tuple[float, str]]:
        inputs = self.grids(i)
        outputs = self.outputs(i)
        params: list = []

        for expression in self.expressions:
            params.append((self.registry, inputs, outputs, expression))

        with Pool() as pool:
            results = pool.starmap(compute, params)

        results = filter(lambda x: x, results)
        results = sorted(results, key = lambda x: (x[0], len(x[1]), x[1]))
        costs = [c for c, _ in results]

        df = self.dfIdentity(i)
        cost: float = 0.0

        for row in df:
            cost += row[0]

        j = bisect_left(costs, cost)

        return results[:j]
