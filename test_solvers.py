import json
import solvers

def loadFolder(folder: str) -> dict:    
    with open(f"../../ARC-AGI-2/data/{folder}.txt", "r") as f:
        taskIds = f.read().split("\n")
    
    data: dict = {}

    for id_ in taskIds:
        with open(f"../../ARC-AGI-2/data/{folder}/{id_}.json") as f:
            data[id_] = json.load(f)
    
    return data

def test_solvers() -> None:
    data: dict = loadFolder("training")

    for name in dir(solvers):
        if (name.startswith("solve_")):
            taskId = name.removeprefix("solve_")

            if (taskId == "e40b9e2f" or not taskId in data):
                print(f"{taskId} task skipped")

                continue

            f = getattr(solvers, name)
            
            train = data[taskId]["train"]

            for pair in train:
                I, O = pair["input"], pair["output"]

                I = tuple(map(tuple, I))
                O = tuple(map(tuple, O))

                assert(O == f(I))
