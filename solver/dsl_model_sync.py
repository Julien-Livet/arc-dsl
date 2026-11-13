
import argparse
from .dsl_engine import Engine
from dsl_model.dsl_model import DSLModel
import dsl_model.utils as utils
from dsl_model.worker_sync import Worker
from dsl_model.utils import VOCAB
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description="DSL model training"
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers used to generate experiments (default: 4)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="dsl_model.pt",
        help="Name of the model (default: dsl_model.pt)",
    )

    args = parser.parse_args()

    if args.num_workers < 1:
        parser.error("--num-workers must be >= 1")

    return args

def truncatedProgram(program: str) -> str:
    return program if len(program) < 16 else program[:16] + "..."

if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = Engine()
    n = engine.count()
    indexes = engine.orderedIndexes()
    process = tqdm(total = len(indexes), desc = "Programs", dynamic_ncols = True)
    dslModel = DSLModel(len(VOCAB.token2id), d_model=256, device = device)
    model = dslModel.to(device)

    if (os.path.exists(args.model)):
        checkpoint = torch.load(args.model, map_location=device)
        model.load_state_dict(checkpoint["model_state"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4
    )

    workers: list = []

    for _ in range(args.num_workers):
        workers.append(Worker())

    model_version: int = 1
    count: int = 0
    trajectoryLen: int = 0
    trajectoryIndex: int = 0
    targetProgram: str = ""
    currentProgram: str = ""

    while (True):
        worker_L_tokens: list = []
        
        for worker in workers:
            loop: bool = True
            c: bool = False

            while (loop):
                if (worker.process(device, engine, model)):
                    if (worker.j != None):
                        count += 1
                        process.update()

                    if (not len(indexes)):
                        c = True
                        break
                    else:
                        worker.init(engine, device, indexes.pop(0))

                        if (args.num_workers == 1):
                            trajectoryLen = len(worker.costs)
                            trajectoryIndex = 1
                            targetProgram = worker.costs[0][1]
                            currentProgram = "I"
                else:
                    loop = False

            if (args.num_workers == 1):
                trajectoryIndex = trajectoryLen - len(worker.costs) + 1
                currentProgram = worker.program if worker.program else currentProgram

            if (count == n):
                break

            if (c):
                continue

            model.train()
            optimizer.zero_grad()
            target_ids = utils.encode_program_tokens(worker.costs[0][1], VOCAB).to(device)
            decoder_input = target_ids[:-1]
            decoder_target = target_ids[1:]
            logits = model.decoder(decoder_input.unsqueeze(0), worker.z_context)

            L_tokens = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1)
                ),
                decoder_target.reshape(-1)
            )
            
            worker_L_tokens.append((worker, L_tokens))

        if (count == n):
            break

        list_L_total: list = []

        for worker, L_tokens in worker_L_tokens:
            list_L_total.append(worker.update(device, model, L_tokens, args.model))
        
        if (not len(list_L_total)):
            continue

        loss = torch.stack(list_L_total).mean()
        loss_value = loss.detach().item()

        loss.backward()
        optimizer.step()

        model_version += 1

        if (args.num_workers == 1):
            process.set_postfix(
                loss=loss_value,
                trajectory=f"{trajectoryIndex}/{trajectoryLen}",
                target=truncatedProgram(targetProgram),
                program=truncatedProgram(currentProgram),
                depth=utils.programDepth(targetProgram)
            )
        else:
            process.set_postfix(
                version=model_version,
                workers=args.num_workers,
                loss=loss_value,
            )

    process.close()
