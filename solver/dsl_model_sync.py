import argparse
from .dsl_engine import Engine
from dsl_model.dsl_model import DSLModel
import dsl_model.utils as utils
from dsl_model.worker_sync import Worker
from dsl_model.utils import VOCAB
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import pandas as pd
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TaskProgressColumn,
)
import torch
import torch.nn.functional as F
from tqdm import tqdm
import view_task

def make_goal_table(current, next_program, target):
    table = Table(
        title="Goal",
        show_header=True,
        header_style="bold cyan",
        expand=False,
    )

    table.add_column("Program")
    table.add_column("Cost", justify="right")
    table.add_column("Expression")

    table.add_row(
        "Current",
        f"{current[0]:.3e}",
        current[1],
    )
    table.add_row(
        "Next",
        f"{next_program[0]:.3e}",
        next_program[1],
    )
    table.add_row(
        "Target",
        f"{target[0]:.3e}",
        target[1],
    )

    return table

def make_pool_table(pool):
    table = Table(
        title="Pool",
        show_header=True,
        header_style="bold cyan",
        expand=False,
    )

    table.add_column("##", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Expression")

    for index, (cost, expression) in enumerate(pool, start=1):
        table.add_row(
            f"{index:02d}",
            f"{cost:.3e}",
            expression,
        )

    return table

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

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot grids",
    )

    args = parser.parse_args()

    if args.num_workers < 1:
        parser.error("--num-workers must be >= 1")

    return args

def get_signature(inputs, outputs):
    return tuple(
        (np.asarray(l).shape, np.asarray(r).shape)
        for l, r in zip(inputs, outputs)
    )

def rebuild_plot(fig, inputs, outputs, dfTrain):
    fig.clear()

    n = len(inputs)
    axs = fig.subplots(n, 3, squeeze=False)
    plots: list = []

    for k, (l, r) in enumerate(zip(inputs, outputs)):
        p1 = view_task.init_plot(axs[k][0], l)
        p2 = view_task.init_plot(axs[k][1], r)
        p3 = view_task.init_cost_plot(axs[k][2], dfTrain.iloc[k], k)

        plots.append((p1, p2, p3))

    return axs, plots

if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = Engine()
    n = engine.count()
    indexes = engine.orderedIndexes()
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
    currentWorker = None
    trajectoryLen: int = 0
    currentProgram: str = "I"
    trajectoryIndex: int = 1
    cost: float = math.inf

    if (args.plot and args.num_workers == 1):
        plt.ion()

        fig, axs = plt.subplots(1, 2)
        fig.suptitle('Current vs target program outputs')

        fig.canvas.manager.set_window_title("Visual model training via ARC grids")
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show(block=False)

        plots: list = []
        previous_signature = None
        axs = None

    if (args.num_workers == 1):
        console = Console()
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("({task.completed:.0f}/{task.total:.0f})"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("loss={task.fields[loss]}"),
            TextColumn("trajectory={task.fields[trajectory]}"),
            TextColumn("iters={task.fields[iters]}"),
            TextColumn("count={task.fields[count]}"),
            console=console,
            auto_refresh=False,
        )
        task = progress.add_task(
            "Programs",
            total=n,
            loss="N/A",
            trajectory="N/A",
            iters="N/A",
            count="N/A",
        )
        live = None
        goal_table = Table(title="Goal")
        goal_table.add_column("Status")
        goal_table.add_row("No active trajectory")
    else:
        process = tqdm(total = len(indexes), desc = "Programs", dynamic_ncols = True)

    while (True):
        worker_L_tokens: list = []
        
        for worker in workers:
            loop: bool = True
            c: bool = False

            while (loop):
                if (worker.process(device, engine, model)):
                    if (worker.j != None):
                        count += 1

                        if (args.num_workers == 1):
                            progress.update(
                                task,
                                advance=1)
                        else:
                            process.update()

                    if (not len(indexes)):
                        c = True
                        break
                    else:
                        worker.init(engine, device, indexes.pop(0))

                        if (args.num_workers == 1):
                            trajectoryLen = len(worker.costs)
                            currentProgram = "I"

                            if (args.plot):
                                axs, plots = rebuild_plot(fig, worker.grids, worker.rawOutputs, worker.candidates[0][1])
                                previous_signature = signature = get_signature(worker.grids, worker.rawOutputs)
                else:
                    loop = False

            if (args.num_workers == 1):
                currentWorker = worker

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
            if (len(currentWorker.costs)):
                trajectoryIndex: int = trajectoryLen - len(currentWorker.costs) + 1

            if (currentWorker.program):
                currentProgram = currentWorker.program

                try:
                    cost = float(currentWorker.df.sum(axis=0, skipna=False)["Total cost"] if isinstance(currentWorker.df, pd.DataFrame) else currentWorker.costs[0][0])

                    if (args.plot):
                        signature = get_signature(currentWorker.programOutputs, currentWorker.rawOutputs)

                        if (signature != previous_signature):
                            axs, plots = rebuild_plot(fig, currentWorker.programOutputs, worker.rawOutputs, worker.df)
                            previous_signature = signature

                        for k, (l, r) in enumerate(zip(currentWorker.programOutputs, worker.rawOutputs)):
                            im1, txt1 = plots[k][0]
                            im2, txt2 = plots[k][1]
                            bars, texts = plots[k][2]

                            view_task.update_plot(im1, txt1, l)
                            view_task.update_plot(im2, txt2, r)
                            view_task.update_cost_plot(axs[k][2], bars, texts, currentWorker.df.iloc[k])

                        fig.suptitle(f'Current vs target program outputs\n'
                                     f'Current: {cost:.3e} {currentWorker.program}\n'
                                     f'Next: {currentWorker.costs[0][0]:.3e} {currentWorker.costs[0][1]}\n'
                                     f'Target: {0.0:.3e} {currentWorker.targetProgram}')
                        fig.canvas.draw_idle()
                        fig.canvas.flush_events()
                        plt.pause(0.01)
                except:
                    cost = math.inf

            targetProgram: str = currentWorker.targetProgram

            progress.update(
                task,
                advance=0,
                loss=f"{loss_value:.3e}",
                trajectory=f"{trajectoryIndex}/{trajectoryLen}",
                iters=currentWorker.iters_since_improvement,
                count=currentWorker.count,
            )

            if (len(currentWorker.costs)):
                goal_table = make_goal_table(
                    (cost, currentProgram),
                    currentWorker.costs[0],
                    (0.0, targetProgram),
                )

            pool: list = [(candidate[1].sum(axis = 0, skipna = False)["Total cost"], candidate[0]) for candidate in currentWorker.candidates]
            unique_pool = {}

            for candidate in currentWorker.candidates:
                expression = candidate[0]
                cost = candidate[1].sum(axis=0, skipna=False)["Total cost"]

                if expression not in unique_pool or cost < unique_pool[expression]:
                    unique_pool[expression] = cost

            pool = [
                (cost, expression)
                for expression, cost in unique_pool.items()
            ]
            pool.sort(key=lambda item: item[0])

            pool_table = make_pool_table(pool)

            renderable = Group(progress, goal_table, pool_table)

            if (live is None):
                live = Live(
                    renderable,
                    console=console,
                    refresh_per_second=4,
                    screen=False,
                )
                live.start()
            else:
                live.update(renderable, refresh=True)
        else:
            process.set_postfix(
                version=model_version,
                workers=args.num_workers,
                loss=f"{loss_value:.3e}",
            )

    if (args.num_works == 1):
        live.stop()

        plt.ioff()
        plt.show()
    else:
        process.close()
