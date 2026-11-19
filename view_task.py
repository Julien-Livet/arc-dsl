import dsl_model.utils as utils
import json
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import sys

arc_colors = [
    '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
    '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
]
cmap_arc = colors.ListedColormap(arc_colors)
norm_arc = colors.BoundaryNorm(np.arange(-0.5, 10, 1), cmap_arc.N)

def neg_arc_color(i: int):
    hex_color = arc_colors[i].lstrip('#')

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    r_neg = 255 - r
    g_neg = 255 - g
    b_neg = 255 - b

    return f"#{r_neg:02X}{g_neg:02X}{b_neg:02X}"

def init_plot(ax, data):
    ax.set_xticks([])
    ax.set_yticks([])

    im = ax.matshow(data, cmap=cmap_arc, norm=norm_arc)

    cbar = ax.figure.colorbar(
        im,
        ax=ax,
        ticks=np.arange(0, 10),
        pad=0.02
    )
    cbar.ax.set_yticklabels(np.arange(0, 10))
    cbar.outline.set_edgecolor('black')

    rows, cols = np.array(data).shape

    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)

    ax.grid(which='minor', color='white', linewidth=1)

    ax.tick_params(
        which='both',
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False
    )

    texts = []

    for (i, j), z in np.ndenumerate(data):
        t = ax.text(
            j, i, f'{z}',
            c=neg_arc_color(z),
            ha='center',
            va='center',
            size='medium'
        )
        texts.append(t)

    return im, texts

def update_plot(im, texts, data):
    im.set_data(data)

    for t, z in zip(texts, np.array(data).flat):
        t.set_text(str(z))
        t.set_color(neg_arc_color(z))

def plot_data(ax, data):
    if hasattr(ax, '_cbar'):
        ax._cbar.remove()
        del ax._cbar

    ax.clear()

    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.matshow(data, cmap = cmap_arc, norm = norm_arc)
    ax._cbar = ax.figure.colorbar(im, ax = ax, ticks = np.arange(0, 10))
    cbar = ax._cbar
    cbar.ax.set_yticklabels(np.arange(0, 10)) 
    cbar.outline.set_edgecolor('black')

    for (i, j), z in np.ndenumerate(data):
        ax.text(j, i, f'{z}', c = neg_arc_color(z), ha = 'center', va = 'center', size = "medium")#, bbox = dict(boxstyle = 'round', facecolor = 'white', edgecolor = '0.3'))

    ax.set_axis_off()
    rows, cols = np.array(data).shape
    ax.set_xticks(np.arange(-0.5, cols, 1), minor = True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor = True)
    ax.grid(which = 'minor', color = 'white', linestyle = '-', linewidth = 1)
    ax.tick_params(which = 'both', bottom = False, left = False, labelbottom = False, labelleft = False)

def show(text, pairs, buffer = None):
    if (buffer):
        iptMaxCols, optMaxCols = 0, 0
        iptRows, optRows = 0, 0

        for k in range(len(pairs)):
            ipt = pairs[k]["input"]
            opt = pairs[k]["output"]

            rows, cols = np.array(ipt).shape
            iptRows += rows
            iptMaxCols = max(iptMaxCols, cols)
            rows, cols = np.array(opt).shape
            optRows += rows
            optMaxCols = max(optMaxCols, cols)

        width, height = iptMaxCols + optMaxCols, max(iptRows, optRows)
        dpi = 150

        fig, axs = plt.subplots(len(pairs), 2, figsize = (100 * width / dpi, 100 * height / dpi), dpi = dpi)
    else:
        fig, axs = plt.subplots(len(pairs), 2)

    if (len(axs.shape) == 1):
        axs.resize((1, axs.shape[0]))

    fig.suptitle(text + ' input/output pairs')

    for k in range(len(pairs)):
        ipt = pairs[k]["input"]
        opt = pairs[k]["output"]

        plot_data(axs[k][0], ipt)
        plot_data(axs[k][1], opt)

    if (buffer):
        plt.savefig(buffer, format = "png", bbox_inches = 'tight')
        plt.close()
    else:
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show()

def view(folder: str, task: str):
    assert(folder in ("training", "evaluation"))

    with open(f"../../ARC-AGI-2/data/{folder}/{task}.json", "r") as file:
        data = json.loads(file.read())

    train = data["train"]
    test = data["test"]

    show("Train", train)
    show("Test", test)

def init_cost_plot(ax, scores, pair_index):
    values = scores[utils.scoreColumns].to_numpy(dtype=float)

    y = np.arange(len(utils.scoreColumns))

    bars = ax.barh(y, values)

    ax.set_yticks(y)
    ax.set_yticklabels(utils.scoreColumns, fontsize=8)
    ax.invert_yaxis()

    ax.set_title(f'Pair {pair_index}', fontsize=10)

    ax.grid(axis='x', linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)

    # Valeurs numériques
    texts = []

    xmax = max(1.0, np.max(np.abs(values)))
    ax.set_xlim(0, xmax * 1.2)

    for i, value in enumerate(values):
        t = ax.text(
            value,
            i,
            f' {value:.3g}',
            va='center',
            fontsize=8
        )
        texts.append(t)

    return bars, texts


def update_cost_plot(ax, bars, texts, scores):
    values = scores[utils.scoreColumns].to_numpy(dtype=float)

    xmax = max(1.0, np.max(np.abs(values)))

    for bar, text, value in zip(bars, texts, values):

        bar.set_width(value)

        text.set_position((value, bar.get_y() + bar.get_height() / 2))
        text.set_text(f' {value:.3g}')

    ax.set_xlim(0, xmax * 1.2)

if (__name__ == "__main__"):
    view(sys.argv[-2], sys.argv[-1])
