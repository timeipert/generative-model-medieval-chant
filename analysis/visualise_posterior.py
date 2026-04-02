import matplotlib.pyplot as plt
import numpy as np

import math
import matplotlib.pyplot as plt
import seaborn as sns

def extract_posterior_data(trace, matrix_name="A"):
    stacked = trace.posterior[matrix_name].stack(samples=("chain", "draw"))
    return stacked.values


def plot_normalized_histogram(ax, samples, bins, color='teal', alpha=0.8):
    counts, bin_edges = np.histogram(samples, bins=bins, density=True)

    max_count = np.max(counts)
    if max_count > 0:
        counts = counts / max_count

    ax.fill_between(bin_edges[:-1], counts, step="post", color=color, alpha=alpha, linewidth=0)
    ax.axhline(0, color='white', alpha=0.1, linewidth=1)


def style_cell_axes(ax, i, j, v_size, alphabet_list):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    if j == 0:
        ax.set_ylabel(alphabet_list[i], rotation=0, labelpad=12, fontweight='bold', va='center')
    if i == v_size - 1:
        ax.set_xlabel(alphabet_list[j], fontweight='bold')


def plot_transition_matrix_posterior(trace, param="A", alphabet="abcdefghjklmnopqrs"):
    alphabet_list = list(alphabet)
    v_size = len(alphabet_list)

    data = extract_posterior_data(trace, param)

    fig, axes = plt.subplots(v_size, v_size, figsize=(18, 18), sharex=True, sharey=True)

    bins = np.linspace(0, 1, 40)

    for i in range(v_size):
        for j in range(v_size):
            ax = axes[i, j]
            samples = data[i, j, :]

            plot_normalized_histogram(ax, samples, bins)
            style_cell_axes(ax, i, j, v_size, alphabet_list)

    plt.suptitle(f"Posterior Transition Densities ({param})\nRows: From | Cols: To", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()





def _plot_single_heatmap(ax, matrix, alphabet, mode_index):
    sns.heatmap(
        matrix,
        xticklabels=alphabet,
        yticklabels=alphabet,
        cmap="binary",
        annot=False,
        ax=ax,
        cbar=False
    )
    ax.set_title(f"Mode {mode_index}")
    ax.set_xlabel("To Pitch")
    ax.set_ylabel("From Pitch")


def plot_posterior_matrices(trace, model, matrix_name="A"):
    posterior_mean = trace.posterior[matrix_name].mean(dim=["chain", "draw"]).values

    num_modes = posterior_mean.shape[0]
    cols = 4
    rows = math.ceil(num_modes / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()

    for i in range(num_modes):
        _plot_single_heatmap(axes[i], posterior_mean[i], model.alphabet, i)

    for j in range(num_modes, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()