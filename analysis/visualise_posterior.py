import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _get_stacked_samples(trace, var_name):
    return trace.posterior[var_name].stack(samples=("chain", "draw")).values


def _get_posterior_mean(trace, var_name):
    return trace.posterior[var_name].mean(dim=["chain", "draw"]).values


def _draw_density_histogram(ax, samples, bins, color="teal", alpha=0.8):
    counts, bin_edges = np.histogram(samples, bins=bins, density=True)

    max_count = np.max(counts)
    if max_count > 0:
        counts = counts / max_count

    ax.fill_between(bin_edges[:-1], counts, step="post", color=color, alpha=alpha, linewidth=0)
    ax.axhline(0, color="white", alpha=0.1, linewidth=1)


def _format_grid_cell_axes(ax, row_idx, col_idx, grid_size, labels):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    if col_idx == 0:
        ax.set_ylabel(labels[row_idx], rotation=0, labelpad=12, fontweight="bold", va="center")
    if row_idx == grid_size - 1:
        ax.set_xlabel(labels[col_idx], fontweight="bold")


def _draw_heatmap(ax, matrix, labels, title=None):
    sns.heatmap(
        matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="binary",
        annot=False,
        ax=ax,
        cbar=False
    )
    if title:
        ax.set_title(title)
    ax.set_xlabel("To Pitch")
    ax.set_ylabel("From Pitch")


def plot_transition_densities(trace, matrix_name="A", alphabet="abcdefghjklmnopqrs"):
    labels = list(alphabet)
    v_size = len(labels)
    data = _get_stacked_samples(trace, matrix_name)
    bins = np.linspace(0, 1, 40)

    fig, axes = plt.subplots(v_size, v_size, figsize=(18, 18), sharex=True, sharey=True)

    for i in range(v_size):
        for j in range(v_size):
            ax = axes[i, j]
            _draw_density_histogram(ax, data[i, j, :], bins)
            _format_grid_cell_axes(ax, i, j, v_size, labels)

    plt.suptitle(f"Posterior Transition Densities ({matrix_name})\nRows: From | Cols: To", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_mean_heatmap(trace, matrix_name="A_0", alphabet="abcdefghjklmnopqrs"):
    labels = list(alphabet)
    posterior_mean = _get_posterior_mean(trace, matrix_name)

    fig, ax = plt.subplots(figsize=(10, 8))
    _draw_heatmap(ax, posterior_mean, labels, title=f"Posterior Mean Transition Matrix ({matrix_name})")
    plt.show()


def plot_mode_heatmaps(trace, matrix_name="A", alphabet="abcdefghjklmnopqrs"):
    labels = list(alphabet)
    posterior_mean = _get_posterior_mean(trace, matrix_name)

    num_modes = posterior_mean.shape[0]
    cols = 4
    rows = math.ceil(num_modes / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()

    for i in range(num_modes):
        _draw_heatmap(axes[i], posterior_mean[i], labels, title=f"Mode {i}")

    for j in range(num_modes, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()