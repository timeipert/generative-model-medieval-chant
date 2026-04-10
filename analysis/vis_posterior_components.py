import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from pathlib import Path


def _generate_pitch_labels(v_size):
    base_notes = ["A", "B", "C", "D", "E", "F", "G"]
    labels = []
    for i in range(v_size):
        octave = i // 7
        note = base_notes[i % 7]
        if octave == 0:
            labels.append(note)
        else:
            labels.append(note.lower() * octave)
    return labels


def _extract_summary_stats(idata, var_name):
    posterior = idata.posterior[var_name]
    mean_vals = posterior.mean(dim=["chain", "draw"]).values
    std_vals = posterior.std(dim=["chain", "draw"]).values
    return mean_vals, std_vals


def plot_rule_weights(idata):
    expected_weights = ["w_interval", "w_register", "w_repeat", "w_direction"]
    present_weights = [w for w in expected_weights if w in idata.posterior]

    fig, ax = plt.subplots(figsize=(10, len(present_weights) * 1.5 + 2))

    az.plot_forest(
        idata,
        var_names=present_weights,
        combined=True,
        hdi_prob=0.94,
        ax=ax,
        colors="teal"
    )

    ax.set_title("Posterior Distributions of Musical Rule Weights", fontsize=14)
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Weight Value\n(Positive = Rule applies strongly | Negative = Inverse rule applies)")

    plt.tight_layout()
    return fig


def plot_register_center(idata, v_size):
    if "register_center" not in idata.posterior:
        return None

    v_labels = _generate_pitch_labels(v_size)
    samples = idata.posterior["register_center"].values.flatten()

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.kdeplot(samples, fill=True, color="indigo", ax=ax, alpha=0.6)

    mean_val = np.mean(samples)
    hdi = az.hdi(samples, hdi_prob=0.94)

    ax.axvline(mean_val, color="black", linestyle="-", label=f"Mean: {mean_val:.2f}")
    ax.axvline(hdi[0], color="black", linestyle=":", alpha=0.7, label="94% HDI")
    ax.axvline(hdi[1], color="black", linestyle=":", alpha=0.7)

    ax.set_xlim(0, v_size - 1)
    ax.set_xticks(range(v_size))
    ax.set_xticklabels(v_labels)

    ax.set_title("Register Center", fontsize=14)
    ax.set_xlabel("Pitch")
    ax.set_ylabel("Density")
    ax.legend()

    plt.tight_layout()
    return fig


def plot_base_logits(idata, v_size):
    if "base_logits" not in idata.posterior:
        return None

    v_labels = _generate_pitch_labels(v_size)
    mean_vals, _ = _extract_summary_stats(idata, "base_logits")

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        mean_vals,
        cmap="coolwarm",
        center=0,
        xticklabels=v_labels,
        yticklabels=v_labels,
        ax=ax,
        cbar_kws={"label": "Base Preference"}
    )

    ax.set_title("Base Logits", fontsize=14)
    ax.set_xlabel("To Pitch")
    ax.set_ylabel("From Pitch")

    plt.tight_layout()
    return fig


def export_component_visualizations(idata, v_size, base=False, save_dir="visual/componentsII"):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    figs = {
        "rule_weights": plot_rule_weights(idata),
        "register_center": plot_register_center(idata, v_size),

    }
    if base:
        figs["base"] = plot_base_logits(idata, v_size)

    for name, fig in figs.items():
        if fig is not None:
            file_path = save_path / f"{name}.png"
            fig.savefig(file_path, bbox_inches="tight", dpi=200)
            plt.close(fig)

    print(f"plots saved: {save_path.absolute()}")
    return figs