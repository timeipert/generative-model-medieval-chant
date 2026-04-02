import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy


def extract_string_intervals(seq_string):
    return [ord(b) - ord(a) for a, b in zip(seq_string, seq_string[1:])]


def extract_pymc_intervals(data, reverse_pymc=True):
    if hasattr(data, "prior_predictive"):
        data = data.prior_predictive
    elif hasattr(data, "posterior_predictive"):
        data = data.posterior_predictive

    try:
        x0 = data["x_0"]
        xt = data["x_t+1"]

        if hasattr(x0, "to_series"):
            s0 = x0.to_series()
            st = xt.to_series()
        else:
            s0 = pd.Series(np.asarray(x0).flatten())
            st = pd.Series(np.asarray(xt).flatten())

        seq = pd.concat([s0, st], ignore_index=True)
        if reverse_pymc:
            seq = seq.iloc[::-1]

        return [int(seq.iloc[i + 1] - seq.iloc[i]) for i in range(len(seq) - 1)]
    except Exception:
        return []


def get_all_intervals(data, reverse_pymc=True):
    if isinstance(data, list):
        intervals = []
        for item in data:
            intervals.extend(get_all_intervals(item, reverse_pymc))
        return intervals

    if isinstance(data, str):
        return extract_string_intervals(data)

    if isinstance(data, pd.Series):
        intervals = []
        for item in data.dropna():
            intervals.extend(get_all_intervals(item, reverse_pymc))
        return intervals

    if hasattr(data, "prior_predictive") or hasattr(data, "posterior_predictive") or (
            hasattr(data, "__getitem__") and "x_0" in data):
        return extract_pymc_intervals(data, reverse_pymc)

    return []


def get_pmf(intervals, support, alpha=1e-6):
    counts = pd.Series(intervals).value_counts(normalize=True)
    pmf = np.array([counts.get(val, 0) + alpha for val in support])
    return pmf / pmf.sum()


def bootstrap_kl(data_p, data_q, n_iterations=100):
    intervals_p = np.array(get_all_intervals(data_p))
    intervals_q = np.array(get_all_intervals(data_q))
    support = sorted(set(intervals_p) | set(intervals_q))

    kl_values = []
    for _ in range(n_iterations):
        samp_p = np.random.choice(intervals_p, size=len(intervals_p), replace=True)
        samp_q = np.random.choice(intervals_q, size=len(intervals_q), replace=True)

        pmf_p = get_pmf(samp_p, support)
        pmf_q = get_pmf(samp_q, support)
        kl_values.append(entropy(pmf_p, pmf_q))

    return kl_values


def bootstrap_internal_kl(data, n_iterations=100):
    intervals = np.array(get_all_intervals(data))
    support = sorted(set(intervals))
    kl_values = []
    half = len(intervals) // 2

    for _ in range(n_iterations):
        samp = np.random.choice(intervals, size=len(intervals), replace=True)
        pmf_p = get_pmf(samp[:half], support)
        pmf_q = get_pmf(samp[half:], support)
        kl_values.append(entropy(pmf_p, pmf_q))

    return kl_values


def get_bootstrapped_distribution(data, n_iterations=100, normalize=False):
    intervals = np.array(get_all_intervals(data))
    support = sorted(set(intervals))

    records = []
    for i in range(n_iterations):
        samp = np.random.choice(intervals, size=len(intervals), replace=True)
        counts = pd.Series(samp).value_counts(normalize=normalize)
        for val in support:
            records.append({
                "Interval": val,
                "Value": counts.get(val, 0),
                "Iteration": i
            })
    return pd.DataFrame(records)


def calculate_kl_divergence(data_p, data_q):
    intervals_p = get_all_intervals(data_p)
    intervals_q = get_all_intervals(data_q)

    support = sorted(set(intervals_p + intervals_q))
    pmf_p = get_pmf(intervals_p, support)
    pmf_q = get_pmf(intervals_q, support)

    return entropy(pmf_p, pmf_q)


def plot_interval_histogram(data, title="Frequency of Pitch Intervals", normalize=False, y_limit=None,
                            n_iterations=100, save=None):
    palette = sns.color_palette("pastel").as_hex()

    if isinstance(data, list) and len(data) > 0 and not isinstance(data[0], (int, float, str, np.integer)):
        num_plots = len(data)
        cols = 3 if num_plots > 1 else 1
        rows = math.ceil(num_plots / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows), squeeze=False)
        axes = axes.flatten()

        titles = title if isinstance(title, list) else [f"{title} {i + 1}" for i in range(num_plots)]

        for i in range(num_plots):
            df_dist = get_bootstrapped_distribution(data[i], n_iterations, normalize)
            current_color = palette[i % len(palette)]

            axes[i].grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
            sns.barplot(
                data=df_dist,
                x="Interval",
                y="Value",
                ax=axes[i],
                color=current_color,
                edgecolor="#555555",
                errorbar="sd",
                capsize=0.15,
                err_kws={'linewidth': 1.2, 'color': '#444444'},
                zorder=3
            )

            axes[i].set_title(titles[i] if i < len(titles) else "", pad=15, fontweight="bold", color="#333333")
            axes[i].tick_params(axis='x', rotation=45, colors="#333333")
            axes[i].tick_params(axis='y', colors="#333333")
            axes[i].set_ylabel("Probability" if normalize else "Count", fontweight="bold", color="#333333")
            axes[i].set_xlabel("Pitch Interval", fontweight="bold", color="#333333")

            if y_limit:
                axes[i].set_ylim(y_limit)
            elif normalize:
                axes[i].set_ylim(0, 1)

            sns.despine(ax=axes[i], trim=True)

        for j in range(num_plots, len(axes)):
            fig.delaxes(axes[j])
    else:
        df_dist = get_bootstrapped_distribution(data, n_iterations, normalize)
        plt.figure(figsize=(12, 6))

        ax = plt.gca()
        ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

        sns.barplot(
            data=df_dist,
            x="Interval",
            y="Value",
            color=palette[0],
            edgecolor="#555555",
            errorbar="sd",
            capsize=0.15,
            err_kws={'linewidth': 1.2, 'color': '#444444'},
            zorder=3
        )

        plt.xticks(rotation=45, ha="right", color="#333333")
        plt.yticks(color="#333333")
        plt.title(title if isinstance(title, str) else title[0], pad=15, fontweight="bold", color="#333333")
        plt.ylabel("Probability" if normalize else "Count", fontweight="bold", color="#333333")
        plt.xlabel("Pitch Interval", fontweight="bold", color="#333333")

        if y_limit:
            plt.ylim(y_limit)
        elif normalize:
            plt.ylim(0, 1)

        sns.despine(trim=True)

    plt.tight_layout()
    if save:
        plt.savefig(save)
    plt.show()



def plot_kl_with_context(pi_data, models_dict, title="KL Divergence Analysis", n_iterations=100, save=None):
    records = []

    for name, samples in models_dict.items():
        kl_vals = bootstrap_kl(pi_data, samples, n_iterations)
        for val in kl_vals:
            records.append({"Source": name, "KL": val, "Type": "Model"})

    #kl_internal_vals = bootstrap_internal_kl(pi_data, n_iterations)
    #for val in kl_internal_vals:
    #    records.append({"Source": "Internal Variance", "KL": val, "Type": "Baseline"})

    df = pd.DataFrame(records)

    #mean_internal = df[df["Source"] == "Internal Variance"]["KL"].mean()

    order_models = df[df["Type"] == "Model"].groupby("Source")["KL"].mean().sort_values().index.tolist()
    order = order_models# + ["Internal Variance"]

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

    base_palette = sns.color_palette("pastel").as_hex()
    colors = []
    for i, src in enumerate(order):
        if src == "Internal Variance":
            colors.append("#E0E0E0")
        else:
            colors.append(base_palette[i % len(base_palette)])

    sns.barplot(
        x="Source",
        y="KL",
        hue="Source",
        legend=False,
        data=df,
        order=order,
        palette=colors,
        edgecolor="#555555",
        errorbar="sd",
        capsize=0.1,
        err_kws={'linewidth': 1.2, 'color': '#444444'},
        zorder=3
    )

    #plt.axhline(mean_internal, color="#888888", linestyle=":", linewidth=2, alpha=0.8, zorder=4)

    means = df.groupby("Source")["KL"].mean()
    stds = df.groupby("Source")["KL"].std()

    for i, patch in enumerate(ax.patches):
        src = order[i]
        mean_val = means[src]
        std_val = stds[src]

        y_pos = mean_val + std_val

        ax.annotate(f'$\mu={mean_val:.3f}$\n$\pm{std_val:.3f}$',
                    (patch.get_x() + patch.get_width() / 2., y_pos),
                    ha='center', va='bottom', fontsize=10, color="#333333",
                    xytext=(0, 5), textcoords='offset points', zorder=5)

    plt.title(title, pad=20, fontweight="bold", color="#333333")
    plt.ylabel("KL Divergence (bits)", fontweight="bold", color="#333333")
    plt.xlabel("")
    plt.xticks(color="#333333")
    plt.yticks(color="#333333")

    sns.despine(trim=True)

    y_max = ax.get_ylim()[1]
    ax.set_ylim(0, y_max * 1.15)

    plt.tight_layout()
    if save:
        plt.savefig(save)
    plt.show()