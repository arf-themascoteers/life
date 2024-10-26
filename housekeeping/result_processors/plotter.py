import time
import accumulate_results
import pandas as pd
import matplotlib.pyplot as plt
import os


ALGS = {
    "bnc": "BS-Net-Classifier [9]",
    "c1": "BS-DSC",
    "all": "All Bands",
    "mcuve": "MCUVE [17]",
    "bsnet": "BS-Net-FC [2]",
    "pcal": "PCAL [16]",
    "bsdr": "BSDR",
    "linspacer": "Linearly Spaced",
    "random": "Randomly Selected",
}

DSS = {
    "indian_pines": "Indian Pines",
    "paviaU": "Pavia University",
    "salinas": "Salinas",
    "ghisaconus": "Ghisaconus",
    "lucas_lc0_s_r": "LUCAS (land cover)",
    "lucas_r": "LUCAS (SOC)",
    "lucas_texture_r": "LUCAS (texture)",
}

FIXED_ALG_COLORS = {
    "bnc": "#1f77b4",
    "c1": "#d62728",
    "all": "#2ca02c",
    "mcuve": "#ff7f0e",
    "bsnet": "#008000",
    "pcal": "#9467bd",
    "bsdr": "#7FFF00",
    "linspacer": "#FF00FF",
    "random": "#d6ff28",
}

ARBITRARY_ALG_COLORS = ["#000000","#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
MARKERS = ['s', 'P', 'D', '^', 'o', '*', '.', 's', 'P', 'D', '^', 'o', '*', '.']
METRIC_LABELS = ["OA", "AA", r"$\kappa$"]
ALG_ORDERS = ["all", "random", "linspacer", "pcal", "mcuve", "bsnet", "bnc", "c1", "bsdr"]


def plot_algorithm(ax, algorithm, algorithm_index, metric, alg_df):
    algorithm_label = algorithm
    if algorithm in ALGS:
        algorithm_label = ALGS[algorithm]

    alg_df = alg_df.sort_values(by='target_size')
    linestyle = "-"
    if algorithm in FIXED_ALG_COLORS:
        color = FIXED_ALG_COLORS[algorithm]
    else:
        color = ARBITRARY_ALG_COLORS[algorithm_index]

    marker = MARKERS[algorithm_index]
    if algorithm == "all":
        oa = alg_df.iloc[0]["oa"]
        aa = alg_df.iloc[0]["aa"]
        k = alg_df.iloc[0]["k"]
        alg_df = pd.DataFrame(
            {'target_size': range(5, 31), 'oa': [oa] * 26, 'aa': [aa] * 26, 'k': [k] * 26})
        linestyle = "--"
        color = "#000000"
        marker = None
    ax.plot(alg_df['target_size'], alg_df[metric],
                                     label=algorithm_label,
                                     color=color,
                                     fillstyle='none', markersize=7, linewidth=2, linestyle=linestyle)


def plot_metric(include, metric, metric_index, dataset_index, dataset, ddf, ax):
    min_lim = min(ddf["oa"].min(), ddf["aa"].min(), ddf["k"].min()) - 0.02
    max_lim = max(ddf["oa"].max(), ddf["aa"].max(), ddf["k"].max()) + 0.02
    for algorithm_index, algorithm in enumerate(include):
        alg_df = ddf[ddf["algorithm"] == algorithm]
        plot_algorithm(ax, algorithm, algorithm_index, metric, alg_df)

    ax.set_xlabel('Target size', fontsize=18)
    ax.set_ylabel(METRIC_LABELS[metric_index], fontsize=18)
    ax.set_ylim(min_lim, max_lim)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, linestyle='-', alpha=0.6)

    if metric_index == 0 and dataset_index == 0:
        # legend = ax.legend(loc='upper left', fontsize=12, ncols=6,
        #                    bbox_to_anchor=(0, 1.35),
        #                    columnspacing=3.8, frameon=True)
        legend = ax.legend(loc='upper left', ncols=5,bbox_to_anchor=(0, 1.3))
        legend.get_title().set_fontsize('12')
        legend.get_title().set_fontweight('bold')

    if metric_index == 1:
        ax.set_title(DSS[dataset], fontsize=20)


def plot_combined(sources=None,exclude=None):
    if exclude is None:
        exclude = []
    if sources is None:
        sources = os.listdir("../../saved_results")
    graphics_folder = "../../saved_graphics"
    os.makedirs(graphics_folder, exist_ok=True)
    dest = f"image_{int(time.time())}.png"
    dest = os.path.join(graphics_folder, dest)
    df = accumulate_results.accumulate_results(sources,excluded=exclude)
    datasets = df["dataset"].unique()
    fig, axes = plt.subplots(nrows=len(datasets), ncols=3, figsize=(18,36))
    for dataset_index, dataset in enumerate(datasets):
        ddf = df[df["dataset"] == dataset].copy()
        if len(ddf) == 0:
            continue

        ddf["sort_order"] = ddf["algorithm"].apply(lambda x: ALG_ORDERS.index(x) if x in ALG_ORDERS else len(ALG_ORDERS) + ord(x[0]))
        ddf = ddf.sort_values("sort_order").drop(columns=["sort_order"])

        algorithms = ddf["algorithm"].unique()
        include = [x for x in algorithms if x not in exclude]
        if len(include) == 0:
            continue

        for metric_index, metric in enumerate(["oa", "aa", "k"]):
            if len(axes.shape) == 1:
                ax = axes[metric_index]
            else:
                ax = axes[dataset_index, metric_index]
            plot_metric(include, metric, metric_index, dataset_index, dataset, ddf, ax)

    #fig.tight_layout()
    #fig.subplots_adjust(wspace=0.3, hspace=0.5, top=0.95, bottom=0.15)
    plt.savefig(dest, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


if __name__ == "__main__":
    plot_combined()
