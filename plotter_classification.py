import time
import accumulate_results
import pandas as pd
import matplotlib.pyplot as plt
import os
from plot_commons import ALGS, FIXED_ALG_COLORS, ARBITRARY_ALG_COLORS, MARKERS, ALG_ORDERS
from ds_manager import DSManager
import random

CLASSIFICATION_METRIC_LABELS = ["OA", "AA", r"$\kappa$"]

DSS = {
    "indian_pines": "Indian Pines",
    "paviaU": "Pavia University",
    "salinas": "Salinas",
    "ghisaconus": "Ghisaconus",
}

def plot_algorithm(ax, algorithm, props, algorithm_index, metric, alg_df):
    props = int(props)
    algorithm_label = algorithm
    if algorithm in ALGS:
        algorithm_label = ALGS[algorithm]
    if props !=0 :
        algorithm_label = f"{algorithm_label}({props})"
    alg_df = alg_df.sort_values(by='target_size')
    linestyle = "-"
    if algorithm in FIXED_ALG_COLORS:
        color = FIXED_ALG_COLORS[algorithm]
    else:
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))#ARBITRARY_ALG_COLORS[algorithm_index]

    if algorithm_index > len(MARKERS) - 1:
        marker = "--"
    else:
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


def plot_metric(algorithms, propses, metric, metric_index, dataset_index, dataset, ddf, ax):
    min_lim = min(ddf["oa"].min(), ddf["aa"].min(), ddf["k"].min()) - 0.02
    max_lim = max(ddf["oa"].max(), ddf["aa"].max(), ddf["k"].max()) + 0.02
    for algorithm_index, algorithm in enumerate(algorithms):
        props = propses[algorithm_index]
        alg_df = ddf[(ddf["algorithm"] == algorithm) & (ddf["props"] == props)]
        plot_algorithm(ax, algorithm, props, algorithm_index, metric, alg_df)

    ax.set_xlabel('Target size', fontsize=18)
    ax.set_ylabel(CLASSIFICATION_METRIC_LABELS[metric_index], fontsize=18)
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


def plot_combined(sources=None,exclude=None,only_algorithms=None,only_datasets=None,pending=False):
    resource = "saved_results"
    if pending:
        resource = "results"
    if exclude is None:
        exclude = []
    if sources is None:
        sources = os.listdir(resource)
    graphics_folder = "saved_graphics"
    os.makedirs(graphics_folder, exist_ok=True)
    dest = f"image_{int(time.time())}.png"
    dest = os.path.join(graphics_folder, dest)
    df = accumulate_results.accumulate_results(sources, excluded=exclude, pending=pending)
    datasets = df["dataset"].unique()
    datasets = [d for d in datasets if DSManager.is_dataset_classification(d)]
    if only_datasets is not None:
        datasets = [d for d in datasets if d in only_datasets]
    fig, axes = plt.subplots(nrows=len(datasets), ncols=3, figsize=(18,10*len(datasets)))
    for dataset_index, dataset in enumerate(datasets):
        ddf = df[df["dataset"] == dataset].copy()
        if len(ddf) == 0:
            continue

        ddf["sort_order"] = ddf["algorithm"].apply(lambda x: ALG_ORDERS.index(x) if x in ALG_ORDERS else len(ALG_ORDERS) + ord(x[0]))
        ddf = ddf.sort_values(["sort_order","props"]).drop(columns=["sort_order"])

        unique_combinations = df[['algorithm', 'props']].drop_duplicates()

        all_algorithms = unique_combinations["algorithm"]
        all_propses = unique_combinations["props"]
        if only_algorithms is None:
            algorithms = all_algorithms.tolist()
            propses = all_propses.tolist()
        else:
            all_algorithms_list = all_algorithms.tolist()
            all_propses_list = all_propses.tolist()
            algorithms = []
            propses = []
            for index, algorithm in enumerate(all_algorithms_list):
                if algorithm in only_algorithms:
                    algorithms.append(algorithm)
                    propses.append(all_propses_list[index])

        if len(algorithms) == 0:
            continue

        ddf = ddf.merge(unique_combinations, on=['algorithm', 'props'], how='inner').copy()

        for metric_index, metric in enumerate(["oa", "aa", "k"]):
            if len(axes.shape) == 1:
                ax = axes[metric_index]
            else:
                ax = axes[dataset_index, metric_index]
            plot_metric(algorithms, propses, metric, metric_index, dataset_index, dataset, ddf, ax)

    #fig.tight_layout()
    #fig.subplots_adjust(wspace=0.3, hspace=0.5, top=0.95, bottom=0.15)
    plt.savefig(dest, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


if __name__ == "__main__":
    plot_combined(sources=["bsdr","bsdrattn"], only_algorithms=["bsdr","bsdrattn","c1"], only_datasets=["indian_pines"])

