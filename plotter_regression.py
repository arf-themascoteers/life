import time
import accumulate_results
import pandas as pd
import matplotlib.pyplot as plt
import os
from plot_commons import ALGS, FIXED_ALG_COLORS, ARBITRARY_ALG_COLORS, MARKERS, ALG_ORDERS
from ds_manager import DSManager
import random

REGRESSION_METRIC_LABELS = [r"$R^2$", "RMSE"]

DSS = {
    "lucas_r": "LUCAS"
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
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))

    if algorithm_index > len(MARKERS) - 1:
        marker = "--"
    else:
        marker = MARKERS[algorithm_index]
    if algorithm == "all":
        r2 = alg_df.iloc[0]["oa"]
        rmse = alg_df.iloc[0]["aa"]
        alg_df = pd.DataFrame(
            {'target_size': range(5, 31), 'oa': [r2] * 26, 'aa': [rmse] * 26, 'k': [0] * 26})
        linestyle = "--"
        color = "#000000"
        marker = None
    print(algorithm_label, props)
    ax.plot(alg_df['target_size'], alg_df[metric],
                                     label=algorithm_label,
                                     color=color,
                                     fillstyle='none', markersize=7, linewidth=2, linestyle=linestyle)


def plot_metric(algorithms, propses, metric, metric_index, dataset_index, dataset, ddf, ax):
    ddf[metric] = ddf[metric].clip(lower=0)
    min_lim = max(ddf[metric].min() - 0.02,0)
    max_lim = min(ddf[metric].max() + 0.02,1)

    for algorithm_index, algorithm in enumerate(algorithms):
        props = propses[algorithm_index]
        alg_df = ddf[(ddf["algorithm"] == algorithm) & (ddf["props"] == props)]
        plot_algorithm(ax, algorithm, props, algorithm_index, metric, alg_df)

    ax.set_xlabel('Target size', fontsize=18)
    ax.set_ylabel(REGRESSION_METRIC_LABELS[metric_index], fontsize=18)
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
        ax.set_title(DSS[dataset], fontsize=20, x=-0.1,y=1)


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
    datasets = [d for d in datasets if not DSManager.is_dataset_classification(d)]
    if only_datasets is not None:
        datasets = [d for d in datasets if d in only_datasets]
    fig, axes = plt.subplots(nrows=len(datasets), ncols=2, figsize=(18,6*len(datasets)))
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
            algorithms = all_algorithms
            propses = all_propses
        else:
            algorithms = []
            propses = []
            for index, algorithm in enumerate(all_algorithms):
                if algorithm in only_algorithms:
                    algorithms.append(algorithm)
                    propses.append(all_propses[index])

        if len(algorithms) == 0:
            continue

        for metric_index, metric in enumerate(["oa", "aa"]):
            if len(axes.shape) == 1:
                ax = axes[metric_index]
            else:
                ax = axes[dataset_index, metric_index]
            plot_metric(algorithms, propses, metric, metric_index, dataset_index, dataset, ddf, ax)

    plt.savefig(dest, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


if __name__ == "__main__":
    plot_combined(sources=["r1"])

