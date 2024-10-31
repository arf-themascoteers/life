import time
import accumulate_results
import pandas as pd
import matplotlib.pyplot as plt
import os
from plot_commons import ALGS, FIXED_ALG_COLORS, ARBITRARY_ALG_COLORS, MARKERS, ALG_ORDERS
from ds_manager import DSManager
import random
from matplotlib.ticker import LogLocator, LogFormatter

DSS = {
    "indian_pines": "Indian Pines",
    "paviaU": "Pavia University",
    "salinas": "Salinas",
    "ghisaconus": "Ghisaconus",
    "lucas_r": "LUCAS"
}

def plot_algorithm(ax, algorithm, props, algorithm_index, alg_df):
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
    ax.plot(alg_df['target_size'], alg_df["time"],
                                     label=algorithm_label,
                                     color=color,
                                     fillstyle='none', markersize=7, linewidth=2, linestyle=linestyle)


def plot_metric(algorithms, propses, dataset_index, dataset, ddf, ax):
    ax.set_xlabel('Target size', fontsize=18)
    ax.set_ylabel('Time', fontsize=18)
    ax.set_yscale("log",base=10)

    for algorithm_index, algorithm in enumerate(algorithms):
        props = propses[algorithm_index]
        alg_df = ddf[(ddf["algorithm"] == algorithm) & (ddf["props"] == props)]
        plot_algorithm(ax, algorithm, props, algorithm_index, alg_df)

    # ax.yaxis.set_major_locator(LogLocator(base=10))
    # ax.yaxis.set_major_formatter(LogFormatter(base=10))
    #ax.tick_params(axis='both', which='major', labelsize=14)
    #ax.grid(True, linestyle='-', alpha=0.6)

    if dataset_index == 0:
        legend = ax.legend(loc='upper left', ncols=5,bbox_to_anchor=(0, 1.3))
        legend.get_title().set_fontsize('12')
        legend.get_title().set_fontweight('bold')
    ax.set_title(DSS[dataset], fontsize=20)


def plot(sources=None,exclude=None,only_algorithms=None,only_datasets=None,pending=False, reg_or_class="both"):
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

    if reg_or_class == "regression":
        datasets = [d for d in datasets if d in ["lucas_r"]]
    elif reg_or_class == "classification":
        datasets = [d for d in datasets if d in list(DSS.keys()).remove("lucas_r")]

    fig, axes = plt.subplots(nrows=len(datasets), ncols=1, figsize=(18,10*len(datasets)))
    for dataset_index, dataset in enumerate(datasets):
        ddf = df[df["dataset"] == dataset].copy().reset_index(drop=True)
        ddf = ddf[ddf["dataset"] != "all"].copy().reset_index(drop=True)
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
            algorithms = []
            propses = []
            for index, algorithm in enumerate(all_algorithms):
                if algorithm in only_algorithms:
                    algorithms.append(algorithm)
                    propses.append(all_propses[index])

        if len(algorithms) == 0:
            continue

        ddf = ddf.merge(unique_combinations, on=['algorithm', 'props'], how='inner').copy().reset_index(drop=True)

        ax = axes[dataset_index]
        plot_metric(algorithms, propses, dataset_index, dataset, ddf, ax)

    plt.savefig(dest, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


