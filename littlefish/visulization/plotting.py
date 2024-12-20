import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from littlefish.log_analysis.simulation_log import get_simulation_logs


def plot_confusion_matrix(
    matrix: np.ndarray,
    ax: plt.Axes = None,
    last_neuron_idx_per_layer: list[int] = None,
    **kwargs,
):
    """
    plot confusion matrix delinated by layers for a brain.

    :param matrix: 2d array, confusion matrix.
    :last_neuron_idx_per_layer: list[int] the index of the last neuron in each layer.
    """

    if ax is None:
        f, ax = plt.subplots()

    ax.imshow(matrix, **kwargs)
    for idx in last_neuron_idx_per_layer:
        ax.axvline(idx + 0.5, color="#aaaaaa", lw=1)
        ax.axhline(idx + 0.5, color="#aaaaaa", lw=1)

    ax.set_xlabel("postsynaptic neuron index")
    ax.set_ylabel("presynaptic neuron index")
    ax.set_aspect("equal")

    return ax


def get_geneartion_life_spans(gen_folder: str) -> pd.DataFrame:
    """
    given a generation folder, return a dataframe with columns
      "name": file name of the simulation of each fish
      "life_span": the life span of the fish in the simulation
    """

    names = []
    life_spans = []

    for fn in os.listdir(gen_folder):
        if fn[0:5] == "fish_" and fn[-5:] == ".hdf5":
            curr_f = h5py.File(os.path.join(gen_folder, fn), "r")
            sim_n = [s for s in curr_f.keys() if s[:11] == "simulation_"]
            if len(sim_n) == 0:
                continue
            elif len(sim_n) > 1:
                print(f"{fn} has more than one simulations, take the first one.")
            sim_n = sim_n[0]
            names.append(fn)
            life_spans.append(curr_f[sim_n]["simulation_cache/last_time_point"][()])
            curr_f.close()

    life_span_df = pd.DataFrame()
    life_span_df["life_span"] = life_spans
    life_span_df["name"] = names
    life_span_df.sort_values(by="life_span", inplace=True)

    return life_span_df


def plot_life_span_distribution(
    life_spans: list,
    ax: plt.Axes,
    bins: int = 60,
    max_life_span: int = 30000,
    **kwargs,
) -> pd.DataFrame:
    """
    Given a list of life spans, plot the distribution.

    :param life_spans: list of non-negative integers, life spans of fishs
    :param ax: plt.Axes, plot axes
    :param bins: int, number of bins
    :param max_life_span: int, the maximum life span to clip to
    :param **kwargs: other parameters to be passed to the ax.bar() function.
    :return: df_plot, pandas.Dataframe, the distribution.
    """

    if ax is None:
        f, ax = plt.subplots(figsize=(7, 5))

    if max_life_span is None:
        max_life_span = max(life_spans)

    life_spans_plot = np.clip(life_spans, 0, max_life_span)
    values, bin_edges = np.histogram(
        life_spans_plot, bins=bins, range=[0, max_life_span]
    )
    bin_width = np.mean(np.diff(bin_edges))
    bin_centers = (bin_edges[:-1] + bin_width / 2.0).astype(int)

    df_plot = pd.DataFrame(
        {
            "fish count": values,
            "life span": bin_centers,
        }
    )

    ax.bar(bin_centers, values, width=bin_width, **kwargs)
    ax.set_xlabel("life span", fontsize=16)
    ax.set_ylabel("fish count", fontsize=16)

    return df_plot


def collect_life_spans(
    simulation_folder: str,
    min_generation: int = None,
    max_generation: int = None,
) -> pd.DataFrame:
    gen_folders = [
        f
        for f in os.listdir(simulation_folder)
        if os.path.isdir(os.path.join(simulation_folder, f))
        and f.startswith("generation")
    ]

    plot_gen_folders = []

    for gen_folder in gen_folders:
        curr_gen = int(gen_folder.split("_")[-1])

        if (min_generation is None or curr_gen >= min_generation) and (
            max_generation is None or curr_gen <= max_generation
        ):
            plot_gen_folders.append(gen_folder)

    generations = []
    fish_names = []
    mean_life_spans = []
    median_life_spans = []
    is_from_last_gen = []

    for gen_i, gen_folder in enumerate(plot_gen_folders):
        curr_gen = int(gen_folder.split("_")[-1])

        print(f"reading {gen_folder}, {gen_i + 1} / {len(plot_gen_folders)} ...")

        curr_folder = os.path.join(simulation_folder, gen_folder)
        fish_fns = [
            f
            for f in os.listdir(curr_folder)
            if f.startswith("fish_") and f.endswith(".hdf5")
        ]

        for fish_fn in fish_fns:
            fish_name = os.path.splitext(fish_fn)[0]
            curr_fn = os.path.join(curr_folder, fish_fn)
            ff = h5py.File(curr_fn, "r")

            if len(ff["generations"]) > 1:
                is_from_last_gen.append(True)
            else:
                is_from_last_gen.append(False)

            # sim_n = [s for s in ff.keys() if s[:11] == "simulation_"]
            # if len(sim_n) == 0:
            #     continue
            # elif len(sim_n) > 1:
            #     print(
            #         f"{gen_folder}/{fish_fn} has more than one simulations, take the first one."
            #     )
            # sim_n = sim_n[0]

            sim_logs = get_simulation_logs(ff)

            fish_names.append(fish_name)
            generations.append(curr_gen)
            mean_life_spans.append(np.mean([s.last_time_point for s in sim_logs]))
            median_life_spans.append(np.median([s.last_time_point for s in sim_logs]))

    life_span_df = pd.DataFrame()
    life_span_df["generation"] = generations
    life_span_df["fish_name"] = fish_names
    life_span_df["mean_life_span"] = mean_life_spans
    life_span_df["median_life_span"] = median_life_spans
    life_span_df["is_from_last_geneartion"] = is_from_last_gen

    return life_span_df


def plot_simulation_life_spans(
    life_span_df: pd.DataFrame,
    ax: plt.Axes = None,
    bins: int = 60,
    max_life_span: int = 30000,
    cmap: str = "cool",
    legend_gap: int = 10,
    **kwargs,
):
    if max_life_span is None:
        max_life_span = max(life_span_df["life_span"])

    bin_width = None
    bin_centers = None
    cmap = plt.get_cmap(cmap)

    if ax is None:
        f, ax = plt.subplot(figsize=(7, 4))

    gens = sorted(life_span_df["generation"].unique())
    for gen_i, gen in enumerate(gens):
        color = mcolors.to_hex(cmap(float(gen_i) / (len(gens) - 1)))

        curr_ls = life_span_df.query("generation == @gen")["life_span"]
        curr_ls = curr_ls.clip(0, max_life_span)
        values, bin_edges = np.histogram(curr_ls, bins=bins, range=[0, max_life_span])

        if bin_width is None:
            bin_width = np.mean(np.diff(bin_edges))

        if bin_centers is None:
            bin_centers = (bin_edges[:-1] + bin_width / 2.0).astype(int)

        if gen_i % legend_gap == 0:
            legend = f"gen{gen:04d}"
        else:
            legend = None

        ax.step(
            bin_centers,
            values,
            where="mid",
            color=color,
            label=legend,
            **kwargs,
        )


if __name__ == "__main__":
    # simulation_folder = r"F:\little_fish_simulation_logs_4"
    # life_span_df = collect_life_spans(
    #     simulation_folder,
    #     min_generation=0,
    #     max_generation=24,
    # )
    # f, ax = plt.subplots()
    # plot_simulation_life_spans(
    #     life_span_df,
    #     ax,
    #     max_life_span=20000,
    #     bins=50,
    # )
    # ax.legend()
    # plt.show()

    mat = np.arange(64).reshape(8, 8)
    plot_confusion_matrix(
        mat,
        ax=None,
        last_neuron_idx_per_layer=[1, 4, 6],
        cmap="magma",
    )
    plt.show()
