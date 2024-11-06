import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import littlefish.core.utilities as util


def plot_brain(
    input_brain,
    plot_axis=None,
    cmap="RdBu_r",
    bl_range=(-0.1, 0.1),
    ca_lw_cap=5.0,
    ca_range=(-1.0, 1.0),
    neuron_kws=None,
    connection_kws=None,
    layer_label_kws=None,
    neuron_label_kws=None,
):
    """
    plot the neurons and connections of a given brain, neuron color indicates its baseline firing rate, connection
    color indicates connection amplitude

    :param input_brain: littlefish.fish.Brain object
    :param plot_axis: matplotlib.Axes object
    :param cmap: matplotlib.cm object
    :param bl_range: tuple of two floats, plotting range of neuron baseline firing rate, to be mapped to the cmap
    :param ca_lw_cap: positive float, cap line width to indicating connection amplitude
    :param ca_range: tuple of two floats, plotting range of connection amplitude, to be mapped to the cmap and
                     line width
    :param neuron_kws: dict, keyword arguments to plot neuron, matplotlib.Axis.plot inputs
    :param connection_kws: dict, keyword arguments to plot connection, matplotl.Axis.plot inputs
    :param layer_label_kws: dict, keyword arguments, to print text
    :return:
    """

    if neuron_kws is None:
        neuron_kws = {"mew": 4, "ms": 80.0}

    if layer_label_kws is None:
        layer_label_kws = {
            "fontsize": 15,
            "va": "center",
            "ha": "center",
            "color": "#333333",
            "weight": "semibold",
        }

    if neuron_label_kws is None:
        neuron_label_kws = {
            "fontsize": 15,
            "va": "center",
            "ha": "center",
            "color": "#dddddd",
            "family": "monospace",
            "weight": "semibold",
        }

    if connection_kws is None:
        connection_kws = {}

    if plot_axis is None:
        f = plt.figure(figsize=(10, 8))
        plot_axis = f.add_axes([0.05, 0.05, 0.9, 0.9])

    plot_axis.get_figure().set_facecolor("#808080")

    layer_num = input_brain.layer_num
    layer_x_pos = np.linspace(0.1, 0.9, layer_num, endpoint=True)
    neurons_df = input_brain.get_neurons().copy()

    neuron_y_pos = []
    for layer_ind in range(layer_num):
        neuron_num = len(neurons_df[neurons_df["layer"] == float(layer_ind)])
        neuron_y_pos.append(np.linspace(0.85, 0.1, neuron_num, endpoint=True))

    xy_pos = [
        [[layer_x_pos[l], neuron_y_pos[l][n]] for n in range(len(neuron_y_pos[l]))]
        for l in range(layer_num)
    ]
    xy_pos = np.concatenate([np.array(p) for p in xy_pos], axis=0)
    neurons_df["plot_y"] = xy_pos[:, 1]
    neurons_df["plot_x"] = xy_pos[:, 0]

    # print layer type
    for layer in range(layer_num):
        layer_type = input_brain.get_layer_type(layer)
        plot_axis.text(layer_x_pos[layer], 0.92, layer_type.upper(), **layer_label_kws)

    # plot connections
    for gap, gap_conn in input_brain._connections.items():
        for i in range(len(gap_conn.index)):
            for j in range(len(gap_conn.columns)):
                pre_ind = gap_conn.columns[j]
                post_ind = gap_conn.index[i]
                curr_conn = gap_conn.iloc[i, j]
                pre_x = neurons_df.loc[pre_ind, "plot_x"]
                pre_y = neurons_df.loc[pre_ind, "plot_y"]
                post_x = neurons_df.loc[post_ind, "plot_x"]
                post_y = neurons_df.loc[post_ind, "plot_y"]
                amp_raw = curr_conn.get_amplitude()
                amp_n = util.normalized_to_range(
                    var=amp_raw, input_range=ca_range, output_range=(0.0, 1.0)
                )
                amp_c = util.value_2_rgb(value=amp_n, cmap=cmap)
                amp_lw = util.normalized_to_range(
                    var=abs(amp_raw),
                    input_range=[0, max([abs(c) for c in ca_range])],
                    output_range=[0.0, ca_lw_cap],
                )
                # plot_axis.plot([pre_x, post_x], [pre_y, post_y], color=amp_c, lw=amp_lw, **connection_kws)
                plot_axis.plot(
                    [pre_x, post_x],
                    [pre_y, post_y],
                    color=amp_c,
                    lw=1.5,
                    **connection_kws,
                )

    # plotting neurons
    for neuron_ind, neuron in neurons_df.iterrows():
        bl = util.normalized_to_range(
            var=neuron["neuron"].get_baseline_rate(),
            input_range=bl_range,
            output_range=(0.0, 1.0),
        )
        bl_c = util.value_2_rgb(value=bl, cmap="cool")
        if neuron["neuron"].get_neuron_type() == "neuron":
            plot_axis.plot(
                neuron["plot_x"],
                neuron["plot_y"],
                ".",
                mfc=bl_c,
                mec="#444444",
                **neuron_kws,
            )
        elif neuron["neuron"].get_neuron_type() == "muscle":
            plot_axis.plot(
                neuron["plot_x"],
                neuron["plot_y"],
                ".",
                mfc=bl_c,
                mec="#444444",
                **neuron_kws,
            )
            plot_axis.text(
                neuron["plot_x"],
                neuron["plot_y"],
                util.short(neuron["neuron"].get_direction()),
                **neuron_label_kws,
            )
        elif neuron["neuron"].get_neuron_type() == "eye":
            if neuron["neuron"].get_input_type() == "terrain":
                plot_axis.plot(
                    neuron["plot_x"],
                    neuron["plot_y"],
                    ".",
                    mfc=bl_c,
                    mec="#065535",
                    **neuron_kws,
                )
            elif neuron["neuron"].get_input_type() == "food":
                plot_axis.plot(
                    neuron["plot_x"],
                    neuron["plot_y"],
                    ".",
                    mfc=bl_c,
                    mec="#800000",
                    **neuron_kws,
                )
            plot_axis.text(
                neuron["plot_x"],
                neuron["plot_y"],
                util.short(neuron["neuron"].get_direction()),
                **neuron_label_kws,
            )

    plot_axis.set_axis_off()
    plot_axis.set_xlim([0, 1])
    plot_axis.set_ylim([0, 1])

    return plot_axis


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
            life_spans.append(curr_f[sim_n]["simulation_log/last_time_point"][()])
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
    bin_centers = (bin_edges[:-1] + bin_width / 2.0).astype(np.int32)

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
    life_spans = []

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
            curr_fn = os.path.join(curr_folder, fish_fn)
            ff = h5py.File(curr_fn, "r")

            sim_n = [s for s in ff.keys() if s[:11] == "simulation_"]
            if len(sim_n) == 0:
                continue
            elif len(sim_n) > 1:
                print(
                    f"{gen_folder}/{fish_fn} has more than one simulations, take the first one."
                )
            sim_n = sim_n[0]

            fish_names.append(ff["fish/name"][()])
            generations.append(curr_gen)
            life_spans.append(ff[sim_n]["simulation_log/last_time_point"][()])

    life_span_df = pd.DataFrame()
    life_span_df["generation"] = generations
    life_span_df["fish_name"] = fish_names
    life_span_df["life_span"] = life_spans

    return life_span_df


def plot_simulation_life_spans(
    life_span_df: pd.DataFrame,
    ax: plt.Axes = None,
    bins: int = 60,
    max_life_span: int = 30000,
    cmap: str = "cool",
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
            bin_centers = (bin_edges[:-1] + bin_width / 2.0).astype(np.int32)

        # ax.bar(bin_centers, values, width=bin_width, color="none", ec=color, **kwargs)
        ax.step(
            bin_centers,
            values,
            where="mid",
            color=color,
            label=f"gen{gen:04d}",
            **kwargs,
        )


if __name__ == "__main__":
    simulation_folder = r"F:\little_fish_simulation_logs_4"
    life_span_df = collect_life_spans(
        simulation_folder,
        min_generation=0,
        max_generation=24,
    )
    f, ax = plt.subplots()
    plot_simulation_life_spans(
        life_span_df,
        ax,
        max_life_span=20000,
        bins=50,
    )
    ax.legend()
    plt.show()
