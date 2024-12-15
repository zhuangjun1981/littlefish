import h5py
import yaml
from typing import Union
import pandas as pd
import matplotlib.pyplot as plt
from littlefish.core import utilities as util
from littlefish.visulization.plotting import plot_confusion_matrix
from littlefish.brain.neuron import (
    Neuron,
    Eye,
    Muscle,
)
from littlefish.brain.layer_sets import *
from littlefish.brain.connection import Connection
from littlefish.brain.brain import Brain


def load_neuron_from_h5_group(h5_group: h5py.Group) -> Union[Neuron, Eye, Muscle]:
    neuron_type = util.decode(h5_group["type"][()])
    neuron_type = neuron_type.split(".")[-1]
    kv_pairs = {}
    for k, v in h5_group.items():
        if k != "type":
            value = v[()]
            if isinstance(value, bytes):
                value = util.decode(value)
            kv_pairs[k] = value
    obj = eval(neuron_type)
    return obj(**kv_pairs)


def load_brain_from_h5_group(
    h5_group: h5py.Group,
    should_load_simulation_cache: bool = False,
) -> Brain:
    """
    load Brain object from h5 group.
    """
    grp_neurons = h5_group["neurons"]
    layers = grp_neurons.attrs["layers"]
    neuron_names = sorted(list(grp_neurons.keys()))
    neurons = []
    for neuron_name in neuron_names:
        neurons.append(load_neuron_from_h5_group(grp_neurons[neuron_name]))
    neurons = pd.DataFrame({"layer": layers, "neuron": neurons})

    grp_connection = h5_group["connections"]
    amplitudes = grp_connection["amplitudes"][()]
    conn_mat = grp_connection["connection_matrix"][()]
    assert len(amplitudes) == conn_mat.shape[0]
    assert conn_mat.shape[1] == 5

    pre_idxs = conn_mat[:, 0]
    post_idxs = conn_mat[:, 1]
    connections = []
    for conn_i, amplitude in enumerate(amplitudes):
        connections.append(
            Connection(
                latency=conn_mat[conn_i, 2],
                amplitude=amplitude,
                rise_time=conn_mat[conn_i, 3],
                decay_time=conn_mat[conn_i, 4],
            )
        )
    connections = pd.DataFrame(
        {"pre_idx": pre_idxs, "post_idx": post_idxs, "connection": connections}
    )

    brain = Brain(neurons=neurons, connections=connections)

    if should_load_simulation_cache:
        if "simulation_cache" in h5_group:
            grp_sim_cache = h5_group["simulation_cache"]

            if "action_histories" in grp_sim_cache and "psp_waveforms" in grp_sim_cache:
                grp_action_histries = grp_sim_cache["action_histries"]
                neuron_names = sorted(list(grp_action_histries.keys()))
                action_histories = []
                for neuron_name in neuron_names:
                    action_histories.append(list(grp_action_histries[neuron_name][()]))

                psp_wavefroms = grp_sim_cache["psp_wavefroms"][()]

                brain.simulation_cache = {
                    "action_histories": action_histories,
                    "psp_waveforms": psp_wavefroms,
                }

    return brain


def generate_brain_from_brain_config(
    brain_config: dict = None,
    brain_config_path: str = None,
):
    # load brain_config dictionary
    if brain_config is None and brain_config_path is None:
        raise LookupError(
            "one of 'brain_config' and 'brain_config_path' should not be None."
        )
    elif brain_config is None and brain_config_path is not None:
        with open(brain_config_path, "r") as f:
            brain_config = yaml.load(f, Loader=yaml.FullLoader)["brain_config"]
    elif brain_config is not None and brain_config_path is None:
        pass
    else:
        raise LookupError(
            "one of 'brain_config' and 'brain_config_path' should be None."
        )

    layers = []
    neurons = []

    curr_layer = 0

    # generate eyes
    eye_set = eval(brain_config["eye_layer"]["eye_set"])
    input_types = brain_config["eye_layer"]["input_types"]
    eye_params = brain_config["eye_layer"].copy()
    eye_params.pop("eye_set")
    eye_params.pop("input_types")
    for input_type in input_types:
        for eye_direction, rf_dict in eye_set.items():
            layers.append(curr_layer)
            curr_eye = Eye(
                eye_direction=eye_direction,
                rf_positions=rf_dict["rf_positions"],
                rf_weights=rf_dict["rf_weights"],
                input_type=input_type,
                **eye_params,
            )
            neurons.append(curr_eye)

    # generate hidden layers
    if "hidden_layers" in brain_config:
        hidden_neuron_numbers = brain_config["hidden_layers"]["neuron_nums"]
        neuron_params = brain_config["hidden_layers"].copy()
        neuron_params.pop("neuron_nums")
        for hidden_neuron_number in hidden_neuron_numbers:
            curr_layer += 1
            for _ in range(hidden_neuron_number):
                layers.append(curr_layer)
                neurons.append(Neuron(**neuron_params))

    # generate muscles
    muscle_set = eval(brain_config["muscle_layer"]["muscle_set"])
    muscle_params = brain_config["muscle_layer"].copy()
    muscle_params.pop("muscle_set")
    curr_layer += 1
    for direction, step_motion in muscle_set:
        layers.append(curr_layer)
        curr_muscle = Muscle(
            direction=direction, step_motion=step_motion, **muscle_params
        )
        neurons.append(curr_muscle)

    neurons = pd.DataFrame({"layer": layers, "neuron": neurons})

    pre_idxs = []
    post_idxs = []
    connections = []
    all_layers = neurons["layer"].unique().tolist()
    for pre_layer in all_layers:
        for post_layer in all_layers:
            connection_group_key = f"connection_{pre_layer}_{post_layer}"

            if connection_group_key in brain_config:
                conn_dict = brain_config[connection_group_key]

                if (
                    conn_dict["connection_type"] == "full"
                ):  # currently only full connection are supported
                    conn_params = conn_dict.copy()
                    conn_params.pop("connection_type")
                    curr_pre_idxs = neurons.query("layer == @pre_layer").index.to_list()
                    curr_post_idxs = neurons.query(
                        "layer == @post_layer"
                    ).index.to_list()

                    for curr_pre_idx in sorted(curr_pre_idxs):
                        for curr_post_idx in sorted(curr_post_idxs):
                            pre_idxs.append(curr_pre_idx)
                            post_idxs.append(curr_post_idx)
                            connections.append(Connection(**conn_params))

    connections = pd.DataFrame(
        {"pre_idx": pre_idxs, "post_idx": post_idxs, "connection": connections}
    )

    return Brain(neurons=neurons, connections=connections)


def plot_brain_connections(brain: Brain):
    """
    plot the confusion matrix of each parameter of all connections in the brain
    """
    n_neurons = brain.neurons.shape[0]
    last_neuron_idx_per_layer = [-1]
    for neuron_i, neuron_row in brain.neurons.iterrows():
        if (
            neuron_i > 0
            and neuron_row["layer"] > brain.neurons.loc[neuron_i - 1, "layer"]
        ):
            last_neuron_idx_per_layer.append(neuron_i - 1)
    last_neuron_idx_per_layer.append(n_neurons - 1)

    f, axs = plt.subplots(2, 2, figsize=(8, 8))
    for param_i, param in enumerate(
        ["amplitude", "latency", "rise_time", "decay_time"]
    ):
        ax = axs[param_i // 2, param_i % 2]
        matrix = np.zeros((n_neurons, n_neurons))
        matrix[:] = np.nan

        for conn_i, conn_row in brain.connections.iterrows():
            matrix[conn_row["pre_idx"], conn_row["post_idx"]] = eval(
                f"conn_row['connection'].{param}"
            )

        if param == "amplitude":
            vmax = np.nanmax(np.abs(matrix))
            vmin = -vmax
            cmap = "RdBu_r"
        else:
            vmax = np.nanmax(np.abs(matrix))
            vmin = 0
            cmap = "magma_r"

        ax = plot_confusion_matrix(
            matrix=matrix,
            ax=ax,
            last_neuron_idx_per_layer=last_neuron_idx_per_layer,
            cmap=cmap,
            vmax=vmax,
            vmin=vmin,
        )
        ax.set_title(param)

    return f


if __name__ == "__main__":
    import os

    # pd.set_option('display.max_rows', None)

    curr_folder = os.path.dirname(os.path.realpath(__file__))

    # brain_config_path_min = os.path.join(
    #     os.path.dirname(curr_folder), "configs", "brain_config_minimal.yml"
    # )
    # brain_min = generate_brain_from_brain_config(
    #     brain_config_path=brain_config_path_min
    # )

    # brain_config_path_4eyes_ff = os.path.join(
    #     os.path.dirname(curr_folder), "configs", "brain_config_4eyes_feedforward.yml"
    # )
    # brain_4eyes_ff = generate_brain_from_brain_config(
    #     brain_config_path=brain_config_path_4eyes_ff
    # )

    # brain_config_path_8eyes_recur = os.path.join(
    #     os.path.dirname(curr_folder), "configs", "brain_config_8eyes_recurrent.yml"
    # )
    # brain_8eyes_recur = generate_brain_from_brain_config(
    #     brain_config_path=brain_config_path_8eyes_recur
    # )
    # print(brain_8eyes_recur.neurons)
    # print(brain_8eyes_recur.connections)

    brain_config_path_8eyes_no_hidden = os.path.join(
        os.path.dirname(curr_folder), "configs", "brain_config_8eyes_no_hidden.yml"
    )
    brain_8eyes_no_hidden = generate_brain_from_brain_config(
        brain_config_path=brain_config_path_8eyes_no_hidden
    )
    print(brain_8eyes_no_hidden.neurons)
    print(brain_8eyes_no_hidden.connections)

    f = plot_brain_connections(brain_8eyes_no_hidden)
    plt.tight_layout()
    plt.show()
