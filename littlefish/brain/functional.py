import h5py
from typing import Union
import pandas as pd
from littlefish.core import utilities as util
from littlefish.brain.neuron import (
    Neuron,
    Eye,
    Muscle,
    FOUR_EYES,
    EIGHT_EYES,
    FOUR_MUSCLES,
)
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


def genearte_brain_from_brain_config(
    brain_config,
):
    neurons = pd.DataFrame(columns=["layer", "neuron_ind", "neuron"])

    neuron_ind = 0
    layer_ind = 0

    # generate eyes
    for input_type in brain_config["input_types"]:
        eye_set = brain_config["eye_set"]
        for eye_direction, eye_dict in eval(f"{eye_set}.items()"):
            curr_eye = Eye(
                eye_direction=eye_direction,
                rf_positions=eye_dict["rf_positions"],
                rf_weights=eye_dict["rf_weights"],
                gain=brain_config["eye_gain"],
                input_type=input_type,
                baseline_rate=brain_config["eye_baseline_rate"],
                refractory_period=brain_config["eye_refractory_period"],
            )
            neurons.loc[neuron_ind, "layer"] = 0
            neurons.loc[neuron_ind, "neuron_ind"] = neuron_ind
            neurons.loc[neuron_ind, "neuron"] = curr_eye
            neuron_ind += 1

    # generate hidden layers
    layer_ind += 1
    hid_nums = brain_config[
        "hidden_neuron_nums"
    ]  # each number is number of neurons in each hidden layer, default is one hidden layer with 8 neurons
    for hid_num in hid_nums:
        for hid_ind in range(hid_num):
            curr_neuron = Neuron(
                baseline_rate=brain_config["neuron_baseline_rate"],
                refractory_period=brain_config["neuron_refractory_period"],
            )
            neurons.loc[neuron_ind, "layer"] = layer_ind
            neurons.loc[neuron_ind, "neuron_ind"] = hid_ind
            neurons.loc[neuron_ind, "neuron"] = curr_neuron
            neuron_ind += 1

        layer_ind += 1

    # generate muscles
    muscle_set = eval(brain_config["muscle_set"])
    for mus_ind, mus_tuple in muscle_set:
        curr_muscle = Muscle(
            direction=mus_tuple[0],
            step_motion=mus_tuple[1],
            baseline_rate=brain_config["muscle_baseline_rate"],
            refractory_period=brain_config["muscle_refractory_period"],
        )
        neurons.loc[neuron_ind, "layer"] = layer_ind
        neurons.loc[neuron_ind, "neuron_ind"] = mus_ind
        neurons.loc[neuron_ind, "neuron"] = curr_muscle
        neuron_ind += 1
    # ================================== generate neurons =========================================

    # ================================== generate connections =========================================
    connections = {}

    default_connection = Connection(
        latency=brain_config["connection_latency"],
        amplitude=brain_config["connection_latency"],
        rise_time=brain_config["connection_rise_time"],
        decay_time=brain_config["connection_decay_time"],
    )
    layer_num = int(round(max(neurons["layer"]))) + 1

    for pre_layer in range(layer_num - 1):
        post_layer = pre_layer + 1

        post_neuron_inds = neurons[neurons["layer"] == post_layer].index.tolist()
        post_neuron_inds.sort()

        pre_neuron_inds = neurons[neurons["layer"] == pre_layer].index.tolist()
        pre_neuron_inds.sort()

        curr_name = (
            "L" + util.int2str(pre_layer, 3) + "_L" + util.int2str(post_layer, 3)
        )
        # curr_df = pd.DataFrame([[default_connection] * len(pre_neuron_inds)] * len(post_neuron_inds),
        #                        columns=pre_neuron_inds, index=post_neuron_inds)
        curr_conn_df = pd.DataFrame(columns=pre_neuron_inds, index=post_neuron_inds)
        curr_conn_df[:] = default_connection
        connections.update({curr_name: curr_conn_df})
    # ================================== generate connections =========================================

    # generate brain
    return Brain(neurons=neurons, connections=connections)
