import pandas as pd
from littlefish.core import utilities as util
from littlefish.brain.base import Neuron, Connection, Brain
from littlefish.brain.eyes import Eye
from littlefish.brain.muscles import SimpleMuscle, get_muscle_direction


def load_neuron_from_h5_group(h5_group):
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


# def generate_minimal_brain():
#     """

#     :return: a Brain object with one eye, two neuron in hidden layer and one muscle
#     """

#     eye = WideEye(
#         direction="east",
#         input_filter=None,
#         gain=0.05,
#         input_type="terrain",
#         baseline_rate=0.0,
#         refractory_period=1.2,
#     )

#     hidden0 = Neuron(baseline_rate=0.005, refractory_period=1.2)
#     hidden1 = Neuron(baseline_rate=0.005, refractory_period=1.2)
#     muscle = SimpleMuscle(direction="east", baseline_rate=0.1, refractory_period=500)

#     neurons = pd.DataFrame(
#         [[0, 0, eye], [1, 0, hidden0], [1, 1, hidden1], [2, 0, muscle]],
#         columns=["layer", "neuron_ind", "neuron"],
#     )

#     connection_eye_hidden0 = Connection(
#         latency=3, amplitude=0.01, rise_time=5, decay_time=10
#     )
#     connection_eye_hidden1 = Connection(
#         latency=3, amplitude=0.0001, rise_time=5, decay_time=10
#     )
#     connection_hidden0_muscle = Connection(
#         latency=3, amplitude=0.0001, rise_time=5, decay_time=10
#     )
#     connection_hidden1_muscle = Connection(
#         latency=3, amplitude=0.01, rise_time=5, decay_time=10
#     )

#     conn_0_1 = pd.DataFrame(
#         [[connection_eye_hidden0], [connection_eye_hidden1]], columns=[0], index=[1, 2]
#     )
#     conn_1_2 = pd.DataFrame(
#         [[connection_hidden0_muscle, connection_hidden1_muscle]],
#         columns=[1, 2],
#         index=[3],
#     )

#     connections = {"L000_L001": conn_0_1, "L001_L002": conn_1_2}

#     return Brain(neurons=neurons, connections=connections)


def genearte_brain_from_brain_config(
    brain_config,
):
    neurons = pd.DataFrame(columns=["layer", "neuron_ind", "neuron"])

    neuron_ind = 0
    layer_ind = 0

    # generate eyes
    eye_num = 8
    for eye_ind in range(eye_num):
        curr_eye_dir, curr_eye_input_type = get_eye_type(eye_ind, dir_num=4)
        curr_eye = WideEye(
            direction=curr_eye_dir,
            gain=brain_config["eye_gain"],
            input_type=curr_eye_input_type,
            baseline_rate=brain_config["eye_baseline_rate"],
            refractory_period=brain_config["eye_refractory_period"],
        )
        neurons.loc[neuron_ind, "layer"] = 0
        neurons.loc[neuron_ind, "neuron_ind"] = eye_ind
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
    mus_num = 4
    for mus_ind in range(mus_num):
        curr_mus_dir = get_muscle_direction(mus_ind)
        curr_muscle = SimpleMuscle(
            direction=curr_mus_dir,
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
