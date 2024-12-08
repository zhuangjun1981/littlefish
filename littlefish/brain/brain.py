import random
import h5py
import numpy as np
import pandas as pd
from littlefish.brain.neuron import Neuron, Eye, Muscle
from littlefish.brain.connection import Connection
from littlefish.core import utilities as util


def generate_minimal_brain():
    """
    :return: a Brain object with one eye, two neuron in hidden layer and one muscle
    """

    neurons = pd.DataFrame()
    neurons["layer"] = [0, 1, 1, 2]
    neurons["neuron"] = [Eye(), Neuron(), Neuron(), Muscle()]

    # connections from eye to hidden layer
    connections = pd.DataFrame()
    connections["pre_idx"] = [0, 0, 1, 2]
    connections["post_idx"] = [1, 2, 3, 3]
    connections["connection"] = [Connection(), Connection(), Connection(), Connection()]

    return Brain(neurons=neurons, connections=connections)


class Brain:
    """
    Brain contains a graph of neurons and their connections.
    Attributeis:
      -neurons is a dataframe with two columns:
        - layer (int)
        - neuron (Neuron or its subclass).

        Each row is one neuron. The index of this dataframe is important
        because it serves as the idetifier of each neuron.

      - connections is a dataframe with three columns:
        - pre_idx (int, index of presynaptic neuron)
        - post_idx (int, index of postsynaptic neuron)

        Each row is one connection. The index of this dataframe serves
        as the identifier of each connection.

      - simulation_cache: dictionary, with two keys: "action_histories" and "psp_waveforms".
        if not in simulation, the value of both keys should be None.
        - action_histories: list[list[int]], outer list has the length of number of neurons,
          with the same order of self.neurons. inner list is the timestamp of action potentials
          of each neuron.
        - psp_waveforms: 2d array, shape: (num_neurons, t_points in simulation). post synaptic
          potential waveform of each neuron.

    The reason the "simulation_cache" is saved in this class but not in simulation class is that
    this allow the simulation to run other types of brains which may not have the same simulation
    cache (memory) structure.
    """

    def __init__(
        self, neurons: pd.DataFrame = None, connections: pd.DataFrame = None
    ) -> None:
        if neurons is None and connections is None:
            min_brain = generate_minimal_brain()
            self.neurons = min_brain.neurons()
            self.connections = min_brain.connections()
        else:
            self.neurons = neurons.sort_index()
            self.connections = connections.sort_index()

        self.clear_simulation_cache()

        self.check_integrity()

    def clear_simulation_cache(self):
        self.simulation_cache = {"action_histories": None, "psp_waveforms": None}

    def check_integrity(self) -> None:
        # check columns of neurons
        assert len(self.neurons.shape) == 2, "self.neurons dataframe should be 2d."
        assert (
            self.neurons.shape[0] >= 1
        ), "self.neurons should have at least one neuron."
        assert (
            self.neurons.shape[1] == 2
        ), "self.neurons dataframe should have two columns."
        assert list(self.neurons.columns) == [
            "layer",
            "neuron",
        ], "columns of self.neuron dataframe should be ['layer', 'neuron']"

        # check index of self.neuron is starting from 0 and continuously increasing with step 1
        assert np.array_equal(
            np.arange(self.neurons.shape[0], dtype=int),
            np.array(self.neurons.index),
        ), "index of self.neurons dataframe should be consecutively increasing integers starting from 0"

        # check layer of self.neuron is starting from 0 and increasing with step 1
        layers = self.neurons["layer"]
        for i in range(1, len(layers)):
            assert (
                layers[i] == layers[i - 1] or layers[i] == layers[i - 1] + 1
            ), "'layer' in self.neurons dataframe should be monotonically increasing with step of 1."

        # check the first layer are all eyes
        # check the last layer are all muscles
        # check all hiddent layers are all neurons
        for neuron_i, neuron_row in self.neurons.iterrows():
            if neuron_row["layer"] == 0:
                assert (
                    neuron_row["neuron"].type == "littlefish.brain.neuron.Eye"
                ), "first layer should be eyes."
            elif neuron_row["layer"] == self.num_layers - 1:
                assert (
                    neuron_row["neuron"].type == "littlefish.brain.neuron.Muscle"
                ), "last layer should be muscles."
            else:
                assert (
                    neuron_row["neuron"].type == "littlefish.brain.neuron.Neuron"
                ), "hidden layer should be neurons."

        # check columns of connections
        assert self.connections.ndim == 2, "self.connections dataframe should be 2d"
        assert (
            self.connections.shape[1] == 3
        ), "self.connections dataframe should have 3 columns"
        assert list(self.connections.columns) == ["pre_idx", "post_idx", "connection"]

        # check pre- and post-synaptic neurons exist in self.neurons
        for connection_i, connection_row in self.connections.iterrows():
            assert (
                connection_row["pre_idx"] < self.num_neurons
            ), "all presynaptic neurons should exist in self.neurons dataframe."
            assert (
                connection_row["post_idx"] < self.num_neurons
            ), "all postsynaptic neurons should exist in self.neurons dataframe."
            assert (
                connection_row["connection"].type
                == "littlefish.brain.connection.Connection"
            )

        # check a unique pair of pre- and post-synaptic neurons has only one connection
        assert (
            len(set(zip(self.connections["pre_idx"], self.connections["post_idx"])))
            == self.num_connections
        ), "connections should be unique."

        # check simulation cache
        action_histories = self.simulation_cache["action_histories"]
        psp_waveforms = self.simulation_cache["psp_waveforms"]

        if action_histories is None or psp_waveforms is None:
            pass
        else:
            assert (
                len(action_histories) == self.num_neurons
            ), "length of self.simulation_cache['action_histories'] should be the same as number of neurons."
            assert (
                psp_waveforms.ndim == 2
            ), "self.simulation_cache['psp_waveforms'] should be 2d array."
            assert (
                psp_waveforms.shape[0] == self.num_neurons
            ), "number of rows of self.simulation_cache['psp_waveforms'] should be the same as number of neurons."

    @property
    def num_neurons(self) -> int:
        return self.neurons.shape[0]

    @property
    def num_layers(self) -> int:
        return self.neurons.loc[self.neurons.shape[0] - 1, "layer"] + 1

    @property
    def num_connections(self) -> int:
        return self.connections.shape[0]

    def get_postsynaptic_indices(self, neuron_idx: int) -> list[int]:
        return list(self.connections.query("pre_idx == @neuron_idx")["post_idx"])

    def get_presynaptic_indices(self, neuron_idx: int) -> list[int]:
        return list(self.connections.query("post_idx == @neuron_idx")["pre_idx"])

    def get_postsynaptic_indices_and_connections(self, neuron_idx: int) -> pd.DataFrame:
        """
        given a presynaptic neuron index, return a dataframe containing all of its postsynaptic neurons
        and corresponding connections.
        """
        return self.connections.query("pre_idx == @neuron_idx")[
            ["post_idx", "connection"]
        ]

    def initiate_simulation(self, max_simulation_length):
        """
        initiate simulation: instantiation self.simulation_cache and pre-allocate memory
        for psp_waveforms.
        """
        self.simulation_cache["action_histories"] = [
            [] for _ in range(self.num_neurons)
        ]
        self.simulation_cache["psp_waveforms"] = np.zeros(
            (self.num_neurons, max_simulation_length), dtype=np.float32
        )

    def neuron_fire(
        self,
        presynaptic_idx: int,
        t_point: int,
    ) -> None:
        """
        updata all corresponding psp waveforms when a presynaptic neuron (only in eye layer and hidden layer) fires

        :param presynaptic_neuron_idx: int, the index of presynaptic neuron in self.neurons
        :param t_point: int, time point in time unit axis of the action potential
        :param psp_waveforms: 2d-array, shape (num_neurons, number of total simulation time points), dtype: floats,
            psp waveforms of all neurons in the brain, each row represents one neuron in the same order as self.neurons
            dataframe, each column represents a time point
        :return: None
        """
        post_df = self.get_postsynaptic_indices_and_connections(
            neuron_idx=presynaptic_idx
        )
        for post_i, post_row in post_df.iterrows():
            post_neuron_idx = post_row["post_idx"]
            curr_connection = post_row["connection"]
            curr_connection.act(
                t_point=t_point,
                postsynaptic_index=post_neuron_idx,
                psp_waveforms=self.simulation_cache["psp_waveforms"],
            )

    def act(
        self,
        t_point: int,
        body_position: tuple[int],
        terrain_map: np.ndarray,
        food_map: np.ndarray = None,
        fish_map: np.ndarray = None,
    ):
        """

        :param t_point: int, current time stamp of time unit axis
        :param body_position: tuple of two ints, (row, col), current position of body center of the fish
        :param terrain_map: 2d array, with only 0s (water) and 1s (land). represents the land scape of the world
        :param food_map: 2d array, with only 0s (no food) and 1s (food). represents the distribution of food
        :param fish_map: not fully implemented right now.
        :return:
          movement_attempt: 1-d array, np.uint8, (row, col), representing the movement attempt, be careful, this
            may not represent the actual movement, it will be evaluated by the fish object
            (fish class) containing this brain to see if the movement is possible. if the fish
            is hitting the edge the world map, then the it will not move out of the map
            None: no movement has been attempted.
          total_action_potential_number: int, number of total action potential of the whole brain at this time point.
            [0, num_neurons], each neuron can only have a single action potential at a time point.
        """

        movement_attempt = np.array([0, 0], dtype=np.uint8)
        total_action_potential_number = 0

        for i, neuron in self.neurons.iterrows():
            if neuron["neuron"].type == "littlefish.brain.neuron.Eye":  # eye layer
                curr_eye = neuron["neuron"]

                if curr_eye.input_type == "terrain":
                    input_map = terrain_map
                    border_value = 1
                elif curr_eye.input_type == "food":
                    input_map = food_map
                    border_value = 0
                elif curr_eye.input_type == "fish":
                    input_map = fish_map
                    border_value = 0
                else:
                    raise ValueError(
                        "Brain: the input_type of eye should be one of the following:"
                        '"terrain", "food" or "fish".'
                    )

                is_fire = curr_eye.act(
                    input_map=input_map,
                    body_position=body_position,
                    border_value=border_value,
                    t_point=t_point,
                    action_history=self.simulation_cache["action_histories"][i],
                )

                if is_fire:  # the current eye fires
                    # print('eye spike')
                    self.neuron_fire(
                        presynaptic_idx=i,
                        t_point=t_point,
                    )
                    total_action_potential_number += 1

            elif (
                neuron["neuron"].type == "littlefish.brain.neuron.Neuron"
            ):  # hidden neuron
                curr_neuron = neuron["neuron"]
                is_fire = curr_neuron.act(
                    t_point=t_point,
                    action_history=self.simulation_cache["action_histories"][i],
                    probability_input=self.simulation_cache["psp_waveforms"][
                        i, t_point
                    ],
                )
                if is_fire:
                    # print('neuron spike')
                    self.neuron_fire(
                        presynaptic_idx=i,
                        t_point=t_point,
                    )
                    total_action_potential_number += 1

            elif neuron["neuron"].type == "littlefish.brain.neuron.Muscle":  # muscle
                curr_muscle = neuron["neuron"]
                curr_movement_attempt = curr_muscle.act(
                    t_point=t_point,
                    action_history=self.simulation_cache["action_histories"][i],
                    probability_input=self.simulation_cache["psp_waveforms"][
                        i, t_point
                    ],
                )
                if curr_movement_attempt is not False:
                    # print('muscle spike')
                    movement_attempt = movement_attempt + curr_movement_attempt
                    total_action_potential_number += 1
            else:
                raise ValueError(
                    "Brain: neuron at index" + str(i) + " has invalid type."
                )

        return movement_attempt, total_action_potential_number

    # def get_connection_matrix(self):
    #     pass

    # def to_h5_group(self, h5_group: h5py.Group):
    #     pass


class BrainOld(object):
    """
    brain class, the neural network from eye to muscle

    a 'brain' has a couple of sets of 8 eyes (brain.Eye object, each at each border pixel of the body). each set of
    eyes are receiving inputs from different objects. i.e. one set of eyes will look at land/water, another set of eyes
    will look for food, another set of eyes will look for other fish.

    a 'brain' has 4 invisible muscles (brain.Muscle object, each controlling the movement in each direction).

    between eyes and muscles are a neural network consists of neurons (brain.Neuron object) and connections
    (brain.Connections object). Number of layers and number of neurons can be specified.
    """

    def __init__(self, neurons=None, connections=None):
        """

        :param neurons: pandas dataframe
        :param connections: dict
        """

        # print('\nBrain: Creating littlefish.core.fish.Brain object ...')

        if neurons is None and connections is None:
            min_brain = generate_minimal_brain()
            self.neurons = min_brain.get_neurons()
            self.connections = min_brain.get_connections()
        else:
            self.neurons = neurons
            self.connections = connections

        self.check_integrity(verbose=False)

        # print('Brain: littlefish.core.fish.Brain created successfully.')

    @property
    def layer_num(self):
        return int(round(max(self.neurons["layer"]))) + 1

    # def get_layer_type(self, layer: int):
    #     """
    #     :return: layer type (str) given the layer number
    #     """

    #     if not isinstance(layer, int):
    #         raise ValueError("Input layer number should be integer.")

    #     if layer == 0:
    #         return "eye"
    #     elif layer == self.layer_num - 1:
    #         return "muscle"
    #     elif 0 < layer < self.layer_num - 1:
    #         return "hidden" + util.int2str(layer, 3)
    #     else:
    #         raise ValueError("layer number out of range.")

    # def get_neuron_type(self, ind):
    #     """
    #     return neuron type as a pair of strings given the index in self.neurons

    #     :param ind: int
    #     :return: for eyes : ('eye', type + short of direction)
    #              for hidden neurons: ('hidden', str(layer))
    #              for muscles ('muscle', short of direction)
    #     """

    #     # self.check_integrity_neurons()

    #     curr_row = self.neurons.loc[ind]
    #     curr_layer = curr_row["layer"]
    #     curr_neuron = curr_row["neuron"]
    #     if curr_layer == 0:  # eye layer
    #         return (
    #             util.short("eye")
    #             + "_"
    #             + util.short(curr_neuron.input_type)
    #             + "_"
    #             + util.short(curr_neuron.eye_direction)
    #         )
    #     elif curr_layer == self.layer_num - 1:  # muscle layer
    #         curr_dir = curr_neuron.direction
    #         return util.short("muscle") + "_" + util.short(curr_dir)
    #     elif 0 < curr_layer < self.layer_num - 1:
    #         curr_layer_num = util.int2str(curr_layer, 3)
    #         curr_neuron_num = util.int2str(curr_row["neuron_ind"], 3)
    #         return "_".join([util.short("hidden"), curr_layer_num, curr_neuron_num])
    #     else:
    #         raise ValueError("layer number out of range.")

    def get_postsynaptic_neuron_inds(self, neuron_ind):
        neuron_layer = int(round(self.neurons.loc[neuron_ind, "layer"]))
        if neuron_layer < 0:
            raise ValueError("Brain: invalid layer. less than 0.")
        elif neuron_layer == self.layer_num - 1:
            print("Brain: cannot fine postsynaptic neurons of neurons in muscle layer.")
        else:
            postsynaptic_neuron_ind = self.neurons.query(
                f"layer == {neuron_layer + 1}"
            ).index.tolist()
            postsynaptic_neuron_ind.sort()
            return postsynaptic_neuron_ind

    def get_presynaptic_neuron_inds(self, neuron_ind):
        neuron_layer = int(round(self.neurons.loc[neuron_ind, "layer"]))
        if neuron_layer < 0:
            raise ValueError("Brain: invalid layer. less than 0.")
        elif neuron_layer == 0:
            print("Brain: cannot fine presynaptic neuron of neurons in eye layer.")
        else:
            presynaptic_neuron_ind = self.neurons.query(
                f"layer == {neuron_layer - 1}"
            ).index.tolist()
            presynaptic_neuron_ind.sort()
            return presynaptic_neuron_ind

    def get_single_connection(self, pre_neuron_ind, post_neuron_ind):
        pre_layer = int(round(self.neurons.loc[pre_neuron_ind, "layer"]))
        post_layer = int(round(self.neurons.loc[post_neuron_ind, "layer"]))

        if post_layer - pre_layer != 1:
            raise LookupError(
                "Brain: presynaptic layer"
                + str(pre_layer)
                + " and postsynaptic layer"
                + str(post_layer)
                + " do not form connections."
            )

        conn_df = self._connections[
            "L" + util.int2str(pre_layer, 3) + "_L" + util.int2str(post_layer, 3)
        ]
        return conn_df.loc[post_neuron_ind, pre_neuron_ind]

    def get_neuron_inds_in_layer(self, layer):
        """
        return a list of sorted neuron_indices of all neurons in a given layer
        """

        inds = self.neurons[self.neurons["layer"] == layer].index.tolist()
        inds.sort()
        return inds

    def check_integrity(self, verbose=True):
        """
        check integrity of object data structure
        """

        if verbose:
            print("Brain: checking integrity of attrbitue data structure ...")

        self.check_integrity_neurons(verbose=verbose)

        self.check_integrity_connection(verbose=verbose)

        if verbose:
            print("Brain: integrity checking finished. All pass.")

    def check_integrity_neurons(self, verbose=False):
        if not util.check_df_index(self.neurons):
            raise ValueError(
                "Brain: the indices of self._neurons are not starting at 0 and increasing with step 1."
            )
        else:
            if verbose:
                print(
                    "Brain: the indices of self._neurons are starting at 0 and increasing with step 1. PASS."
                )
            else:
                pass

        layer = 0
        ind = -1
        for i, neuron in self._neurons.iterrows():
            curr_layer = int(round(neuron["layer"]))
            curr_neuron_ind = neuron["neuron_ind"]
            if curr_layer < layer:
                raise ValueError(
                    'Brain: the "layer" in self.neurons is not in ascending order.'
                )
            elif curr_layer == layer:
                if curr_neuron_ind != ind + 1:
                    raise ValueError(
                        'Brain: the "neuron_ind" in self._neurons is not in ascending by step 1 for'
                        ' each "layer"'
                    )
                else:
                    ind += 1
            else:
                layer = curr_layer
                if curr_neuron_ind != 0:
                    raise ValueError(
                        'Brain: the "neuron_ind" in self._neurons does not start with 0 for each '
                        '"layer".'
                    )
                ind = 0

            if curr_layer == 0:  # eye layer
                if not (
                    str(neuron["neuron"]) == "littlefish.brain.Eye object"
                    or str(neuron["neuron"]) == "littlefish.brain.Eye2 object"
                ):
                    raise ValueError("Brain: non-eye object in eye layer.")
            elif curr_layer == self.layer_num - 1:  # muscle layer
                if not str(neuron["neuron"]) == "littlefish.brain.Muscle object":
                    raise ValueError("Brain: non-muscle object in muscle layer.")
            else:  # hidden layer
                if not str(neuron["neuron"]) == "littlefish.brain.Neuron object":
                    raise ValueError("Brain: non-neuron object in hidden layer.")

        if verbose:
            print(
                'Brain: the "layer" of self._neurons is in a non-descending order. PASS'
            )
            print(
                'Brain: the "neuron_ind" of self._neurons for each layer is ascending from 0 by step 1. PASS'
            )
            print(
                "Brain: eyes in eye layer, muscles in muscle layer, neurons in hidden layer. PASS"
            )

    def check_integrity_connection(self, verbose=False):
        matching_keys = []
        for i in range(self.layer_num - 1):
            matching_keys.append(
                "L" + util.int2str(i, 3) + "_L" + util.int2str(i + 1, 3)
            )
        matching_keys.sort()

        conn_keys = list(self._connections.keys())
        conn_keys.sort()

        if not conn_keys == matching_keys:
            raise ValueError("Brain: invalid keys in self._connections.")
        else:
            if verbose:
                print("Brain: valid keys in self._connections. PASS")
            else:
                pass

        for key in conn_keys:
            pre_layer = int(key[1:4])
            post_layer = int(key[6:9])
            curr_conn_df = self._connections[key]
            pre_neuron_ind = self.get_neuron_inds_in_layer(pre_layer)
            post_neuron_ind = self.get_neuron_inds_in_layer(post_layer)
            if not np.array_equal(pre_neuron_ind, curr_conn_df.columns.tolist()):
                raise ValueError(
                    "Brain: connections dataframe "
                    + key
                    + " does not have valid column name."
                )
            if not np.array_equal(post_neuron_ind, curr_conn_df.index.tolist()):
                raise ValueError(
                    "Brain: connections dataframe "
                    + key
                    + " does not have valid index name."
                )

        if verbose:
            print(
                "Brain: dataframes in self._connections have valid column and index names. PASS"
            )

    def act(
        self,
        t_point,
        action_histories,
        psp_waveforms,
        body_position,
        terrain_map,
        food_map=None,
        fish_map=None,
    ):
        """

        :param t_point: int, current time stamp of time unit axis
        :param action_histories: data frame of lists, each list is the action history of a particular neuron, in the
                                 same order as self._neurons data frame, columns = ['action_history']
        :param psp_waveforms: 2d-array of floats, psp waveforms of all neurons in the brain, each row represents one
                              neuron in the same order as self._neurons data frame, each column represents a time point
        :param body_position: tuple of two ints, (row, col), current position of body center of the fish
        :param terrain_map: 2d array, with only 0s (water) and 1s (land). represents the land scape of the world
        :param food_map: 2d array, with only 0s (no food) and 1s (food). represents the distribution of food
        :param fish_map: not fully implemented right now.
        :return: movement_attempt: 2-d array, np.uint8, (row, col), representing the movement attempt, be careful, this
                                   may not represent the actual movement, it will be evaluated by the fish object
                                   (fish class) containing this brain to see if the movement is possible. if the fish
                                   is hitting the edge the world map, then the it will not move out of the map
                                   None: no movement has been attempted,
        """

        movement_attempt = np.array([0, 0], dtype=np.uint8)

        for i, neuron in self._neurons.iterrows():
            if neuron["layer"] == 0:  # eye layer
                curr_eye = neuron["neuron"]

                if curr_eye.get_input_type() == "terrain":
                    is_fire = curr_eye.act(
                        t_point=t_point,
                        action_history=action_histories.iloc[i, 0],
                        body_position=body_position,
                        input_map=terrain_map,
                    )
                elif curr_eye.get_input_type() == "food":
                    if food_map is not None:
                        is_fire = curr_eye.act(
                            t_point=t_point,
                            action_history=action_histories.loc[i, "action_history"],
                            body_position=body_position,
                            input_map=food_map,
                        )
                    else:
                        is_fire = False
                elif curr_eye.get_input_type() == "fish":
                    if fish_map is not None:
                        is_fire = curr_eye.act(
                            t_point=t_point,
                            action_history=action_histories.loc[i, "action_history"],
                            body_position=body_position,
                            input_map=fish_map,
                        )
                    else:
                        is_fire = False
                else:
                    raise ValueError(
                        "Brain: the input_type of eye should be one of the following:"
                        '"terrain", "food" or "fish".'
                    )

                if is_fire:  # the current eye fires
                    # print('eye spike')
                    self.neuron_fire(
                        presynaptic_neuron_ind=i,
                        t_point=t_point,
                        psp_waveforms=psp_waveforms,
                    )

            elif neuron["layer"] < self.layer_num - 1:  # hidden layer
                curr_neuron = neuron["neuron"]
                is_fire = curr_neuron.act(
                    t_point=t_point,
                    action_history=action_histories.loc[i, "action_history"],
                    probability_input=psp_waveforms[i, t_point],
                )
                if is_fire:
                    # print('neuron spike')
                    self.neuron_fire(
                        presynaptic_neuron_ind=i,
                        t_point=t_point,
                        psp_waveforms=psp_waveforms,
                    )

            elif neuron["layer"] == self.layer_num - 1:  # muscle layer
                curr_muscle = neuron["neuron"]
                curr_movement_attempt = curr_muscle.act(
                    t_point=t_point,
                    action_history=action_histories.loc[i, "action_history"],
                    probability_input=psp_waveforms[i, t_point],
                )
                if curr_movement_attempt is not False:
                    # print('muscle spike')
                    movement_attempt = movement_attempt + curr_movement_attempt
            else:
                raise ValueError(
                    "Brain: neuron at index" + str(i) + " has invalid layer location."
                )

        return movement_attempt

    def neuron_fire(self, presynaptic_neuron_ind, t_point, psp_waveforms):
        """
        updata all corresponding psp waveforms when a presynaptic neuron (only in eye layer and hidden layer) fires

        :param presynaptic_neuron_ind: int, the index of presynaptic neuron in self._neurons
        :param t_point: int, time point in time unit axis of the action
        :param psp_waveforms: 2d-array of floats, psp waveforms of all neurons in the brain, each row represents one
                              neuron in the same order as self._neurons data frame, each column represents a time point
        :return: None
        """

        neuron_layer = int(round(self._neurons.loc[presynaptic_neuron_ind, "layer"]))

        # ========================= slower but better method =====================================================
        # if 0 <= neuron_layer < self.layer_num - 1:  # eye layer or hidden layer
        #     curr_conn_df = self._connections['L' + util.int2str(neuron_layer, 3) +
        #                                      '_L' + util.int2str(neuron_layer + 1, 3)]
        #     postsynaptic_neuron_inds = self.get_postsynaptic_neuron_inds(neuron_ind=presynaptic_neuron_ind)
        #
        #     for postsynaptic_neuron_ind in postsynaptic_neuron_inds:
        #         curr_connection = curr_conn_df.loc[postsynaptic_neuron_ind, presynaptic_neuron_ind]
        #         curr_connection.act(t_point=t_point, postsynaptic_index=postsynaptic_neuron_ind,
        #                             psp_waveforms=psp_waveforms)
        # elif neuron_layer == self.layer_num - 1:  # muscle layer
        #     print('Brain: a firing of a muscle has no effect on brain itself. Please use Muscle.act() method to '
        #           'generate movement attempt.')
        # else:
        #     raise ValueError('Brain: neuron at index' + str(presynaptic_neuron_ind) + ' has invalid layer location.')
        # ========================= slower but better method =====================================================

        # ========================= faster but unsafe method =====================================================
        curr_conn_df = self._connections[
            "L"
            + util.int2str(neuron_layer, 3)
            + "_L"
            + util.int2str(neuron_layer + 1, 3)
        ]
        postsynaptic_neuron_inds = self.get_postsynaptic_neuron_inds(
            neuron_ind=presynaptic_neuron_ind
        )

        for postsynaptic_neuron_ind in postsynaptic_neuron_inds:
            curr_connection = curr_conn_df.loc[
                postsynaptic_neuron_ind, presynaptic_neuron_ind
            ]
            curr_connection.act(
                t_point=t_point,
                postsynaptic_index=postsynaptic_neuron_ind,
                psp_waveforms=psp_waveforms,
            )
        # ========================= faster but unsafe method =====================================================

    def get_all_presynaptic_neuron_indices(self):
        """
        get indices of all presynaptic neurons
        """

        layer_num = int(max(self._neurons["layer"])) + 1
        ind = self._neurons[self._neurons["layer"] < layer_num - 1].index
        return ind.sort_values()

    def get_all_postsynaptic_neuron_indices(self):
        """
        get indices of all postsynaptic neurons
        """

        ind = self._neurons[self._neurons["layer"] > 0].index
        return ind.sort_values()

    def get_connection_matrices(self, pre_layer, post_layer):
        """
        return several numpy arrays each represent one parameter of all connections between a presynaptic layer and
        a postsynaptic layer, each row is a postsynaptic neuron, each column is a presynaptic neuron

        :param pre_layer: int, layer number of presynaptic layer
        :param post_layer: int, layer number of postsynaptic layer
        :return rows: list of ints, postsynaptic neuron inds for each row
        :return cols: list of ints, presynaptic neuron inds for each column
        :return latencies: amplitudes, rise_times, decay_times: matrices for each connection parameter as described
                           above
        """

        rows = self.get_neuron_inds_in_layer(post_layer)
        cols = self.get_neuron_inds_in_layer(pre_layer)

        latencies = np.empty((len(rows), len(cols)), dtype=np.uint)
        amplitudes = np.empty((len(rows), len(cols)), dtype=np.float32)
        rise_times = np.empty((len(rows), len(cols)), dtype=np.uint)
        decay_times = np.empty((len(rows), len(cols)), dtype=np.uint)

        conn_df = self._connections[
            "L" + util.int2str(pre_layer, 3) + "_L" + util.int2str(post_layer, 3)
        ]

        for i in range(conn_df.shape[0]):
            for j in range(conn_df.shape[1]):
                latencies[i, j] = conn_df.iloc[i, j].get_latency()
                amplitudes[i, j] = conn_df.iloc[i, j].get_amplitude()
                rise_times[i, j] = conn_df.iloc[i, j].get_rise_time()
                decay_times[i, j] = conn_df.iloc[i, j].get_decay_time()
        return rows, cols, latencies, amplitudes, rise_times, decay_times

    def to_h5_group(self, h5_group):
        neuron_group = h5_group.create_group("neurons")
        for i, neuron_df in self._neurons.iterrows():
            neuron_name = "neuron_" + util.int2str(i, 4)
            curr_neuron_group = neuron_group.create_group(neuron_name)
            neuron_df["neuron"].to_h5_group(curr_neuron_group)
            curr_neuron_group.attrs["ind"] = i
            curr_neuron_group.attrs["layer"] = neuron_df["layer"]
            curr_neuron_group.attrs["neuron_ind"] = neuron_df["neuron_ind"]

        connection_group = h5_group.create_group("connections")
        for pre_layer in range(self.layer_num - 1):
            post_layer = pre_layer + 1
            curr_connection_matrices = self.get_connection_matrices(
                pre_layer=pre_layer, post_layer=post_layer
            )

            curr_layer_group = connection_group.create_group(
                "L" + util.int2str(pre_layer, 3) + "_L" + util.int2str(post_layer, 3)
            )
            curr_layer_group.attrs["rows"] = curr_connection_matrices[0]
            curr_layer_group.attrs["cols"] = curr_connection_matrices[1]
            curr_layer_group.attrs["doc"] = (
                "rows: indices of postsynatpic neurons in the neuron group; "
                "cols: indices of presynaptic neurons in the neuron group."
            )
            curr_layer_group.create_dataset(
                name="latencies_tu", data=curr_connection_matrices[2]
            )
            curr_layer_group.create_dataset(
                name="amplitudes", data=curr_connection_matrices[3]
            )
            curr_layer_group.create_dataset(
                name="rise_times_tu", data=curr_connection_matrices[4]
            )
            curr_layer_group.create_dataset(
                name="decay_times_tu", data=curr_connection_matrices[5]
            )

    @staticmethod
    def from_h5_group(h5_group):
        neurons = pd.DataFrame(columns=["layer", "neuron_ind", "neuron"])

        neurons_group = h5_group["neurons"]
        neuron_names = list(neurons_group.keys())
        neuron_names.sort()
        for neuron_name in neuron_names:
            curr_neuron_group = neurons_group[neuron_name]
            curr_layer = curr_neuron_group.attrs["layer"]
            curr_neuron_ind = curr_neuron_group.attrs["neuron_ind"]
            curr_ind = curr_neuron_group.attrs["ind"]

            curr_neuron_type = util.decode(curr_neuron_group.attrs["neuron_type"])

            if curr_neuron_type == "neuron":
                curr_neuron = Neuron.from_h5_group(curr_neuron_group)
            elif curr_neuron_type == "eye":
                curr_neuron = Eye.from_h5_group(curr_neuron_group)
            elif curr_neuron_type == "muscle":
                curr_neuron = Muscle.from_h5_group(curr_neuron_group)
            else:
                raise LookupError(
                    'Brain: fail to load neuron. "neuron_type" attribute should be one of the '
                    'following: "eye", "neuron" or "muscle".'
                )

            neurons.loc[curr_ind] = [curr_layer, curr_neuron_ind, curr_neuron]

        connections = {}

        connections_group = h5_group["connections"]
        connections_names = list(connections_group.keys())
        connections_names.sort()
        for connections_name in connections_names:
            curr_conn_group = connections_group[connections_name]
            curr_inds = curr_conn_group.attrs["rows"]
            curr_cols = curr_conn_group.attrs["cols"]
            curr_amplitudes = curr_conn_group["amplitudes"][()]
            curr_decay_times = curr_conn_group["decay_times_tu"][()]
            curr_rise_times = curr_conn_group["rise_times_tu"][()]
            curr_latencies = curr_conn_group["latencies_tu"][()]

            curr_conn_df = pd.DataFrame(columns=curr_cols, index=curr_inds)

            for i in range(len(curr_inds)):
                for j in range(len(curr_cols)):
                    curr_conn_df.iloc[i, j] = Connection(
                        latency=curr_latencies[i, j],
                        amplitude=curr_amplitudes[i, j],
                        rise_time=curr_rise_times[i, j],
                        decay_time=curr_decay_times[i, j],
                    )
            connections.update({connections_name: curr_conn_df})

        loaded_brain = Brain(neurons=neurons, connections=connections)

        return loaded_brain

    def generate_empty_action_histories(self):
        """

        :return: a data frame with empty lists, each list is the action history of a particular neuron, in the same
                 order as self._neurons data frame, columns = ['action_history']
        """

        empty_action_histories = pd.Series([[] for i in range(len(self._neurons))])
        empty_action_histories = pd.DataFrame(
            empty_action_histories, columns=["action_history"]
        )
        return empty_action_histories

    def generate_empty_psp_waveforms(self, simulation_length):
        """

        :param simulation_length: int, number of time points of the simulation
        :return: 2d-array of zeros, float32, psp waveforms of all neurons in the brain, each row represents one
                 neuron in the same order as self._neurons data frame, each column represents a time point
        """

        return np.zeros((len(self._neurons), simulation_length), dtype=np.float32)


if __name__ == "__main__":
    brain = generate_minimal_brain()
    print(brain.neurons)
    print(brain.connections)
    print(brain.get_postsynaptic_indices(0))  # return [1, 2]
    print(brain.get_presynaptic_indices(3))  # return [1, 2]
    print(brain.get_postsynaptic_indices_and_connections(0))
