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
            self.neurons = min_brain.neurons
            self.connections = min_brain.connections
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

    def initiate_simulation(self, max_simulation_length: int) -> None:
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
        updata all corresponding psp waveforms (in self.simulation_cache["psp_waveforms"]) when
        a presynaptic neuron (only in eye layer and hidden layer) fires

        :param presynaptic_neuron_idx: int, the index of presynaptic neuron in self.neurons
        :param t_point: int, time point in time unit axis of the action potential
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
        given the environment (terrain_map, food_map, and fish_map) and body position of the fish,
        iterate through every neuron in the brain, evaluate if it will fire, if fires, update its
        action potential history (self.simulation_cache["action_histories"] and all the
        psp_waveforms (self.simulation_cache["psp_waveforms"]) of its postsynaptic neurons. Return
        movement_attempt (by the firing of the muscles) and number of total action potentials from
        all neurons at this time point.

        :param t_point: int, current timestamp of time unit axis
        :param body_position: tuple of two ints, (row, col), current position of body center of the fish
        :param terrain_map: 2d array, with only 0s (water) and 1s (land). represents the land scape of the world
        :param food_map: 2d array, with only 0s (no food) and 1s (food). represents the distribution of food
        :param fish_map: not fully implemented right now.
        :return:
          movement_attempt: 1-d array, int, (row, col), representing the movement attempt, be careful, this
            may not represent the actual movement, it will be evaluated by the fish object
            (fish class) containing this brain to see if the movement is possible. if the fish
            is hitting the edge the world map, then the it will not move out of the map
            None: no movement has been attempted.
          total_action_potential_number: int, number of total action potential of the whole brain at this time point.
            [0, num_neurons], each neuron can only have a single action potential at a time point.
        """

        movement_attempt = np.array([0, 0], dtype=int)
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

    def to_h5_group(
        self,
        h5_group: h5py.Group,
        should_save_cache: bool = False,
        should_save_psp_waveforms: bool = False,
    ):
        """
        save brain object to a hdf5 group
        """
        grp_neurons = h5_group.create_group("neurons")
        grp_neurons.attrs["layers"] = list(self.neurons["layer"])
        for neuron_i, neuron_row in self.neurons.iterrows():
            neuron_name = f"neuron_{neuron_i:04d}"
            grp_neuron = grp_neurons.create_group(neuron_name)
            neuron_row["neuron"].to_h5_group(grp_neuron)

        grp_connections = h5_group.create_group("connections")
        connection_mat = []
        amplitudes = []
        for conn_i, conn_row in self.connections.iterrows():
            curr_connection = conn_row["connection"]
            connection_mat.append(
                [
                    conn_row["pre_idx"],
                    conn_row["post_idx"],
                    curr_connection.latency,
                    curr_connection.rise_time,
                    curr_connection.decay_time,
                ]
            )
            amplitudes.append(curr_connection.amplitude)
        dset_conn_matrix = grp_connections.create_dataset(
            name="connection_matrix", data=np.array(connection_mat, dtype=int)
        )
        dset_conn_matrix.attrs["column_names"] = [
            "pre_idx",
            "post_idx",
            "latency",
            "rise_time",
            "decay_time",
        ]
        grp_connections.create_dataset(name="amplitudes", data=amplitudes)

        if should_save_cache:
            grp_sim_cache = h5_group.create_group("simulation_cache")
            self.save_simulation_cache_to_h5_group(
                grp_sim_cache, should_save_psp_waveforms
            )

    def save_simulation_cache_to_h5_group(
        self,
        h5_group: h5py.Group,
        should_save_psp_waveforms: bool = False,
    ):
        grp_action = h5_group.create_group("action_histories")
        if self.simulation_cache["action_histories"] is not None:
            for i, action_history in enumerate(
                self.simulation_cache["action_histories"]
            ):
                neuron_name = f"neuron_{i:04d}"
                grp_action.create_dataset(name=neuron_name, data=action_history)

        if (
            should_save_psp_waveforms
            and self.simulation_cache["psp_waveforms"] is not None
        ):
            psp_dset = h5_group.create_dataset(
                name="psp_waveforms", data=self.simulation_cache["psp_waveforms"]
            )
            psp_dset.attrs["data_format"] = "neuron_ind x time_point"
