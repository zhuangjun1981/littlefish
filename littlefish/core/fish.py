# from __future__ import (absolute_import, division,
#                         print_function, unicode_literals)
# from builtins import *
import random
import numbers
import numpy as np
import pandas as pd
from littlefish.core import utilities as util
import h5py
import os
import matplotlib.pyplot as plt

# unnecessary global varible
#
# SIMULATION_LENGTH = 100000
#
# EYE_GAIN = 0.005
# EYE_BASELINE_RATE = 0.
# EYE_REFRACTORY_PERIOD = 10
# EYE_BORDER_VALUE = 1
# EYE_INPUT_FILTER = np.array([0.2, 0.6, 0.2])
# EYE2_INPUT_FILTER = np.array([0.15, 0.3, 0.15, 0.1, 0.2, 0.1])
# EYE_DIRECTIONS = ['east', 'northeast', 'north', 'northwest', 'west', 'southwest', 'south', 'southeast']
# EYE_INPUT_TYPES = ['terrain', 'food', 'fish']
#
# NEURON_REFRACTORY_PERIOD = 10
# NEURON_BASELINE_RATE = 0.0001
#
# MUSCLE_DIRECTIONS = ['east', 'north', 'west', 'south']
# MUSCLE_REFRACTORY_PERIOD = 5000
# MUSCLE_BASELINE_RATE = 0.00001
#
# CONNECTION_LATENCY = 30
# CONNECTION_AMPLITUDE = 0.0001
# CONNECTION_RISE_TIME = 50
# CONNECTION_DECAY_TIME = 100
#
# FISH_MAX_HEALTH = 100.
# FISH_HEALTH_DECAY_RATE = 0.0001
# FISH_LAND_PENALTY_RATE = 0.005
# FISH_FOOD_RATE = 20


def generate_minimal_brain():
    """

    :return: a Brain object with one eye, two neuron in hidden layer and one muscle
    """

    eye = Eye(
        direction="east",
        input_filter=None,
        gain=0.05,
        input_type="terrain",
        baseline_rate=0.0,
        refractory_period=1.2,
    )

    hidden0 = Neuron(baseline_rate=0.005, refractory_period=1.2)
    hidden1 = Neuron(baseline_rate=0.005, refractory_period=1.2)
    muscle = Muscle(direction="east", baseline_rate=0.1, refractory_period=500)

    neurons = pd.DataFrame(
        [[0, 0, eye], [1, 0, hidden0], [1, 1, hidden1], [2, 0, muscle]],
        columns=["layer", "neuron_ind", "neuron"],
    )

    connection_eye_hidden0 = Connection(
        latency=3, amplitude=0.01, rise_time=5, decay_time=10
    )
    connection_eye_hidden1 = Connection(
        latency=3, amplitude=0.0001, rise_time=5, decay_time=10
    )
    connection_hidden0_muscle = Connection(
        latency=3, amplitude=0.0001, rise_time=5, decay_time=10
    )
    connection_hidden1_muscle = Connection(
        latency=3, amplitude=0.01, rise_time=5, decay_time=10
    )

    conn_0_1 = pd.DataFrame(
        [[connection_eye_hidden0], [connection_eye_hidden1]], columns=[0], index=[1, 2]
    )
    conn_1_2 = pd.DataFrame(
        [[connection_hidden0_muscle, connection_hidden1_muscle]],
        columns=[1, 2],
        index=[3],
    )

    connections = {"L000_L001": conn_0_1, "L001_L002": conn_1_2}

    return Brain(neurons=neurons, connections=connections)


def generate_standard_fish():
    default_config = util.get_default_config()

    brain = genearte_brain_from_brain_config(default_config["brain_config"])

    return Fish(brain=brain, **default_config["fish_config"])


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
        curr_eye = Eye(
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
        curr_muscle = Muscle(
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


def get_eye_type(ind, dir_num=4):
    """
    given the neuron_ind in the eye layer return direction and input type of a specific eye

    :param ind: non-negative int, index of the eye in eye layer
    :param dir_num: 4 or 8, if 4, eye directions iterate through 4 cardinal directions; if 8, eye directions will
                    iterate through 8 directions
    :return: two strings, (direction, type)
    """

    if dir_num == 8:
        eye_directions = [
            "east",
            "northeast",
            "north",
            "northwest",
            "west",
            "southwest",
            "south",
            "southeast",
        ]
    elif dir_num == 4:
        eye_directions = ["east", "north", "west", "south"]
    else:
        raise ValueError("cannot understand dir_num, should be 4 or 8.")

    eye_types = ["terrain", "food", "fish"]
    direction_num = len(eye_directions)
    type_num = len(eye_types)
    return (
        eye_directions[ind % direction_num],
        eye_types[(ind // direction_num) % type_num],
    )


def get_muscle_direction(ind):
    """
    given the neuron_ind in the muscle layer return direction of a specific muscle

    :return: string, direction of the muscle
    """

    muscle_directions = ["east", "north", "west", "south"]
    direction_num = len(muscle_directions)
    return muscle_directions[ind % direction_num]


class Neuron(object):
    """
    a very simple neuron class
    """

    def __init__(self, baseline_rate=0.001, refractory_period=1.2):
        """
        action is the equivalent of action potential in biology, and consider one time unit is 0.1 milisecond

        :param baseline_rate: float, probablity of a action per time unit.
        :param refractory_period: float, refractory_period in time unit
        """

        self._baseline_rate = float(baseline_rate)
        self._refractory_period = float(refractory_period)

    def __str__(self):
        return "littlefish.brain.Neuron object"

    def copy(self):
        """

        :return: a copy of self for i.e. mutation
        """

        return Neuron(
            baseline_rate=self.get_baseline_rate(),
            refractory_period=self.get_refractory_period(),
        )

    def get_baseline_rate(self):
        return self._baseline_rate

    def get_refractory_period(self):
        return self._refractory_period

    def get_neuron_type(self):
        return "neuron"

    def set_baseline_rate(self, new_baseline_rate):
        self._baseline_rate = float(new_baseline_rate)

    def set_refractory_period(self, new_refractory_period):
        self._refractory_period = new_refractory_period

    def act(
        self, t_point, action_history=[], probability_input=0.0, probability_base=None
    ):
        """
        evaluate if the neuron will fire at given time point

        :param t_point: int, current time point as the index of time unit axis
        :param action_history: list of positive integers, list of time stamps of actions of this neuron,
                               should be monotonically increasing
        :param probability_input: float, summed connection inputs, as add on to baseline_rate
        :param probability_base: float, a random number no less than 0 and less than 1, to determine if the neuron
                                 is going to act or not, if None, a random number will be generated by
                                 random.random()
        :return: bool, True: fire; False: quite
        """

        if probability_base is None:
            probability_base = random.random()

        # this block of code is commented out to speed up simulation
        # else:
        #     if probability_base < 0. or probability_base >= 1.:
        #         raise ValueError('Neuron: probability_base should be no less than 0 and less than 1.')

        # this block of code is commented out to speed up simulation
        # if len(action_history) >= 2:
        #     if not util.check_monotonicity(np.array(action_history), direction='increasing'):
        #         raise ValueError('Neuron: action history should be monotonically increasing.')

        if (
            len(action_history) > 0
            and t_point - action_history[-1] < self._refractory_period
        ):
            return False
        else:
            curr_rate = self._baseline_rate + probability_input
            if probability_base <= curr_rate:
                action_history.append(t_point)
                return True
            else:
                return False

    def to_h5_group(self, h5_group):
        br_dset = h5_group.create_dataset("baseline_rate", data=self._baseline_rate)
        br_dset.attrs["unit"] = "action_per_time_unit"
        rp_dset = h5_group.create_dataset(
            "refractory_period", data=self._refractory_period
        )
        rp_dset.attrs["unit"] = "time_unit"
        h5_group.attrs["neuron_type"] = "neuron"

    @staticmethod
    def from_h5_group(h5_group):
        neuron_type = util.decode(h5_group.attrs["neuron_type"])

        if neuron_type != "neuron":
            raise ValueError(
                'Neuron: loading from h5 file failed. "neuron_type" attribute should be "neuron".'
            )

        neuron = Neuron(
            baseline_rate=h5_group["baseline_rate"][()],
            refractory_period=h5_group["refractory_period"][()],
        )
        return neuron


class Eye(Neuron):
    """
    Eye class to observe the environment and the inside of fish body, subclass of Neuron, has eye sight as 2 pixels
    """

    def __init__(
        self,
        direction,
        input_filter=None,
        gain=0.001,
        input_type="terrain",
        baseline_rate=0.0,
        refractory_period=1.2,
    ):
        """
        for a fish occupies 3x3 space, consider the eyes are in the outer rim of the body (the 4 pixels surrounding the
        central pixel in 4 cardinal direction. Each eye receives the inputs from the given direction in the environment.
        for example:

        fish (1) in the environment(0):
        0 0 0 0 0 0 0
        0 0 0 0 0 0 0
        0 0 1 1 1 0 0
        0 0 1 1 1 0 0
        0 0 1 1 1 0 0
        0 0 0 0 0 0 0
        0 0 0 0 0 0 0

        eye (2) in the east direction is:
        0 0 0 0 0 0 0
        0 0 0 0 0 0 0
        0 0 1 1 1 0 0
        0 0 1 1 2 0 0
        0 0 1 1 1 0 0
        0 0 0 0 0 0 0
        0 0 0 0 0 0 0

        it receive inputs from pixels labelled as 3 in the environment:
        0 0 0 0 0 0 3
        0 0 0 0 0 3 3
        0 0 0 0 3 3 3
        0 0 0 3 3 3 3
        0 0 0 0 3 3 3
        0 0 0 0 0 3 3
        0 0 0 0 0 0 3

        the inputs from the environment (1x16 array) will be filtered by a array with same size, to generate
        a single value as the base of its input. this value will be multiplied by a float number gain to generate
        final input probability.

        self._get_input_pixels will pick pixels within the eye's receptive field in the following order:
        0 0 0 0 0 0 10
        0 0 0 0 0 5 11
        0 0 0 0 2 6 12
        0 0 0 1 3 7 13
        0 0 0 0 4 8 14
        0 0 0 0 0 9 15
        0 0 0 0 0 0 16

        :param direction: str, the aim of the eye, should be one of the following, 'east', 'north', 'west' and 'south'
        :param input_filter: 1d array, shape: (16,), filter to transform input pixel values to a single number.
                             analog of linear receptive field of a retinal ganglion cell. default input_filter is
                             set as the following:

                             input_filter[pixel_i] = exp(-1 * cartesian_distance(pixel_i, pixel_body_center)
        :param baseline_rate: float, probablity of a action per time unit.
        :param refractory_period: float, refractory_period in time unit
        :param input_type: str, type of the input the eye receives, should be one of 'terrain', 'food', 'fish',
               default: 'terrain'
        """

        self._direction = direction
        self._rf_positions = self._get_rf_positions()
        self._gain = float(gain)

        if input_filter is None:
            self._input_filter = np.array(
                [
                    1,
                    0.243116734,
                    0.367879441,
                    0.243116734,
                    0.059105747,
                    0.106877926,
                    0.135335283,
                    0.106877926,
                    0.059105747,
                    0.014369596,
                    0.027172461,
                    0.04232922,
                    0.049787068,
                    0.04232922,
                    0.027172461,
                    0.014369596,
                ]
            )
        else:
            self._input_filter = input_filter.astype(np.float32)

        if input_type in ["terrain", "food", "fish"]:
            self._input_type = input_type
        else:
            raise ValueError(
                'Eye: type should be one of the following: "terrain", "food", "fish".'
            )

        curr_baseline_rate = float(baseline_rate)
        curr_refractory_period = float(refractory_period)

        super(Eye, self).__init__(
            baseline_rate=curr_baseline_rate, refractory_period=curr_refractory_period
        )

    def __str__(self):
        return "littlefish.brain.Eye object"

    def copy(self):
        """

        :return: a copy of self for i.e. mutation
        """

        return Eye(
            direction=self.get_direction(),
            input_filter=self.get_input_filter(),
            gain=self.get_gain(),
            input_type=self.get_input_type(),
            baseline_rate=self.get_baseline_rate(),
            refractory_period=self.get_refractory_period(),
        )

    def get_direction(self):
        return self._direction

    def get_input_filter(self):
        return self._input_filter

    def get_gain(self):
        return self._gain

    def get_input_type(self):
        return self._input_type

    def get_neuron_type(self):
        return "eye"

    def _get_rf_positions(self):
        """
        get pixel coordinate of receptive field

        :return rf_positions: 2d array, shape: (16, 2), _dtype: int64. each row is a pixel in the receptive field,
                              [row, col] relative to the position of the body center pixel
        """

        array1 = np.array(
            [0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3], dtype=np.int32
        )
        array2 = np.array(
            [0, -1, 0, 1, -2, -1, 0, 1, 2, -3, -2, -1, 0, 1, 2, 3], dtype=np.int32
        )

        if self._direction == "east":
            rf_pos = np.array([array2, array1]).transpose()
        elif self._direction == "north":
            rf_pos = np.array([array1 * -1, array2]).transpose()
        elif self._direction == "west":
            rf_pos = np.array([array2, array1 * -1]).transpose()
        elif self._direction == "south":
            rf_pos = np.array([array1, array2]).transpose()
        else:
            raise ValueError(
                "Eye: direction should be one of the following: ['east', 'north', 'west', 'south']."
            )

        return rf_pos

    def _get_input_pixels(self, input_map, body_position, border_value=1):
        """
        get a 1d array of all pixels within the eye's receptive field

        :param input_map: 2d array, the map the eye is looking
        :param body_position: list of two non-negative integers, [row, col] location of body center pixel
        :param border_value: float, values for pixels outside the border of input_map
        :return: the 1d array with the values of the 16 pixels in the eye's receptive field. pixels out of the
        input_map range will be returned as border_value
        """

        body_pos = np.array(body_position, dtype=np.int32)
        input_pixels = np.zeros(16, dtype=np.float32)

        for ind, curr_pos in enumerate(self._rf_positions):
            curr_rf_pos = body_pos + curr_pos
            try:
                if curr_rf_pos[0] < 0 or curr_rf_pos[1] < 0:
                    input_pixels[ind] = border_value
                else:
                    input_pixels[ind] = input_map[curr_rf_pos[0], curr_rf_pos[1]]
            except IndexError:
                input_pixels[ind] = border_value

        return input_pixels

    def _get_input(self, input_map, body_position, border_value=1):
        """

        :return: float, calculate real time input from the visual field
        """

        input_pixels = self._get_input_pixels(
            input_map, body_position, border_value=border_value
        )
        probability_input = self._gain * np.sum(input_pixels * self._input_filter)

        return probability_input

    def act(
        self,
        t_point,
        body_position,
        input_map,
        action_history=[],
        border_value=1,
        probability_base=None,
    ):
        """
        evaluate if the eye neuron will fire at given time point

        :param t_point: int, current time point as the index of time unit axis
        :param body_position: tuple of two ints, (row, col),  position of the body center pixel
        :param input_map: binary 2-d map, for now it should only contain 0s and 1s
        :param action_history: list of positive integers, list of time stamps of actions of this neuron,
                               should be monotonically increasing
        :param border_value: int, default 1, value for pixels outside the terrain_map
        :param probability_base: float, a random number no less than 0 and less than 1, to determine if the neuron
                                 is going to act or not, if None, a random number will be generated by
                                 random.random()
        :return: bool, True: fire; False: quite
        """

        probability_input = self._get_input(
            input_map=input_map, body_position=body_position, border_value=border_value
        )

        return super(Eye, self).act(
            t_point,
            action_history=action_history,
            probability_input=probability_input,
            probability_base=probability_base,
        )

    def to_h5_group(self, h5_group):
        br_dset = h5_group.create_dataset("baseline_rate", data=self._baseline_rate)
        br_dset.attrs["unit"] = "action_per_time_unit"
        rp_dset = h5_group.create_dataset(
            "refractory_period", data=self._refractory_period
        )
        rp_dset.attrs["unit"] = "time_unit"
        h5_group.create_dataset("direction", data=self._direction)
        h5_group.create_dataset("input_filter", data=self._input_filter)
        h5_group.create_dataset("gain", data=self._gain)
        h5_group.create_dataset("input_type", data=self._input_type)
        rf_pos_dset = h5_group.create_dataset("rf_positions", data=self._rf_positions)
        rf_pos_dset.attrs["data_format"] = "16 (pixel_num) x 2 ([row, col])"
        rf_pos_dset.attrs["description"] = (
            "coordinates of each pixel of this eye's receptive field relative to body "
            "center position"
        )
        h5_group.attrs["neuron_type"] = "eye"

    @staticmethod
    def from_h5_group(h5_group):
        if util.decode(h5_group.attrs["neuron_type"]) != "eye":
            raise ValueError(
                'Eye: loading from h5 file failed. "neuron_type" attribute should be "eye".'
            )

        direction = util.decode(h5_group["direction"][()])
        input_type = util.decode(h5_group["input_type"][()])

        eye = Eye(
            direction=direction,
            input_filter=h5_group["input_filter"][()],
            gain=h5_group["gain"][()],
            input_type=input_type,
            baseline_rate=h5_group["baseline_rate"][()],
            refractory_period=h5_group["refractory_period"][()],
        )
        return eye


class Muscle(Neuron):
    """
    muscle class for determining the motion of the fish. Subclass of Neuron class
    """

    def __init__(self, direction, baseline_rate=0.001, refractory_period=500.0):
        self._direction = direction

        if self._direction == "east":
            self._step_motion = np.array([0, 1], dtype=np.int8)
        elif self._direction == "north":
            self._step_motion = np.array([-1, 0], dtype=np.int8)
        elif self._direction == "west":
            self._step_motion = np.array([0, -1], dtype=np.int8)
        elif self._direction == "south":
            self._step_motion = np.array([1, 0], dtype=np.int8)
        else:
            raise ValueError(
                "self.direction should be one of the following: "
                "['east', 'north', 'west', 'south']."
            )

        super(Muscle, self).__init__(
            baseline_rate=baseline_rate, refractory_period=refractory_period
        )

    def __str__(self):
        return "littlefish.brain.Muscle object"

    def copy(self):
        """

        :return: a copy of self for i.e. mutation
        """

        return Muscle(
            direction=self.get_direction(),
            baseline_rate=self.get_baseline_rate(),
            refractory_period=self.get_refractory_period(),
        )

    def get_neuron_type(self):
        return "muscle"

    def get_direction(self):
        return self._direction

    def act(
        self, t_point, action_history=[], probability_input=0.0, probability_base=None
    ):
        """
        evaluate if the muscle will try to move the fish or not

        :param t_point: int, current time point as the index of time unit axis
        :param probability_input: float, summed connection inputs, as add on to baseline_rate
        :param action_history: list of positive integers, list of time stamps of actions of this neuron,
                               should be monotonically increasing
        :param probability_base: float, a random number no less than 0 and less than 1, to determine if the neuron
                                 is going to act or not, if None, a random number will be generated by
                                 random.random()
        :return: no attempt: False
                 attempt: movement vector, 1d array with two ints, [row_update, col_update]
        """

        is_act = super(Muscle, self).act(
            t_point,
            action_history=action_history,
            probability_input=probability_input,
            probability_base=probability_base,
        )

        if is_act:
            return self._step_motion
        else:
            return is_act

    def to_h5_group(self, h5_group):
        br_dset = h5_group.create_dataset("baseline_rate", data=self._baseline_rate)
        br_dset.attrs["unit"] = "action_per_time_unit"
        rp_dset = h5_group.create_dataset(
            "refractory_period", data=self._refractory_period
        )
        rp_dset.attrs["unit"] = "time_unit"
        h5_group.create_dataset("direction", data=self._direction)
        h5_group.attrs["neuron_type"] = "muscle"

    @staticmethod
    def from_h5_group(h5_group):
        if util.decode(h5_group.attrs["neuron_type"]) != "muscle":
            raise ValueError(
                'Muscle: loading from h5 file failed. "neuron_type" attribute should be "muscle".'
            )

        direction = util.decode(h5_group["direction"][()])
        muscle = Muscle(
            direction=direction,
            baseline_rate=h5_group["baseline_rate"][()],
            refractory_period=h5_group["refractory_period"][()],
        )
        return muscle


class Connection(object):
    """
    synaptic connection between two neurons
    """

    def __init__(self, latency=3, amplitude=0.001, rise_time=5, decay_time=10):
        """

        :param latency: int, temporal latency from presynaptic neuron action to the postsynaptic effect onset, number
                        of time units
        :param amplitude: float, peak change of the firing rate in the postsynaptic neuron, probablity of a action per
                          time unit. can be positive (excitatiory) or negative (inhibitory)
        :param rise_time: int, temporal duration from onset to peak, number of time units
        :param decay_time: int, temporal duration from peak to baseline, number of time units
        """

        if latency is not None:
            if not util.is_integer(latency):
                raise ValueError("latency should be an integer.")
            self._latency = int(latency)

        if amplitude is not None:
            self._amplitude = float(amplitude)

        if rise_time is not None:
            if not util.is_integer(rise_time):
                raise ValueError("rise_time should be an integer.")
            self._rise_time = int(rise_time)

        if decay_time is not None:
            if not util.is_integer(decay_time):
                raise ValueError("decay_time should be an integer.")
            self._decay_time = int(decay_time)

        self._generate_psp()

    def __str__(self):
        return "littlefish.brain.Connection object"

    def copy(self):
        """

        :return: a copy of self for i.e. mutation
        """

        return Connection(
            latency=self.get_latency(),
            amplitude=self.get_amplitude(),
            rise_time=self.get_rise_time(),
            decay_time=self.get_decay_time(),
        )

    def get_latency(self):
        return self._latency

    def set_latency(self, new_latency):
        if new_latency is not None:
            if not util.is_integer(new_latency):
                raise ValueError("new_latency should be an integer.")
            self._latency = int(new_latency)
            self._generate_psp()

    def get_amplitude(self):
        return self._amplitude

    def set_ampletude(self, new_amplitude):
        if new_amplitude is not None:
            self._amplitude = float(new_amplitude)
            self._generate_psp()

    def get_rise_time(self):
        return self._rise_time

    def set_rise_time(self, new_rise_time):
        if new_rise_time is not None:
            if not util.is_integer(new_rise_time):
                raise ValueError("new_rise should be an integer.")
            self._rise_time = int(new_rise_time)
            self._generate_psp()

    def get_decay_time(self):
        return self._decay_time

    def set_decay_time(self, new_decay_time):
        if new_decay_time is not None:
            if not util.is_integer(new_decay_time):
                raise ValueError("new_decay should be an integer.")
            self._decay_time = int(new_decay_time)
            self._generate_psp()

    def _generate_psp(self):
        """

        generate post synaptic probability wave form
        """

        self._psp = np.zeros(self._latency + self._rise_time + self._decay_time)
        self._psp[self._latency : self._latency + self._rise_time] = (
            self._amplitude
            * (np.arange(self._rise_time) + 1).astype(np.float32)
            / float(self._rise_time)
        )

        self._psp[-self._decay_time :] = (
            self._amplitude
            * (np.arange(self._decay_time, 0, -1) - 1).astype(np.float32)
            / float(self._decay_time)
        )

    def get_psp(self):
        return self._psp

    def set_params(self, latency=None, amplitude=None, rise_time=None, decay_time=None):
        """
        set new parameters and regenerate psp waveform

        :param latency: int, number of time units for time delay
        :param amplitude: float, peak probability
        :param rise_time: int, number of time units to rise to peak
        :param decay_time: int, number of time units to decay to baseline
        """

        changed = False

        if latency is not None:
            if not isinstance(latency, int):
                raise ValueError("latency should be an integer.")
            self._latency = latency
            changed = True

        if amplitude is not None:
            self._amplitude = float(amplitude)
            changed = True

        if rise_time is not None:
            if not isinstance(rise_time, int):
                raise ValueError("rise_time should be an integer.")
            self._rise_time = rise_time
            changed = True

        if decay_time is not None:
            if not isinstance(decay_time, int):
                raise ValueError("decay_time should be an integer.")
            self._decay_time = decay_time
            changed = True

        if changed:
            self._generate_psp()
        else:
            print("Brain.Connection: no parameter has been changed. Do nothing.")

    def act(self, t_point, postsynaptic_index, psp_waveforms):
        """
        if the presynaptic neuron fires at the 'time_point', a psp wave form will be generated and add to the
        input waveform of postsynaptic neuron defined by postsynaptic_index

        :param t_point: int, current time point as the index of time unit axis
        :param postsynaptic_index: uint, the index of postsynaptic neuron
        :param psp_waveforms: 2-d array, float 32, the psp waveforms of all neurons in the brain, neuron id x t-point,
                              the generated psp will be added to the postsynaptic_index th line of the array
        :return:
        """

        psp_end = t_point + len(self._psp)
        if psp_end <= psp_waveforms.shape[1]:
            psp_waveforms[postsynaptic_index, t_point:psp_end] += self._psp
        else:
            psp_waveforms[postsynaptic_index, t_point:] += self._psp[
                : psp_waveforms.shape[1] - t_point
            ]


class Brain(object):
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
            self._neurons = min_brain.get_neurons()
            self._connections = min_brain.get_connections()
        else:
            self._neurons = neurons
            self._connections = connections

        self.check_integrity(verbose=False)

        # print('Brain: littlefish.core.fish.Brain created successfully.')

    def __str__(self):
        return "littlefish.brain.Brain object"

    def copy(self):
        """

        :return: a copy of self for i.e. mutation
        """

        return Brain(
            neurons=self.get_neurons().copy(), connections=dict(self.get_connections())
        )

    def get_neurons(self):
        return self._neurons

    @property
    def layer_num(self):
        return int(round(max(self._neurons["layer"]))) + 1

    def get_layer_type(self, layer):
        """

        :return: layer type (str) given the layer number
        """

        if not isinstance(layer, int):
            raise ValueError("Input layer number should be integer.")

        if layer == 0:
            return "eye"
        elif layer == self.layer_num - 1:
            return "muscle"
        elif 0 < layer < self.layer_num - 1:
            return "hidden" + util.int2str(layer, 3)
        else:
            raise ValueError("layer number out of range.")

    def get_neuron_type(self, ind):
        """
        return neuron type as a pair of strings given the index in self._neurons

        :param ind: int
        :return: for eyes : ('eye', type + short of direction)
                 for hidden neurons: ('hidden', str(layer))
                 for muscles ('muscle', short of direction)
        """

        # self.check_integrity_neurons()

        curr_row = self._neurons.loc[ind]
        curr_layer = curr_row["layer"]
        if curr_layer == 0:  # eye layer
            curr_dir, curr_type = get_eye_type(curr_row["neuron_ind"])
            return (
                util.short("eye")
                + "_"
                + util.short(curr_type)
                + "_"
                + util.short(curr_dir)
            )
        elif curr_layer == self.layer_num - 1:  # muscle layer
            curr_dir = get_muscle_direction(curr_row["neuron_ind"])
            return util.short("muscle") + "_" + util.short(curr_dir)
        elif 0 < curr_layer < self.layer_num - 1:
            curr_layer_num = util.int2str(curr_layer, 3)
            curr_neuron_num = util.int2str(curr_row["neuron_ind"], 3)
            return "_".join([util.short("hidden"), curr_layer_num, curr_neuron_num])
        else:
            raise ValueError("layer number out of range.")

    def get_connections(self):
        return self._connections

    def get_postsynaptic_neuron_inds(self, neuron_ind):
        neuron_layer = int(round(self._neurons.loc[neuron_ind, "layer"]))
        if neuron_layer < 0:
            raise ValueError("Brain: invalid layer. less than 0.")
        elif neuron_layer == self.layer_num - 1:
            print("Brain: cannot fine postsynaptic neuron of neurons in muscle layer.")
        else:
            postsynaptic_neuron_ind = self._neurons[
                self._neurons["layer"] == neuron_layer + 1
            ].index.tolist()
            postsynaptic_neuron_ind.sort()
            return postsynaptic_neuron_ind

    def get_presynaptic_neuron_inds(self, neuron_ind):
        neuron_layer = int(round(self._neurons.loc[neuron_ind, "layer"]))
        if neuron_layer < 0:
            raise ValueError("Brain: invalid layer. less than 0.")
        elif neuron_layer == 0:
            print("Brain: cannot fine presynaptic neuron of neurons in eye layer.")
        else:
            presynaptic_neuron_ind = self._neurons[
                self._neurons["layer"] == neuron_layer - 1
            ].index.tolist()
            presynaptic_neuron_ind.sort()
            return presynaptic_neuron_ind

    def get_single_connection(self, pre_neuron_ind, post_neuron_ind):
        pre_layer = int(round(self._neurons.loc[pre_neuron_ind, "layer"]))
        post_layer = int(round(self._neurons.loc[post_neuron_ind, "layer"]))

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

        inds = self._neurons[self._neurons["layer"] == layer].index.tolist()
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
        if not util.check_df_index(self._neurons):
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
                    'Brain: the "layer" in self._neurons is not in ascending order.'
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


class Fish(object):
    """
    the main fish class

    a 'fish' has a body occupies a 3x3 space.

    a 'fish' has health point (goes down over time and increases after eating a food). fish moves around in a 2d
    landscape (2-d binary map, 0 represents water, 1 represents land), when hits land, health point will go down
    quickly.

    the purpose of simulation is to train the fish use its eyes, brain and muscles to avoid land and look for food.

    the simulation works on "real-time" basis on a time unit axis (consider one time unit is equivalent to 0.1
    millisecond.

    :attr _brain: a brain.Brain object
    :attr _max_health: float, maximum health point a fish can have
    :attr _health_decay_rate: float, the constant rate of health reduction, health point / time unit
    :attr _land_penalty_rate: float, the penalty of health point, if the fish's body covers land pixels (1s) in
                                   the terrain map, health point / (pixel * time unit)
    :attri _food_rate: float, the gaining of health point if fish's body covers food pixels (1s) in the food map,
                           health point / pixel. the food after taken will disappear, so no health gaining is a
                           transient event
    """

    def __init__(
        self,
        name=None,
        mother_name=None,
        brain=None,
        max_health=100.0,
        health_decay_rate=0.0001,
        land_penalty_rate=0.005,
        food_rate=20.0,
    ):
        """

        :param brain: a brain.Brain object
        :param max_health: float, maximum health point a fish can have
        :param health_decay_rate: float, the constant rate of health reduction, health point / time unit
        :param land_penalty_rate: float, the penalty of health point, if the fish's body covers land pixels (1s) in
                                  the terrain map, health point / (pixel * time unit)
        :param food_rate: float, the gaining of health point if fish's body covers food pixels (1s) in the food map,
                          health point / pixel. the food after taken will disappear, so no health gaining is a
                          transient event
        """

        # print('\nFish: Creating littlefish.core.fish.Fish object.')

        if name is None:
            self._name = "test_fish"
        else:
            self._name = name

        if mother_name is None:
            self._mother_name = ""
        else:
            self._mother_name = mother_name

        self._max_health = float(max_health)
        self._health_decay_rate = float(health_decay_rate)
        self._land_penalty_rate = land_penalty_rate
        self._food_rate = food_rate

        if brain is None:
            self._brain = Brain()
        else:
            brain.check_integrity(verbose=False)
            self._brain = brain

        # print('Fish: littlefish.core.fish.Fish object created successfully.')

    def copy(self):
        """

        :return: a copy of self for i.e. mutation
        """

        return Fish(
            name=self.get_name(),
            mother_name=self.get_mother_name(),
            brain=self.get_brain(),
            max_health=self.get_max_health(),
            health_decay_rate=self.get_health_decay_rate(),
            food_rate=self.get_food_rate(),
        )

    def get_name(self):
        return self._name

    @property
    def name(self):
        return self.get_name()

    def get_mother_name(self):
        return self._mother_name

    def get_brain(self):
        return self._brain

    def get_max_health(self):
        return self._max_health

    def get_health_decay_rate(self):
        return self._health_decay_rate

    def get_land_penalty_rate(self):
        return self._land_penalty_rate

    def get_food_rate(self):
        return self._food_rate

    def set_name(self, name):
        self._name = name

    def set_brain(self, brain):
        brain.check_integrity(verbose=False)
        self._brain = brain

    def set_food_rate(self, food_rate):
        self._food_rate = float(food_rate)

    def set_health_decay_rate(self, health_decay_rate):
        self._health_decay_rate = health_decay_rate

    def act(
        self,
        t_point,
        curr_position,
        curr_health,
        action_histories,
        psp_waveforms,
        terrain_map,
        food_map=None,
        fish_map=None,
    ):
        """
        simulate the fish's action at a given time point

        :param t_point: non-negative int, time point
        :param curr_position: list or tuple of two non-negative integers, coordinates of fish center position,
                              [row, col]
        :param curr_health: positive float, health point at the beginning of t_point
        :param action_histories: data frame of lists, each list is the action history of a particular neuron, in the
                                 same order as self._neurons data frame, columns = ['action_history'], used by
                                 self._brain.act()
        :param psp_waveforms: 2d-array of floats, psp waveforms of all neurons in the brain, each row represents one
                              neuron in the same order as self._neurons data frame, each column represents a time point,
                              used by self._brain.act()
        :param terrain_map: 2d array, with only 0s (water) and 1s (land). represents the land scape of the world
        :param food_map: 2d array, with only 0s (no food) and 1s (food). represents the distribution of food
        :param fish_map: not fully implemented right now.
        :return updated_health: float, health point at the end of t_point
        :return movement_attempt: list of two integers, the attempt the fish is trying to move [row_shift, col_shift].
                                  None if updated_health is below 0 (fish is dead).
        """

        updated_health = float(curr_health)

        # evaluate food
        if food_map is not None:
            body_food_overlap = self._eval_food(
                food_map=food_map, curr_position=curr_position
            )
            updated_health = self._eat_food(
                body_food_overlap=body_food_overlap, curr_health=updated_health
            )
            food_eated = body_food_overlap

            # update food map
            food_map[
                curr_position[0] - 1 : curr_position[0] + 2,
                curr_position[1] - 1 : curr_position[1] + 2,
            ] = 0
        else:
            food_eated = 0

        # evaluate the extend of how much of the fish is on the land
        body_land_overlap = self._eval_terrain(
            terrain_map=terrain_map, curr_position=curr_position
        )

        # update current health with land penalty
        updated_health -= body_land_overlap * self._land_penalty_rate

        # ----------------- not implemented --------------------------
        # if fish_map is not None:
        #     self._eval_fish(fish_map=fish_map)
        # ----------------- not implemented --------------------------

        # update health
        updated_health = updated_health - self._health_decay_rate

        if updated_health > 0:  # still alive
            movement_attempt = self._brain.act(
                t_point=t_point,
                action_histories=action_histories,
                psp_waveforms=psp_waveforms,
                body_position=curr_position,
                terrain_map=terrain_map,
                food_map=food_map,
                fish_map=fish_map,
            )
        else:
            movement_attempt = None

        return updated_health, movement_attempt, food_eated

    @staticmethod
    def _eval_terrain(terrain_map, curr_position):
        """
        Evaluate the coverage of fish body on terrain map, return the sum of all terrain pixels that are covered by
        the fish body
        """

        if terrain_map is None:
            raise ValueError("Fish: _eval_terrain failure. terrain_map is None.")
        else:
            curr_body = terrain_map[
                curr_position[0] - 1 : curr_position[0] + 2,
                curr_position[1] - 1 : curr_position[1] + 2,
            ]
            body_land_overlap = np.sum(curr_body.flat)
        return body_land_overlap

    @staticmethod
    def _eval_food(food_map, curr_position):
        """
        find out how many foods are covered by the fish body

        :param food_map: 2d array, binary map of current food
        :param curr_position: tuple of two positive ints, (row, col) of current location of fish
        :return: non-negative int, number of food taken
        """

        curr_body = food_map[
            curr_position[0] - 1 : curr_position[0] + 2,
            curr_position[1] - 1 : curr_position[1] + 2,
        ]
        body_food_overlap = np.sum(curr_body.flat)

        return body_food_overlap

    @staticmethod
    def _eval_fish(self, fish_map):
        """currently not implemented"""

        pass

    def _eat_food(self, body_food_overlap, curr_health):
        """
        count the number of food to be taken, add relevant HP to curr_health, but not exceed the maximum health
        """

        if body_food_overlap == 0:
            return curr_health
        else:
            updated_health = curr_health + self._food_rate * body_food_overlap
            if updated_health > self._max_health:
                updated_health = self._max_health
            return updated_health

    def to_h5_group(self, h5_grp):
        h5_grp.create_dataset("name", data=self._name)
        h5_grp.create_dataset("mother_name", data=self._mother_name)
        h5_grp.create_dataset("max_health", data=self._max_health)
        h5_grp.create_dataset("health_decay_rate_per_tu", data=self._health_decay_rate)
        h5_grp.create_dataset(
            "land_penalty_rate_per_pixel_tu", data=self._land_penalty_rate
        )
        h5_grp.create_dataset("food_rate_per_pixel", data=self._food_rate)
        brain_group = h5_grp.create_group("brain")
        self._brain.to_h5_group(brain_group)

    @staticmethod
    def from_h5_group(h5_grp):
        brain_grp = h5_grp["brain"]
        curr_brain = Brain.from_h5_group(brain_grp)
        curr_name = util.decode(h5_grp["name"][()])
        curr_mother_name = util.decode(h5_grp["mother_name"][()])
        curr_max_health = h5_grp["max_health"][()]
        curr_health_decay_rate = h5_grp["health_decay_rate_per_tu"][()]
        curr_land_penalty_rate = h5_grp["land_penalty_rate_per_pixel_tu"][()]
        curr_food_rate = h5_grp["food_rate_per_pixel"][()]

        curr_fish = Fish(
            name=curr_name,
            mother_name=curr_mother_name,
            brain=curr_brain,
            max_health=curr_max_health,
            health_decay_rate=curr_health_decay_rate,
            land_penalty_rate=curr_land_penalty_rate,
            food_rate=curr_food_rate,
        )

        return curr_fish


if __name__ == "__main__":
    # =========================================================================================
    # starting_position = (10, 10)
    # terrain_map = np.zeros((20, 20), _dtype=np.uint8)
    #
    # fish = Fish()
    # fish.initialize_simulation(starting_position=starting_position, terrain_map=terrain_map)
    # print(fish.get_curr_health())
    # print(fish.get_curr_position())
    # fish.act(0, terrain_map=terrain_map)
    # =========================================================================================

    # =========================================================================================
    # save_path = r"G:\little_fish_test\fish.hdf5"
    # if os.path.isfile(save_path):
    #     os.remove(save_path)
    # fish_group = h5py.File(save_path).create_group('fish')
    #
    # fish = Fish()
    # fish.to_h5_group(fish_group)
    # =========================================================================================

    # =========================================================================================
    # dfile = h5py.File(r"F:\littlefish\test_folder\neuron_test.hdf5")
    # neuron_group = dfile.create_group('test_neuron')
    # neuron = Neuron()
    # for i in range(SIMULATION_LENGTH):
    #     neuron.act(i)
    # neuron.to_h5_group(neuron_group)
    # =========================================================================================

    # =========================================================================================
    # connection = Connection(amplitude=10, latency=5)
    # print(connection.get_psp())
    # =========================================================================================

    # =========================================================================================
    # SIMULATION_LENGTH = 50
    # postsynaptic_input = np.zeros(SIMULATION_LENGTH)
    # connection = Connection(amplitude=10, latency=5, rise_time=5, decay_time=10)
    # connection.act(2, postsynaptic_input)
    # print(postsynaptic_input)
    # connection.act(4, postsynaptic_input)
    # print(postsynaptic_input)
    # connection.act(40, postsynaptic_input)
    # print(postsynaptic_input)
    # =========================================================================================

    # =========================================================================================
    # SIMULATION_LENGTH = 5000
    # neuron_pre = Neuron(baseline_rate=0.005)
    # neuron_post = Neuron(baseline_rate=0.000)
    # connection = Connection(amplitude=1, latency=5, rise_time=1, decay_time=1)
    #
    # postsynaptic_input = np.zeros(SIMULATION_LENGTH)
    #
    # for i in range(SIMULATION_LENGTH):
    #
    #     is_firing = neuron_pre.act(i)
    #     if is_firing:
    #         connection.act(i, postsynaptic_input)
    #     neuron_post.act(i, probability_input=postsynaptic_input[i])
    #
    # spk_train_pre = neuron_pre.get_action_history()
    # spk_train_post = neuron_post.get_action_history()
    #
    # # print(postsynaptic_input)
    # print(len(spk_train_pre))
    # print(len(spk_train_post))
    #
    # ccg, t = util.discreat_crosscorrelation(np.array(spk_train_pre), np.array(spk_train_post))
    # plt.bar(t, ccg)
    # plt.show()
    # =========================================================================================

    # =========================================================================================
    # SIMULATION_LENGTH = 100000
    #
    # terrain_map = np.zeros((5, 5), _dtype=np.uint8)
    # terrain_map[3, 3] = 1
    # print(terrain_map)
    #
    # eye = Eye(position=(2, 3), direction='south')
    # print(eye._get_input(terrain_map=terrain_map))
    #
    # for i in range(SIMULATION_LENGTH):
    #     eye.act(i, terrain_map=terrain_map)
    # print(len(eye.get_action_history()))
    # =========================================================================================

    # =========================================================================================
    # SIMULATION_LENGTH = 20000
    # muscle = Muscle(direction='east', baseline_rate=0., refractory_period=5000)
    # movements = []
    # for i in range(SIMULATION_LENGTH):
    #     movement=muscle.act(i, probability_input=0.5, probability_base=0.1)
    #     if movement:
    #         movements.append(movement)
    # print(movements)
    # print(muscle.get_action_history())
    # =========================================================================================

    # =========================================================================================
    # brain = Brain()
    # neurons_df = brain.generate_default_neurons_df()
    # connections_df = brain.generate_default_connections_df(neurons_df)
    # =========================================================================================

    # =========================================================================================
    # SIMULATION_LENGTH = 100000
    #
    # terrain_map = np.zeros((5, 5), _dtype=np.uint8)
    # terrain_map[3, 3] = 1
    # print(terrain_map)
    #
    # eye = Eye2(direction='south')
    # position = (2, 3)
    # print(eye._get_input_pixels(position=position, terrain_map=terrain_map))
    # print(eye._get_input(position=position, terrain_map=terrain_map))
    #
    # for i in range(SIMULATION_LENGTH):
    #     eye.act(t_point=i, position=position, terrain_map=terrain_map)
    # print(len(eye.get_action_history()))
    # =========================================================================================

    # =========================================================================================
    # brain = Brain()
    # print(brain.has_action_histories())
    # brain._generate_empty_psp_waveforms()
    # print(brain.get_neuron_inds_in_layer(2))
    # print(brain.get_all_presynaptic_neuron_indices())
    # print(brain.get_all_postsynaptic_neuron_indices())
    # print(brain.get_eye_type(13))
    # print(brain.layer_num)
    # print(brain.get_postsynaptic_neuron_inds(8))
    # print(brain.get_presynaptic_neuron_inds(8))
    # print(brain.get_single_connection(8, 13))
    # print(brain.get_single_connection(8, 16))
    # print(brain.get_neuron_inds_in_layer(3))
    # #
    # test_file_path = r"F:\littlefish\test_folder\brain_test.hdf5"
    # if os.path.isfile(test_file_path):
    #     os.remove(test_file_path)
    # test_file = h5py.File(test_file_path)
    # brain_group = test_file.create_group('brain')
    # brain.to_h5_group(brain_group)
    # test_file.close()
    # =========================================================================================

    # =========================================================================================
    # test_file = h5py.File(r"F:\littlefish\test_folder\brain_test.hdf5")
    # brain = Brain.from_h5_group(test_file['brain'])
    # =========================================================================================

    # =========================================================================================
    # dfile = h5py.File(r"F:\littlefish\test_folder\brain_test.hdf5")
    # neuron = Neuron.from_h5_group(dfile['brain/neurons/neuron_00008'])
    # print(neuron.get_action_history())
    # =========================================================================================

    # =========================================================================================
    # bb = Brain()
    # print(bb)
    # print(len(bb.get_connections()))
    # print(bb.get_connections()['L000_L001'])
    # =========================================================================================

    # =========================================================================================
    # mb = Brain()
    # eah = mb.generate_empty_action_histories()
    # print eah

    # =========================================================================================
    # min_brain = generate_minimal_brain()
    # print min_brain._neurons
    # =========================================================================================

    # =========================================================================================
    # generate_standard_fish()
    # =========================================================================================

    print("\nfor debug ...")
