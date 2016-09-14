# from __future__ import (absolute_import, division,
#                         print_function, unicode_literals)
# from builtins import *

import random
import numpy as np


# consider one time unit is 0.1 milisecond, time unit should be small enough that no more than one action is possible
# per time unit
SIMULATION_LENGTH = int(1e5)  # number of time units in simulation


class Neuron(object):
    """
    a very simple neuron class
    """

    def __init__(self, baseline_rate=0.0001, refractory_period=10):
        """
        action is the equivalent of action potential in biology, and consider one time unit is 0.1 milisecond

        :param baseline_rate: float, probablity of a action per time unit.
        :param refractory_period: float, refractory_period in time unit
        """

        self._baseline_rate = float(baseline_rate)
        self._refractory_period = float(refractory_period)
        self._action_history = []

    def get_baseline_rate(self):
        return self._baseline_rate

    def get_refractory_period(self):
        return self._refractory_period

    def get_action_history(self):
        """
        :return: action_history, list of ints, timing of all actions as indices in time axis of time unit
        """
        return self._action_history

    def act(self, t_point, probability_input=0):
        """
        evaluate if the neuron will fire at given time point
        :param t_point: int, current time point as the index of time unit axis
        :param probability_input: float, summed connection inputs, as add on to baseline_rate
        :return: bool, True: fire; False: quite
        """

        if len(self._action_history) > 0 and t_point - self._action_history[-1] <= self._refractory_period:
            return False
        else:
            curr_rate = self._baseline_rate + probability_input
            if random.random() <= curr_rate:
                self._action_history.append(t_point)
                # print(t_point)
                return True
            else:
                return False


class Eye(Neuron):
    """
    Eye class to observe the environment, subclass of Neuron
    """

    def __init__(self, position, direction, input_filter=None, gain=None, baseline_rate=0., refractory_period=10):
        """
        for a fish occupies 3x3 space, consider the eyes are in the outer rim of the body (the 8 pixels surrounding the
        central pixel. Each pixel is an eye, receiving the input from the closest 3 pixels in the environment.
        for example:

        fish (1) in the environment(0):
        0 0 0 0 0 0 0
        0 0 0 0 0 0 0
        0 0 1 1 1 0 0
        0 0 1 1 1 0 0
        0 0 1 1 1 0 0
        0 0 0 0 0 0 0
        0 0 0 0 0 0 0

        eye (2) in the northwest connor is:
        0 0 0 0 0 0 0
        0 0 0 0 0 0 0
        0 0 2 1 1 0 0
        0 0 1 1 1 0 0
        0 0 1 1 1 0 0
        0 0 0 0 0 0 0
        0 0 0 0 0 0 0

        it receive inputs from pixels labelled as 3 in the environment:
        0 0 0 0 0 0 0
        0 3 3 0 0 0 0
        0 3 2 1 1 0 0
        0 0 1 1 1 0 0
        0 0 1 1 1 0 0
        0 0 0 0 0 0 0
        0 0 0 0 0 0 0

        the inputs from the environment (1x3 array) will be filtered by a array with same size, to generate
        a single value as the base of its input. this value will be multiplied by a float number gain to generate
        final input probability

        :param position: the coordinate of the eye, tuple of two ints, (row, col)
        :param direction: the aim of the eye, should be one of the following, 'north', 'south', 'east', 'west',
                          'northwest', 'northeast', 'southwest', 'southeast'
        :param baseline_rate: float, probablity of a action per time unit.
        :param refractory_period: float, refractory_period in time unit
        """

        super(Eye, self).__init__(baseline_rate=baseline_rate, refractory_period=refractory_period)

        if len(position) != 2:
            raise(ValueError, 'position should have 2 elements.')

        if isinstance(position[0], int) and isinstance(position[1], int):
            self._position = position
        else:
            raise(ValueError, 'Elements in position should both be integers.')

        if direction in ['north', 'south', 'east', 'west', 'northwest', 'northeast', 'southwest', 'southeast']:
            self._direction = direction
        else:
            raise(ValueError, "direction should be one of the following: ['north', 'south', 'east', 'west', "
                              "'northwest', 'northeast', 'southwest', 'southeast'].")

        if input_filter is None:
            self._input_filter = np.array([0.2, 0.6, 0.2])
        else:
            self._input_filter = input_filter.astype(np.float)

        if gain is None:
            self._gain = 0.001
        else:
            self._gain = float(gain)

    def move(self, movement):
        """
        update eye position

        :param movement: movement vector, tuple of two ints, (row, col)
        """
        if len(movement) != 2:
            raise(ValueError, 'Movement vector should have 2 elements.')

        if isinstance(movement[0], int) and isinstance(movement[1], int):
            self._position = (self._position[0] + movement[0],
                              self._position[1] + movement[1])
        else:
            raise(ValueError, 'Elements in movement should both be integers.')

    def _get_input_pixels(self, world_map, border_value=1.):
        """
        :return: the 1d array with the values of the 3 pixels the eye is suppose to look at. pixels out of the world_map
        range will be returned as border_value
        """

        if len(world_map.shape) != 2:
            raise(ValueError, 'world_map should a 2-d array.')

        if self._direction == 'east':
            ind = [[self._position[0] + 1, self._position[1] + 1],
                   [self._position[0],     self._position[1] + 1],
                   [self._position[0] - 1, self._position[1] + 1]]
        elif self._direction == 'northeast':
            ind = [[self._position[0],     self._position[1] + 1],
                   [self._position[0] - 1, self._position[1] + 1],
                   [self._position[0] - 1, self._position[1]]]
        elif self._direction == 'north':
            ind = [[self._position[0] - 1, self._position[1] + 1],
                   [self._position[0] - 1, self._position[1]],
                   [self._position[0] - 1, self._position[1] - 1]]
        elif self._direction == 'northwest':
            ind = [[self._position[0] - 1, self._position[1]],
                   [self._position[0] - 1, self._position[1] - 1],
                   [self._position[0],     self._position[1] - 1]]
        elif self._direction == 'west':
            ind = [[self._position[0] - 1, self._position[1] - 1],
                   [self._position[0],     self._position[1] - 1],
                   [self._position[0] + 1, self._position[1] - 1]]
        elif self._direction == 'southwest':
            ind = [[self._position[0],     self._position[1] - 1],
                   [self._position[0] + 1, self._position[1] - 1],
                   [self._position[0] + 1, self._position[1]]]
        elif self._direction == 'south':
            ind = [[self._position[0] + 1, self._position[1] - 1],
                   [self._position[0] + 1, self._position[1]],
                   [self._position[0] + 1, self._position[1] + 1]]
        elif self._direction == 'southeast':
            ind = [[self._position[0] + 1, self._position[1]],
                   [self._position[0] + 1, self._position[1] + 1],
                   [self._position[0],     self._position[1] + 1]]
        else:
            raise(ValueError, "direction should be one of the following: ['north', 'south', 'east', 'west', "
                              "'northwest', 'northeast', 'southwest', 'southeast'].")

        # print(ind)

        input_pixels = []

        for cor in ind:
            if cor[0] < 0 or cor[0] > world_map.shape[0] or cor[1] < 0 or cor[1] > world_map.shape[1]:
                input_pixels.append(border_value)
            else:
                input_pixels.append(world_map[cor[0], cor[1]])

        return np.array(input_pixels)

    def _get_input(self, **kwargs):
        """
        :return: calculate real time input from the visual field
        """
        input_pixels = self._get_input_pixels(**kwargs)
        probablity_input = self._gain * np.sum(input_pixels * self._input_filter)

        return probablity_input

    def act(self, t_point, **kwargs):
        """
        evaluate if the eye neuron will fire at given time point
        :param t_point: int, current time point as the index of time unit axis
        :param kwargs, border_value trace back to self._get_input_pixels
        :return: bool, True: fire; False: quite
        """

        probability_input = self._get_input(**kwargs)

        if len(self._action_history) > 0 and t_point - self._action_history[-1] <= self._refractory_period:
            return False
        else:
            curr_rate = self._baseline_rate + probability_input
            if random.random() <= curr_rate:
                self._action_history.append(t_point)
                return True
            else:
                return False


class Connection(object):
    """
    synaptic connection between two neurons
    """

    def __init__(self, latency=30, amplitude=0.0001, rise_time=5, decay_time=10):
        """

        :param latency: int, temporal latency from presynaptic neuron action to the postsynaptic effect onset, number
                        of time units
        :param amplitude: float, peak change of the firing rate in the postsynaptic neuron, probablity of a action per
                          time unit. can be positive (excitatiory) or negative (inhibitory)
        :param rise_time: int, temporal duration from onset to peak, number of time units
        :param decay_time: int, temporal duration from peak to baseline, number of time units
        """

        self._latency = latency
        self._amplitude = amplitude
        self._rise_time = rise_time
        self._decay_time = decay_time

        self._generate_psp()

    def _generate_psp(self):
        """
        generate post synaptic probability wave form
        """

        self._psp = np.zeros(self._latency + self._rise_time + self._decay_time)
        self._psp[self._latency: self._latency + self._rise_time] = self._amplitude * \
                                                                   (np.arange(self._rise_time, dtype=np.float32) +
                                                                    1) / self._rise_time
        self._psp[-self._decay_time:] = self._amplitude * (np.arange(self._decay_time, 0, -1, dtype=np.float32)
                                                          - 1) / self._decay_time

    def get_psp(self):
        return self._psp

    def act(self, time_point, postsynaptic_input):
        """
        if the presynaptic neuron fires at the 'time_point', a psp wave form will be generated and add to the
        input array of the postsynaptic neuron
        :param time_point: int, current time point as the index of time unit axis
        :param postsynaptic_input: 1-d array of floats
        :return:
        """
        psp_end = time_point + len(self._psp)
        if psp_end <= len(postsynaptic_input):
            postsynaptic_input[time_point: psp_end] += self._psp
        else:
            postsynaptic_input[time_point:] += self._psp[:len(postsynaptic_input)-time_point]


if __name__ == '__main__':

    # =========================================================================================
    # neuron = Neuron()
    # for i in range(SIMULATION_LENGTH):
    #     neuron.act(i)
    # print(len(neuron.get_action_history()))
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
    # SIMULATION_LENGTH = 500000
    # neuron_pre = Neuron(baseline_rate=0.005)
    # neuron_post = Neuron(baseline_rate=0.002)
    # connection = Connection(amplitude=0.01, latency=5)
    #
    # postsynaptic_input = np.zeros(SIMULATION_LENGTH)
    #
    # for i in range(SIMULATION_LENGTH):
    #
    #     is_firing = neuron_pre.act(i)
    #     if is_firing:
    #         connection.act(i, postsynaptic_input)
    #     neuron_post.act(i, postsynaptic_input[i])
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
    SIMULATION_LENGTH = 100000

    world_map = np.zeros((5, 5), dtype=np.uint8)
    world_map[3, 3] = 1
    print(world_map)

    eye = Eye(position=(2, 3), direction='south')

    for i in range(SIMULATION_LENGTH):
        eye.act(i, world_map=world_map)
    print(len(eye.get_action_history()))
    # =========================================================================================

    print('debug...')
