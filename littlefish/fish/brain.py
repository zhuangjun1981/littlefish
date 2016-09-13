from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *
import random
import numpy as np
import matplotlib.pyplot as plt
import littlefish.utilities as util

# consider one time unit is 0.1 milisecond, time unit should be small enough that no more than one action is possible
# per time unit
SIMULATION_LENGTH = int(1e5) # number of time units in simulation

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

        self._baseline_rate = baseline_rate
        self._refractory_period = refractory_period
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

    def act(self, t_point, input=0):
        """
        evaluate if the neuron will fire at given time point
        :param t_point: int, current time point as the index of time unit axis
        :param input: float, summed connection inputs, as add on to baseline_rate
        :return: bool, True: fire; False: quite
        """

        if len(self._action_history) > 0 and t_point - self._action_history[-1] <= self._refractory_period:
            return False
        else:
            curr_rate = self._baseline_rate + input
            if random.random() <= curr_rate:
                self._action_history.append(t_point)
                # print(t_point)
                return True
            else:
                return False

class Eye(Neuron):
    """
    Eye class is a subclass of neuron. With 0 baseline rate.
    """
    # todo: add eye class
    pass

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

    #===========================================
    # neuron = Neuron()
    # for i in range(SIMULATION_LENGTH):
    #     neuron.act(i)
    # print(len(neuron.get_action_history()))
    # ===========================================

    # ===========================================
    # connection = Connection(amplitude=10, latency=5)
    # print(connection.get_psp())
    # ===========================================

    # ===========================================
    # SIMULATION_LENGTH = 50
    # postsynaptic_input = np.zeros(SIMULATION_LENGTH)
    # connection = Connection(amplitude=10, latency=5, rise_time=5, decay_time=10)
    # connection.act(2, postsynaptic_input)
    # print(postsynaptic_input)
    # connection.act(4, postsynaptic_input)
    # print(postsynaptic_input)
    # connection.act(40, postsynaptic_input)
    # print(postsynaptic_input)
    # ===========================================

    # ===========================================
    SIMULATION_LENGTH = 500000
    neuron_pre = Neuron(baseline_rate=0.005)
    neuron_post = Neuron(baseline_rate=0.002)
    connection = Connection(amplitude=0.01, latency=5)

    postsynaptic_input = np.zeros(SIMULATION_LENGTH)

    for i in range(SIMULATION_LENGTH):

        is_firing = neuron_pre.act(i)
        if is_firing:
            connection.act(i, postsynaptic_input)
        neuron_post.act(i, postsynaptic_input[i])

    spk_train_pre = neuron_pre.get_action_history()
    spk_train_post = neuron_post.get_action_history()

    # print(postsynaptic_input)
    print(len(spk_train_pre))
    print(len(spk_train_post))

    ccg, t = util.discreat_crosscorrelation(np.array(spk_train_pre), np.array(spk_train_post))
    plt.bar(t, ccg)
    plt.show()
    # ===========================================


    print('debug...')