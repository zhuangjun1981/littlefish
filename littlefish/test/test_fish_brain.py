# from __future__ import (absolute_import, division,
#                         print_function, unicode_literals)
# from builtins import *

import os
import sys
package_path, _ = os.path.split(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(package_path)
import fish.brain as brain
import utilities as util
import numpy as np


def test_connection_psp():
    connection = brain.Connection(amplitude=10, latency=5, rise_time=5, decay_time=10)
    assert(np.array_equal(connection.get_psp(), np.array([0.,0.,0.,0.,0.,2.,4.,6.,8.,10.,9.,8.,7.,6.,5.,4.,3.,2.,1.,0.])))


def test_connection_act():
    SIMULATION_LENGTH = 50
    connection = brain.Connection(amplitude=10, latency=5, rise_time=5, decay_time=10)
    postsynaptic_input = np.zeros(SIMULATION_LENGTH)
    connection.act(2, postsynaptic_input)
    assert(np.array_equal(postsynaptic_input,
                          np.array([0.,0.,0.,0.,0.,0.,0.,2.,4.,6.,8.,10.,9.,8.,7.,6.,5.,4.,3.,2.,1.,0.,0.,0.,0.,0.,0.,
                                    0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])))
    connection.act(4, postsynaptic_input)
    assert (np.array_equal(postsynaptic_input,
                           np.array([0.,0.,0.,0.,0.,0.,0.,2.,4.,8.,12.,16.,17.,18.,16.,14.,12.,10.,8.,6.,4.,2.,1.,0.,0.,
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])))
    connection.act(40, postsynaptic_input)
    assert (np.array_equal(postsynaptic_input,
                           np.array([0.,0.,0.,0.,0.,0.,0.,2.,4.,8.,12.,16.,17.,18.,16.,14.,12.,10.,8.,6.,4.,2.,1.,0.,0.,
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,2.,4.,6.,8.,10.])))
    connection.act(50, postsynaptic_input)
    assert (np.array_equal(postsynaptic_input,
                           np.array([0.,0.,0.,0.,0.,0.,0.,2.,4.,8.,12.,16.,17.,18.,16.,14.,12.,10.,8.,6.,4.,2.,1.,0.,0.,
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,2.,4.,6.,8.,10.])))


def test_eye_get_input_pixels():

    world_map = np.zeros((5, 5), dtype=np.uint8)
    world_map[3, 3] = 1
    world_map[2, 2] = 1
    world_map[2, 1] = 1

    eye = brain.Eye(direction='south')
    assert(np.array_equal(eye._get_input_pixels(position=(2, 3), input_map=world_map), [0, 1, 0]))
    eye = brain.Eye(direction='southeast')
    assert (np.array_equal(eye._get_input_pixels(position=(2, 3), input_map=world_map), [1, 0, 0]))
    eye = brain.Eye(direction='east')
    assert (np.array_equal(eye._get_input_pixels(position=(3, 2), input_map=world_map), [0, 1, 0]))
    eye = brain.Eye(direction='northeast')
    assert (np.array_equal(eye._get_input_pixels(position=(3, 2), input_map=world_map), [1, 0, 1]))
    eye = brain.Eye(direction='north')
    assert (np.array_equal(eye._get_input_pixels(position=(3, 2), input_map=world_map), [0, 1, 1]))
    eye = brain.Eye(direction='northwest')
    assert (np.array_equal(eye._get_input_pixels(position=(3, 2), input_map=world_map), [1, 1, 0]))
    eye = brain.Eye(direction='west')
    assert (np.array_equal(eye._get_input_pixels(position=(3, 2), input_map=world_map), [1, 0, 0]))
    eye = brain.Eye(direction='southwest')
    assert (np.array_equal(eye._get_input_pixels(position=(3, 2), input_map=world_map), [0, 0, 0]))
    eye = brain.Eye(direction='north')
    assert (np.array_equal(eye._get_input_pixels(position=(0, 0), input_map=world_map), [1, 1, 1]))
    eye = brain.Eye(direction='northeast')
    assert (np.array_equal(eye._get_input_pixels(position=(0, 0), input_map=world_map), [0, 1, 1]))
    eye = brain.Eye(direction='east')
    assert (np.array_equal(eye._get_input_pixels(position=(0, 0), input_map=world_map), [0, 0, 1]))


def test_eye2_get_input_pixels():

    world_map = np.zeros((5, 5), dtype=np.uint8)
    world_map[3, 3] = 1
    world_map[2, 2] = 1
    world_map[2, 1] = 1
    world_map[1, 1] = 1

    eye2 = brain.Eye2(direction='south')
    assert(np.array_equal(eye2._get_input_pixels(position=(2, 3), input_map=world_map), [0, 1, 0, 0, 0, 0]))
    eye2 = brain.Eye2(direction='southeast')
    assert (np.array_equal(eye2._get_input_pixels(position=(2, 3), input_map=world_map), [1, 0, 0, 0, 1, 1]))
    eye2 = brain.Eye2(direction='east')
    assert (np.array_equal(eye2._get_input_pixels(position=(3, 2), input_map=world_map), [0, 1, 0, 0, 0, 0]))
    eye2 = brain.Eye2(direction='northeast')
    assert (np.array_equal(eye2._get_input_pixels(position=(3, 2), input_map=world_map), [1, 0, 1, 0, 0, 0]))
    eye2 = brain.Eye2(direction='north')
    assert (np.array_equal(eye2._get_input_pixels(position=(3, 2), input_map=world_map), [0, 1, 1, 0, 0, 1]))
    eye2 = brain.Eye2(direction='northwest')
    assert (np.array_equal(eye2._get_input_pixels(position=(3, 2), input_map=world_map), [1, 1, 0, 1, 0, 0]))
    eye2 = brain.Eye2(direction='west')
    assert (np.array_equal(eye2._get_input_pixels(position=(3, 2), input_map=world_map), [1, 0, 0, 0, 0, 0]))
    eye2 = brain.Eye2(direction='southwest')
    assert (np.array_equal(eye2._get_input_pixels(position=(3, 2), input_map=world_map), [0, 0, 0, 0, 1, 1]))
    eye2 = brain.Eye2(direction='north')
    assert (np.array_equal(eye2._get_input_pixels(position=(0, 0), input_map=world_map), [1, 1, 1, 1, 1, 1]))
    eye2 = brain.Eye2(direction='northeast')
    assert (np.array_equal(eye2._get_input_pixels(position=(0, 0), input_map=world_map), [0, 1, 1, 1, 1, 1]))
    eye2 = brain.Eye2(direction='east')
    assert (np.array_equal(eye2._get_input_pixels(position=(0, 0), input_map=world_map), [1, 0, 1, 0, 0, 1]))


def test_neuron_connection():
    SIMULATION_LENGTH = 5000
    neuron_pre = brain.Neuron(baseline_rate=0.005)
    neuron_post = brain.Neuron(baseline_rate=0.000)
    connection = brain.Connection(amplitude=1, latency=5, rise_time=1, decay_time=1)

    postsynaptic_input = np.zeros(SIMULATION_LENGTH)
    action_history_pre = []
    action_history_post = []

    for i in range(SIMULATION_LENGTH):

        is_firing = neuron_pre.act(i, action_history=action_history_pre)
        if is_firing:
            connection.act(i, postsynaptic_input)
        neuron_post.act(i, action_history=action_history_post, probability_input=postsynaptic_input[i])

    ccg, t = util.discreat_crosscorrelation(np.array(action_history_pre), np.array(action_history_post))
    assert(np.argmax(ccg) == 15)


def test_muscle_action():
    SIMULATION_LENGTH = 20000
    muscle = brain.Muscle(direction='east', baseline_rate=0., refractory_period=5000)
    action_history_muscle = []
    movements = []
    for i in range(SIMULATION_LENGTH):
        movement = muscle.act(i, action_history=action_history_muscle, probability_input=0.5, probability_base=0.1)
        if movement is not False:
            movements.append(movement)
    target_movements = [np.array([0, 1], dtype=np.uint8), np.array([0, 1], dtype=np.uint8),
                         np.array([0, 1], dtype=np.uint8), np.array([0, 1], dtype=np.uint8)]
    assert(all([np.array_equal(movements[i], target_movements[i]) for i in range(4)]))
    assert(action_history_muscle == [0, 5000, 10000, 15000])


def test_brain_default():
    brain1 = brain.Brain()
    assert(len(brain1.get_neurons()) == 20)
    assert(len(brain1.get_connections()) == 2)
    assert(brain1.get_connections()['L000_L001'].shape[0] == 8)
    assert(brain1.get_connections()['L000_L001'].shape[1] == 8)
    assert(brain1.get_connections()['L001_L002'].shape[0] == 4)
    assert(brain1.get_connections()['L001_L002'].shape[1] == 8)


def run():
    test_connection_psp()
    test_connection_act()
    test_eye_get_input_pixels()
    test_eye2_get_input_pixels()
    test_neuron_connection()
    test_muscle_action()
    test_brain_default()


if __name__ == '__main__':

    run()
