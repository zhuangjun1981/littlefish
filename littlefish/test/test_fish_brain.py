

import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import random
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
    psp_waveforms = np.zeros((1, SIMULATION_LENGTH))
    connection.act(t_point=2, postsynaptic_index=0, psp_waveforms=psp_waveforms)
    assert(np.array_equal(psp_waveforms[0],
                          np.array([0.,0.,0.,0.,0.,0.,0.,2.,4.,6.,8.,10.,9.,8.,7.,6.,5.,4.,3.,2.,1.,0.,0.,0.,0.,0.,0.,
                                    0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])))
    connection.act(t_point=4, postsynaptic_index=0, psp_waveforms=psp_waveforms)
    assert (np.array_equal(psp_waveforms[0],
                           np.array([0.,0.,0.,0.,0.,0.,0.,2.,4.,8.,12.,16.,17.,18.,16.,14.,12.,10.,8.,6.,4.,2.,1.,0.,0.,
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])))
    connection.act(t_point=40, postsynaptic_index=0, psp_waveforms=psp_waveforms)
    assert (np.array_equal(psp_waveforms[0],
                           np.array([0.,0.,0.,0.,0.,0.,0.,2.,4.,8.,12.,16.,17.,18.,16.,14.,12.,10.,8.,6.,4.,2.,1.,0.,0.,
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,2.,4.,6.,8.,10.])))
    connection.act(t_point=50, postsynaptic_index=0, psp_waveforms=psp_waveforms)
    assert (np.array_equal(psp_waveforms[0],
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

    psp_waveforms = np.zeros((1, SIMULATION_LENGTH))
    action_history_pre = []
    action_history_post = []

    for i in range(SIMULATION_LENGTH):

        is_firing = neuron_pre.act(i, action_history=action_history_pre)
        if is_firing:
            connection.act(t_point=i, postsynaptic_index=0, psp_waveforms=psp_waveforms)
        neuron_post.act(i, action_history=action_history_post, probability_input=psp_waveforms[0, i])

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


def test_brain_minimum_brain():

    simulation_length = int(1e2)
    random.seed(111)

    eye = brain.Eye2(direction='east', input_filter=np.array([0.15, 0.3, 0.15, 0.1, 0.2, 0.1]), gain=0.05,
                     input_type='terrain', baseline_rate=0., refractory_period=10)
    hidden0 = brain.Neuron(baseline_rate=0.0005, refractory_period=10)
    hidden1 = brain.Neuron(baseline_rate=0.0005, refractory_period=10)
    muscle = brain.Muscle(direction='east', baseline_rate=0.1, refractory_period=5000)

    neurons = pd.DataFrame([[0, 0, eye],
                            [1, 0, hidden0],
                            [1, 1, hidden1],
                            [2, 0, muscle]], columns=['layer', 'neuron_ind', 'neuron'])

    connection_eye_hidden0 = brain.Connection(latency=30, amplitude=0.01, rise_time=50, decay_time=100)
    connection_eye_hidden1 = brain.Connection(latency=30, amplitude=0.0001, rise_time=50, decay_time=100)
    connection_hidden0_muscle = brain.Connection(latency=30, amplitude=0.0001, rise_time=50, decay_time=100)
    connection_hidden1_muscle = brain.Connection(latency=30, amplitude=0.01, rise_time=50, decay_time=100)

    conn_0_1 = pd.DataFrame([[connection_eye_hidden0], [connection_eye_hidden1]], columns=[0], index=[1, 2])
    conn_1_2 = pd.DataFrame([[connection_hidden0_muscle, connection_hidden1_muscle]], columns=[1, 2], index=[3])

    connections = {'L000_L001': conn_0_1,
                   'L001_L002': conn_1_2}

    minimum_brain = brain.Brain(neurons=neurons, connections=connections)
    terrain_map = np.zeros((10, 10), dtype=np.uint8)
    terrain_map[2:4, 4:6] = 1

    action_histories = pd.Series([[] for i in range(len(neurons))])
    action_histories = pd.DataFrame(action_histories, columns=['action_history'])
    psp_waveforms = np.zeros((len(neurons), simulation_length), dtype=np.float32)
    body_position = np.array([3, 2], dtype=np.uint32)

    curr_percentage = -1

    t1 = time.time()
    for i in range(simulation_length):

        # # print simulation progress
        # if i // (simulation_length / 10) > curr_percentage:
        #     curr_percentage += 1
        #     print('simulation progress: ' + util.int2str(curr_percentage * 10, 2) + '%')

        movement = minimum_brain.act(t_point=i, body_position=(3, 2), action_histories=action_histories,
                                     psp_waveforms=psp_waveforms, terrain_map=terrain_map)
        if not np.array_equal(movement, [0, 0]):
            # print(i, movement)
            body_position = body_position + movement

    # print('simulation time: ' + str(time.time() - t1) + ' seconds.')
    #
    # print(action_histories)
    #
    # plt.imshow(psp_waveforms, interpolation='none')
    # plt.axes().set_aspect(100)
    # plt.colorbar()
    # plt.show()
    assert(action_histories.iloc[0, 0] == [53, 93])
    assert(action_histories.iloc[3, 0] == [6])


def run():
    test_connection_psp()
    test_connection_act()
    test_eye_get_input_pixels()
    test_eye2_get_input_pixels()
    test_neuron_connection()
    test_muscle_action()
    test_brain_default()
    test_brain_minimum_brain()


if __name__ == '__main__':

    run()
