import time

import h5py
import littlefish.core.fish
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from littlefish.core import utilities as util

SIMULATION_LENGTH = int(1e4)

eye = littlefish.core.fish.Eye2(direction='east', input_filter=np.array([0.15, 0.3, 0.15, 0.1, 0.2, 0.1]), gain=0.005,
                                input_type='terrain', baseline_rate=0., refractory_period=10)
hidden0 = littlefish.core.fish.Neuron(baseline_rate=0.0005, refractory_period=10)
hidden1 = littlefish.core.fish.Neuron(baseline_rate=0.0005, refractory_period=10)
muscle = littlefish.core.fish.Muscle(direction='east', baseline_rate=0.1, refractory_period=5000)

neurons = pd.DataFrame([[0, 0, eye],
                        [1, 0, hidden0],
                        [1, 1, hidden1],
                        [2, 0, muscle]], columns=['layer', 'neuron_ind', 'neuron'])

connection_eye_hidden0 = littlefish.core.fish.Connection(latency=30, amplitude=0.01, rise_time=50, decay_time=100)
connection_eye_hidden1 = littlefish.core.fish.Connection(latency=30, amplitude=0.0001, rise_time=50, decay_time=100)
connection_hidden0_muscle = littlefish.core.fish.Connection(latency=30, amplitude=0.0001, rise_time=50, decay_time=100)
connection_hidden1_muscle = littlefish.core.fish.Connection(latency=30, amplitude=0.01, rise_time=50, decay_time=100)

conn_0_1 = pd.DataFrame([[connection_eye_hidden0], [connection_eye_hidden1]], columns=[0], index=[1, 2])
conn_1_2 = pd.DataFrame([[connection_hidden0_muscle, connection_hidden1_muscle]], columns=[1, 2], index=[3])

connections = {'L000_L001': conn_0_1,
               'L001_L002': conn_1_2}

brain = littlefish.core.fish.Brain(neurons=neurons, connections=connections)

terrain_map = np.zeros((10, 10), dtype=np.uint8)
terrain_map[2:4, 4:6] = 1

curr_percentage = -1

t1 = time.time()
for i in range(SIMULATION_LENGTH):
    if i // (SIMULATION_LENGTH / 10) > curr_percentage:
        curr_percentage += 1
        print('simulation progress: ' + util.int2str(curr_percentage*10, 2) + '%')

    movement = brain.act(i, (3,2), terrain_map)
    if not np.array_equal(movement, [0, 0]):
        print(i, movement)

print('simulation time: ' + str(time.time() - t1) + ' seconds.')

brain.plot_action_histories_scatter(plot_length=SIMULATION_LENGTH, ms=10, mec='none')
plt.show()

dfile_path = r"D:\littlefish\test_folder\minimum_brain.hdf5"
dfile = h5py.File(dfile_path, "a")
group = dfile.create_group('brain')
brain.to_h5_group(group)

print('for debug ...')

