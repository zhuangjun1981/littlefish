from littlefish.fish import brain
from littlefish import utilities as util
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

SIMULATION_LENGTH = int(1e4)
CONNECTION_AMPLITUDE = 0.01

neurons_df = pd.DataFrame([[0, 0, 0.0, 10],
                           [1, 0, 0.0005, 10],
                           [2, 0, 0.0, 5000]], columns=['layer', 'neuron_ind', 'baseline_rate', 'refractory_period'])

connections_df = pd.DataFrame([[0, 1], [1, 2]], columns=['presynaptic_ind', 'postsynaptic_ind'])

brain = brain.Brain(neurons_df=neurons_df, connections_df=connections_df)

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

# print(brain.get_neurons().loc[0, 'neuron'].get_action_history())
# print(len(brain.get_neurons().loc[0, 'neuron'].get_action_history()) / 10.)
# print(brain.get_neurons().loc[1, 'neuron'].get_action_history())
# print(len(brain.get_neurons().loc[1, 'neuron'].get_action_history()) / 10.)
# print(brain.get_neurons().loc[2, 'neuron'].get_action_history())
# print(len(brain.get_neurons().loc[2, 'neuron'].get_action_history()) / 10.)

brain.plot_action_histories_scatter(plot_length=SIMULATION_LENGTH, ms=10, mec='none')
plt.show()

