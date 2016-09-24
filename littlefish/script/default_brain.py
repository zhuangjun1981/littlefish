from littlefish.fish import brain
from littlefish import utilities as util
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

SIMULATION_LENGTH = int(5e4)

brain = brain.Brain()

print(brain.get_neurons())
neuron = brain.get_neurons().loc[10, 'neuron']
conn = brain.get_single_connection(0, 10)
print(conn.get_amplitude())

# terrain_map = np.zeros((10, 10), dtype=np.uint8)
# terrain_map[2:4, 4:6] = 1
#
# curr_percentage = -1
#
# t1 = time.time()
# for i in range(SIMULATION_LENGTH):
#     if i // (SIMULATION_LENGTH / 10) > curr_percentage:
#         curr_percentage += 1
#         print('simulation progress: ' + util.int2str(curr_percentage*10, 2) + '%')
#
#     movement = brain.act(i, (3,2), terrain_map)
#     if not np.array_equal(movement, [0, 0]):
#         print(i, movement)
#
# print('simulation time: ' + str(time.time() - t1) + ' seconds.')
#
# f = plt.figure(figsize=(20, 5))
# ax = f.add_subplot(111)
# brain.plot_action_histories_scatter(plot_axis=ax, plot_length=SIMULATION_LENGTH, ms=10, mec='none')
# plt.show()

print('for debug ...')

