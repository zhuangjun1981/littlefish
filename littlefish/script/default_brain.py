from littlefish.fish import brain
from littlefish import utilities as util
import pandas as pd
import numpy as np
import time

SIMULATION_LENGTH = int(1e5)
CONNECTION_AMPLITUDE = 0.01

brain = brain.Brain()

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

