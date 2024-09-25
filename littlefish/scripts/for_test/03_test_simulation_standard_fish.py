import random

import littlefish.core.fish as fi
import littlefish.core.simulation as si
import littlefish.core.terrain as tr
import numpy as np

simulation_length = 5000  # 100000

random.seed(111)
np.random.seed(50)

terrain_map = np.array(
    [
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
    ],
    dtype=np.uint8,
)

terrain = tr.BinaryTerrain(terrain_map)
fish = fi.generate_standard_fish()
simulation = si.Simulation(
    terrain=terrain, fish_list=[fish], simulation_length=simulation_length, food_num=2
)
simulation.initiate_simulation()
msg = simulation.run(verbose=1)
simulation.save_log(
    r"C:\little_fish_simulation_logs", msg=msg, is_save_psp_waveforms=False
)

print("for debug ...")
