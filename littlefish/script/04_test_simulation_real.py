import time
import random
import numpy as np
import matplotlib.pyplot as plt
import littlefish.core.fish as fi
import littlefish.core.simulation as si
import littlefish.core.terrain as tr

simulation_length = 100000  # 100000

random.seed(111)
np.random.seed(50)

tg = tr.TerrainGenerator(size=[128, 128], sea_level=0.6)
terrain_map = tg.generate_binary_map(sigma=3., is_plot=True)
plt.show()
terrain = tr.BinaryTerrain(terrain_map)
fish = fi.generate_standard_fish()
simulation = si.Simulation(terrain=terrain, fish_list=[fish],
                           simulation_length=simulation_length, food_num=20)

simulation.initiate_simulation()
simulation.run(verbose=1)
simulation.save_log(r'C:\little_fish_simulation_logs')

print 'for debug ...'