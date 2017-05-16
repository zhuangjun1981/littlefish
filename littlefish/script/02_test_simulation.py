import numpy as np
import random
import littlefish.terrain.terrain_2d as tr
import littlefish.fish.fish as fi
import littlefish.simulation.simulation as si

simulation_length = 10  # 100000

random.seed(111)
np.random.seed(50)

terrain_map = np.array([[0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1],
                        [0, 0, 1, 0, 0],
                        [0, 1, 0, 1, 0]], dtype=np.uint8)

terrain = tr.BinaryTerrain(terrain_map)
fish = fi.Fish()
simulation = si.Simulation(terrain=terrain, fish_list=[fish],
                           simulation_length=simulation_length, food_num=1)
simulation.initiate_simulation()
simulation.run(verbose=1)
simulation.save_log(r'D:\little_fish')

print 'for debug ...'