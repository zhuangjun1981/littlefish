import os
import sys
import h5py
import random
import inspect
import numpy as np
import datetime
import littlefish.core.utilities as util
import littlefish.core.fish as fi
import littlefish.core.terrain as tr
import littlefish.core.simulation as si

data_folder = r"C:\little_fish_simulation_logs"

generation = 48

# five times of a standard fish's life span without hitting land and eating, 5 * max_health / health_decay_rate
# max_health of a standard fish: 100
# health_decay_rate of a standard fish: 0.01
simulation_length = 50000
sim_num = 3
terrain_size = [128, 128]
sea_level = 0.5
food_num = 200
hard_thr = 5000

gen_folder = os.path.join(data_folder, "generation_" + util.int2str(generation, 6))
os.chdir(gen_folder)

fish_lst = [
    f for f in os.listdir(gen_folder) if f[0:5] == "fish_" and f[-5:] == ".hdf5"
]
fish_lst.sort()
print("\n".join(fish_lst) + "\n")

total_fish_num = len(fish_lst)

tg = tr.TerrainGenerator(size=terrain_size, sea_level=sea_level)

for fish_ind, fish_path in enumerate(fish_lst):
    si.simulate_one_fish(
        fish_path=fish_path,
        simulation_length=simulation_length,
        simulation_num=sim_num,
        terrain_size=terrain_size,
        sea_level=sea_level,
        food_num=food_num,
        hard_thr=hard_thr,
        fish_ind=fish_ind,
        fish_num=total_fish_num,
    )
