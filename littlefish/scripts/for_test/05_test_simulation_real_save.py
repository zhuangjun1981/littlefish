import sys
import os
import datetime
import inspect
import h5py
import random
import numpy as np
import matplotlib.pyplot as plt
import littlefish.core.fish as fi
import littlefish.core.simulation as si
import littlefish.core.terrain as tr

log_folder = r'C:\little_fish_simulation_logs'
simulation_length = 2000  # 100000
random_seed = 111
np_random_seed = 50

if not os.path.isdir(log_folder):
    os.mkdir(log_folder)
os.chdir(log_folder)

random.seed(random_seed)
np.random.seed(np_random_seed)

fish = fi.generate_standard_fish()
save_name = 'fish_' + datetime.datetime.now().strftime('%y%m%d_%H_%M_%S') + '.hdf5'
save_f = h5py.File(save_name, "a")
fish_grp = save_f.create_group('fish_' + fish.name)
fish.to_h5_group(fish_grp)

tg = tr.TerrainGenerator(size=[128, 128], sea_level=0.6)
terrain_map = tg.generate_binary_map(sigma=3., is_plot=True)
plt.show()
terrain = tr.BinaryTerrain(terrain_map)
simulation = si.Simulation(terrain=terrain, fish_list=[fish],
                           simulation_length=simulation_length, food_num=20)

simulation.initiate_simulation()
msg = simulation.run(verbose=1)

sim_grp = save_f.create_group('simulation_' + datetime.datetime.now().strftime('%y%m%d_%H_%M_%S'))
sim_grp['random_seed'] = random_seed
sim_grp['np_random_seed'] = np_random_seed
sim_grp['simulation_length'] = simulation_length
sim_grp['script_txt'] = inspect.getsource(sys.modules[__name__])
simulation.save_log_to_h5_grp(sim_grp, is_save_psp_waveforms=False)

print('for debug ...')