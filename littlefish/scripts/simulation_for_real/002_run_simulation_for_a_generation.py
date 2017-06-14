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

data_folder = r'C:\little_fish_simulation_logs'
# a standard fish's life span without hitting land and eating, max_health / health_decay_rate
simulation_length = 100000
generation = 1
random_seeds = [30, 57, 68]
terrain_size = [128, 128]
sea_level = 0.6
food_num = 20

gen_folder = os.path.join(data_folder, 'generation_' + util.int2str(generation, 6))
os.chdir(gen_folder)

fish_lst = [f for f in os.listdir(gen_folder) if f[0:5] == 'fish_']
print(fish_lst)

tg = tr.TerrainGenerator(size=terrain_size, sea_level=sea_level)

for fish_path in fish_lst:
    curr_fish_f = h5py.File(fish_path)
    curr_fish = fi.Fish.from_h5_group(curr_fish_f['fish'])

    for sim_num, curr_seed in enumerate(random_seeds):
        random.seed(curr_seed)
        np.random.seed(curr_seed)

        curr_terrain_map = tg.generate_binary_map(sigma=3., is_plot=False)
        curr_terrain = tr.BinaryTerrain(curr_terrain_map)
        curr_simulation = si.Simulation(terrain=curr_terrain, fish_list=[curr_fish],
                                        simulation_length=simulation_length, food_num=food_num)

        curr_simulation.initiate_simulation()
        curr_msg = curr_simulation.run(verbose=1)

        curr_sim_grp = curr_fish_f.create_group('simulation_' + util.int2str(sim_num, 3) + '_seed_' +
                                                util.int2str(curr_seed, 5))
        curr_sim_grp['ending_time'] = datetime.datetime.now().strftime('%y%m%d_%H_%M_%S')
        curr_sim_grp['random_seed'] = curr_seed
        curr_sim_grp['simulation_length'] = simulation_length
        curr_sim_grp['script_txt'] = inspect.getsource(sys.modules[__name__])
        curr_simulation.save_log_to_h5_grp(curr_sim_grp, msg=curr_msg, is_save_psp_waveforms=False)


    curr_fish_f.close()

