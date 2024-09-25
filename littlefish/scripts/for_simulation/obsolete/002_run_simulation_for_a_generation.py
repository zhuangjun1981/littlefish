import os
import sys

sys.path.extend([r"C:\Users\woodstocker\PycharmProjects\littlefish"])
import time
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

generation = 1

# five times of a standard fish's life span without hitting land and eating, 5 * max_health / health_decay_rate
# max_health of a standard fish: 100
# health_decay_rate of a standard fish: 0.01
simulation_length = 50000
sim_num = 3
terrain_size = [128, 128]
sea_level = 0.5
food_num = 200
hard_thr = 1000

gen_folder = os.path.join(
    data_folder, "generation_" + util.int2str(generation, 6) + "_single_thread"
)
os.chdir(gen_folder)

fish_lst = [f for f in os.listdir(gen_folder) if f[0:5] == "fish_"]
fish_lst.sort()
print("\n".join(fish_lst) + "\n")

t0 = time.time()

total_fish_num = len(fish_lst)

tg = tr.TerrainGenerator(size=terrain_size, sea_level=sea_level)

for fish_num, fish_path in enumerate(fish_lst):
    curr_fish_f = h5py.File(fish_path)
    curr_fish = fi.Fish.from_h5_group(curr_fish_f["fish"])

    for sim_ind in range(sim_num):
        curr_seed = random.randrange(2**32 - 1)
        random.seed(curr_seed)
        np.random.seed(curr_seed)

        print(
            "\n\n============================= {}/{}; fish: {}; simulation: {} start ===============================".format(
                fish_num, total_fish_num - 1, curr_fish.name, sim_ind
            )
        )

        curr_terrain_map = tg.generate_binary_map(sigma=3.0, is_plot=False)
        curr_terrain = tr.BinaryTerrain(curr_terrain_map)
        curr_simulation = si.Simulation(
            terrain=curr_terrain,
            fish_list=[curr_fish],
            simulation_length=simulation_length,
            food_num=food_num,
        )

        curr_simulation.initiate_simulation()
        curr_msg = curr_simulation.run(verbose=1)

        curr_sim_grp = curr_fish_f.create_group(
            "simulation_" + util.int2str(sim_ind, 3)
        )
        curr_sim_grp["ending_time"] = datetime.datetime.now().strftime(
            "%y%m%d_%H_%M_%S"
        )
        curr_sim_grp["random_seed"] = curr_seed
        curr_sim_grp["simulation_length"] = simulation_length
        curr_sim_grp["script_txt"] = inspect.getsource(sys.modules[__name__])
        curr_simulation.save_log_to_h5_grp(curr_sim_grp, is_save_psp_waveforms=False)

        print(
            "\n============================= {}/{}; fish: {}; simulation: {} end ===============================\n".format(
                fish_num, total_fish_num - 1, curr_fish.name, sim_ind
            )
        )

        if curr_simulation._end_t < hard_thr:
            print(
                "\nSimulation: fish life span was less than the hard threshold. end the simulation of current"
                "fish: {}".format(curr_fish.name)
            )
            break

    curr_fish_f.close()

print("time: {} min.".format((time.time() - t0) / 60))
