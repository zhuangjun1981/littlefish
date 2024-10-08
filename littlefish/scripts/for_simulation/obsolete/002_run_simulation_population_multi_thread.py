import os
import sys
import time

sys.path.extend([r"C:\Users\woodstocker\PycharmProjects\littlefish"])
from multiprocessing import Pool
import littlefish.core.utilities as util
import littlefish.core.simulation as si

SIMULATION_LENGTH = 50000
SIMULATION_NUM = 3
TERRAIN_SIZE = [64, 64]
SEA_PORTION = 0.5
FOOD_NUM = 50
HARD_THR = 5000


def simulation_fish_multiprocessing(simulation_params):
    f_path, fish_ind, fish_num = simulation_params
    si.simulate_one_fish(
        fish_path=f_path,
        simulation_length=SIMULATION_LENGTH,
        simulation_num=SIMULATION_NUM,
        terrain_size=TERRAIN_SIZE,
        sea_portion=SEA_PORTION,
        food_num=FOOD_NUM,
        hard_thr=HARD_THR,
        fish_ind=fish_ind,
        fish_num=fish_num,
    )


if __name__ == "__main__":
    data_folder = r"C:\little_fish_simulation_logs"
    generation_num = 98
    process_num = 8

    gen_folder = os.path.join(
        data_folder, "generation_" + util.int2str(generation_num, 6)
    )
    os.chdir(gen_folder)

    fish_lst = [
        f for f in os.listdir(gen_folder) if f[0:5] == "fish_" and f[-5:] == ".hdf5"
    ]
    fish_lst.sort()
    print("\n".join(fish_lst) + "\n")

    sim_params = zip(fish_lst, range(len(fish_lst)), [len(fish_lst)] * len(fish_lst))

    t0 = time.time()

    with Pool(5) as p:
        p.map(simulation_fish_multiprocessing, sim_params)

    print("time: {} min.".format((time.time() - t0) / 60))
