import os
import sys
import time
sys.path.extend([r"C:\Users\woodstocker\PycharmProjects\littlefish"])
from multiprocessing import Pool
import littlefish.core.utilities as util
import littlefish.core.simulation as si


SIMULATION_LENGTH = 50000
SIMULATION_NUM = 3
TERRAIN_SIZE = [128, 128]
SEA_LEVEL = 0.5
FOOD_NUM = 200
HARD_THR = 1000


def simulation_fish_multiprocessing(f_path):
    si.simulate_one_fish(fish_path=f_path,
                         simulation_length=SIMULATION_LENGTH,
                         simulation_num=SIMULATION_NUM,
                         terrain_size=TERRAIN_SIZE,
                         sea_level=SEA_LEVEL,
                         food_num=FOOD_NUM,
                         hard_thr=HARD_THR,
                         fish_ind=0,
                         fish_num=0)


if __name__ == '__main__':

    data_folder = r"C:\little_fish_simulation_logs"
    generation_num = 1
    process_num = 6

    gen_folder = os.path.join(data_folder, 'generation_' + util.int2str(generation_num, 6))
    os.chdir(gen_folder)

    fish_lst = [f for f in os.listdir(gen_folder) if f[0:5] == 'fish_' and f[-5:] == '.hdf5']
    fish_lst.sort()
    print('\n'.join(fish_lst) + '\n')

    t0 = time.time()

    with Pool(5) as p:
        p.map(simulation_fish_multiprocessing, fish_lst)

    print("time: {} min.".format((time.time() - t0) / 60))






