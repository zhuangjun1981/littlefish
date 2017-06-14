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

log_folder = r'C:\little_fish_simulation_logs\generation_000000'

if not os.path.isdir(log_folder):
    os.mkdir(log_folder)
os.chdir(log_folder)

save_name = 'fish_' + datetime.datetime.now().strftime('%y%m%d_%H_%M_%S') + '.hdf5'
save_f = h5py.File(save_name)
fish = fi.generate_standard_fish()
fish.set_name(save_name)
fish_grp = save_f.create_group('fish')
fish.to_h5_group(fish_grp)

print('for debug ...')