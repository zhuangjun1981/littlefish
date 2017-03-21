import os
import h5py
from littlefish.fish import fish
# from littlefish import utilities as util
# import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

save_name = 'fish_default.hdf5'

curr_folder = os.path.dirname(os.path.realpath(__file__))
save_folder = os.path.join(os.path.dirname(curr_folder), 'test')

if os.path.isfile(os.path.join(save_folder, save_name)):
    print('hdf5 file for default fish object already exists in the test folder. Abort saving ...')
else:
    default_fish = fish.Fish(name='fish_default')
    ff = h5py.File(os.path.join(save_folder, 'fish_default.hdf5'))
    grp = ff.create_group('fish')
    default_fish.to_h5_group(grp)
    ff.close()

print('for debug ...')

