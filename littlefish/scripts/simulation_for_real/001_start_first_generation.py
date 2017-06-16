import os
import datetime
import h5py
import littlefish.core.fish as fi

log_folder = r'C:\little_fish_simulation_logs\generation_000000'

if not os.path.isdir(log_folder):
    os.mkdir(log_folder)
os.chdir(log_folder)

save_name = 'fish_' + datetime.datetime.now().strftime('%y%m%d_%H_%M_%S') + '.hdf5'
save_f = h5py.File(save_name)
fish = fi.generate_standard_fish()
fish.set_name(os.path.splitext(save_name)[0])
fish_grp = save_f.create_group('fish')
fish.to_h5_group(fish_grp)
save_f['generation'] = 0

print('for debug ...')