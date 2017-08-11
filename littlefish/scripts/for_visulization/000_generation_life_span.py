import os
import h5py
import matplotlib.pyplot as plt

gen_folder = r"C:\little_fish_simulation_logs_2\generation_0000199"

life_spans = []

for fn in os.listdir(gen_folder):
    if fn[0:5] == 'fish_' and fn[-5:] == '.hdf5':
        curr_f = h5py.File(os.path.join(gen_folder, fn), 'r')
        sim_n = [s for s in curr_f.keys() if s[:11] == 'simulation_']
        if len(sim_n) != 1:
            raise ValueError('number of simulation for fish ({}) does not '
                             'equal one'.format(curr_f['fish/name'].value))
        sim_n = sim_n[0]
        life_spans.append(curr_f[sim_n]['simulation_log/last_time_point'].value)
        curr_f.close()

plt.hist(life_spans, bins=30)
plt.show()