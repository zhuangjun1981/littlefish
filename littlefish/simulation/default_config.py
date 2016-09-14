# from __future__ import (absolute_import, division,
#                         print_function, unicode_literals)
# from builtins import *

import json
import os


def generate_default_config_file(save_path):
    """
    generate a default config file (.json) at given path
    """

    config_dict = {}

    config_dict_simulation = \
        {
            'SIMULATION_LENGTH' : int(1e5), # number of time units (consider 0.1 ms per time unit) for simulation
        }

    config_dict.update({ 'simulation' : config_dict_simulation})

    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=4, separators=(',',': '))


if __name__ == '__main__':

    curr_folder = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(curr_folder, 'config.json')
    if os.path.isfile(config_path):
        os.remove(config_path)
    generate_default_config_file(config_path)
