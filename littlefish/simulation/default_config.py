# from __future__ import (absolute_import, division,
#                         print_function, unicode_literals)
# from builtins import *

import json
import os
import numpy as np


def generate_default_config_file(save_path):
    """
    generate a default config file (.json) at given path
    """

    # all configuration
    config_dict = {}

    # configuration for simulation
    config_dict_simulation = \
        {
            'SIMULATION_LENGTH' : int(1e5),  # number of time units (consider 0.1 ms per time unit) for simulation
            'RANDOM_SEED': 123,  # seed to generate random numbers
        }


    # configuration for brain
    config_dict_brain = \
        {
            'EYE_GAIN': 0.001,
            'EYE_INPUT_FILTER': [0.2, 0.6, 0.2],
            'EYE2_INPUT_FILTER': [0.15, 0.3, 0.15, 0.1, 0.2, 0.1],
            'EYE_DIRECTIONS': ['east', 'northeast', 'north', 'northwest', 'west', 'southwest', 'south', 'southeast'],
            'EYE_TYPES' : ['terrain', 'food', 'fish'],
            'EYE_BASELINE_RATE': 0.,
            'EYE_REFRACTORY_PERIOD': 10,
            'EYE_BORDER_VALUE': 1,

            'NEURON_REFRACTORY_PERIOD': 10,
            'NEURON_BASELINE_RATE': 0.00001,

            'MUSCLE_DIRECTIONS': ['east', 'north', 'west', 'south'],
            'MUSCLE_REFRACTORY_PERIOD': 5000,
            'MUSCLE_BASELINE_RATE': 0.0001,

            'CONNECTION_LATENCY': 30,
            'CONNECTION_AMPLITUDE': 0.0001,
            'CONNECTION_RISE_TIME': 5,
            'CONNECTION_DECAY_TIME': 10
        }


    # configuration for fish
    config_dict_fish = \
        {
            'FISH_MAX_HEALTH': 100.,
            'FISH_HEALTH_DECAY_RATE': 0.0001
        }


    config_dict.update({
                         'simulation': config_dict_simulation,
                         'brain': config_dict_brain,
                         'fish': config_dict_fish
                         })

    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=4, separators=(',', ': '), sort_keys=True)


if __name__ == '__main__':

    curr_folder = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(curr_folder, 'config.json')
    if os.path.isfile(config_path):
        os.remove(config_path)
    generate_default_config_file(config_path)
