# from __future__ import (absolute_import, division,
#                         print_function, unicode_literals)
# from builtins import *

import brain as brain
import numpy as np
import pandas as pd
import h5py
import os

# unnecessary global varible
# SIMULATION_LENGTH = 100000
# FISH_MAX_HEALTH = 100.
# FISH_HEALTH_DECAY_RATE = 0.0001
# FISH_LAND_PENALTY_RATE = 0.005
# FISH_FOOD_RATE = 20.


class Fish(object):
    """
    the main fish class

    a 'fish' has a body occupies a 3x3 space.

    a 'fish' has health point (goes down over time and increases after eating a food). fish moves around in a 2d
    landscape (2-d binary map, 0 represents water, 1 represents land), when hits land, health point will go down
    quickly.

    the purpose of simulation is to train the fish use its eyes, brain and muscles to avoid land and look for food.

    the simulation works on "real-time" basis on a time unit axis (consider one time unit is equivalent to 0.1
    millisecond.

    self._brain: a brain.Brain object
    self._max_health: float, maximum health point a fish can have
    self._health_decay_rate: float, the constant rate of health reduction, health point / time unit
    self._land_penalty_rate: float, the penalty of health point, if the fish's body covers land pixels (1s) in
                             the terrain map, health point / (pixel * time unit)
    self._food_rate: float, the gaining of health point if fish's body covers food pixels (1s) in the food map,
                     health point / pixel. the food after taken will disappear, so no health gaining is a
                     transient event
    self._simulation_status: 0, has not simulated
                             1, during simulation
                             2, after simulation
    self._simulation_history: pandas dataframe, columns: ['t_point', 'row', 'column', 'health']
    """

    def __init__(self, name=None, mother_name=None, input_brain=None, max_health=100., health_decay_rate=0.0001,
                 land_penalty_rate=0.005, food_rate=20.):

        """
        :param input_brain: a brain.Brain object
        :param max_health: float, maximum health point a fish can have
        :param health_decay_rate: float, the constant rate of health reduction, health point / time unit
        :param land_penalty_rate: float, the penalty of health point, if the fish's body covers land pixels (1s) in
                                  the terrain map, health point / (pixel * time unit)
        :param food_rate: float, the gaining of health point if fish's body covers food pixels (1s) in the food map,
                          health point / pixel. the food after taken will disappear, so no health gaining is a
                          transient event
        """

        if name is None:
            self._name = ''
        else:
            self._name = name

        if mother_name is None:
            self._mother_name = ''
        else:
            self._mother_name = mother_name

        self._max_health = float(max_health)
        self._health_decay_rate = float(health_decay_rate)
        self._land_penalty_rate = land_penalty_rate
        self._food_rate = food_rate

        if input_brain is None:
            self._brain = brain.Brain()
        else:
            input_brain.check_integrity()
            self._brain = input_brain

        self._curr_health = None

        print('\nFish: fish object generated successfully.')

    def get_curr_health(self):
        return self._curr_health

    def has_history(self):
        return self._brain.has_psp_waveforms()

    def initialize_simulation(self):
        if self.has_history():
            raise RuntimeError('Fish: Can not initialize simulation. Already has history.')
        else:
            self._curr_health = self._max_health
            self._brain.generate_empty_psp_waveforms()
            print('Fish: Simulation initialized successfully.')

    def act(self, t_point, action_histories, psp_waveforms, body_position, terrain_map, food_map=None, fish_map=None):

        if not self.has_history():
            raise RuntimeError('Fish: action failure. Fish does not have history.')

        self._eval_terrain(terrain_map=terrain_map)
        food_taken_positions = self._eval_food(food_map=food_map)
        self._eval_fish(fish_map=fish_map)

        movement_attempt = self._brain.act(t_point=t_point, action_histories=action_histories,
                                           psp_waveforms=psp_waveforms, body_position=body_position,
                                           terrain_map=terrain_map, food_map=food_map, fish_map=fish_map)

        # update health
        self._curr_health += (- self._health_decay_rate)

        return movement_attempt, food_taken_positions

    def _eval_fish(self, fish_map):
        # todo: finish this method
        pass

    def _eval_terrain(self, terrain_map):
        """
        Evaluate the coverage of fish body on terrain map, apply land penalty to current health accordingly
        """

        if terrain_map is None:
            raise ValueError('Fish: _eval_terrain failure. terrain_map is None.')
        else:
            curr_body = terrain_map[self._curr_position[0] - 1: self._curr_position[0] + 2,
                                    self._curr_position[1] - 1: self._curr_position[1] + 2]
            self._curr_health += (-1. * np.sum(curr_body[:]) * self._land_penalty_rate)

    def _eval_food(self, food_map):

        food_taken_positions = []

        if food_map is not None:
            food_positions = zip(*np.where(food_map == 1))

            for food_position in food_positions:
                if self._curr_position[0] - 1 <= food_position[0] < self._curr_position[0] + 2 and \
                        self._curr_position[1] - 1 <= food_position[1] < self._curr_position[1] + 2:
                    self._curr_health += self._food_rate
                    food_taken_positions.append(food_position)

        return food_taken_positions

    def clear_history(self):
        """
        clear all psp waveforms, clear action history for all neurons, clear position
        """
        self._curr_health = None
        self._brain.clear_simulation_data()
        print('Fish: All history cleared. Simulation now can be initialized.')

    def to_h5_group(self, h5_group):

        h5_group.create_dataset('name', data=self._name)
        h5_group.create_dataset('mother_name', data=self._mother_name)
        h5_group.create_dataset('max_health', data=self._max_health)
        h5_group.create_dataset('health_decay_rate_per_tu', data=self._health_decay_rate)
        h5_group.create_dataset('land_penalty_rate_per_pixel_tu', data=self._land_penalty_rate)
        h5_group.create_dataset('food_rate_per_pixel', data=self._food_rate)
        brain_group = h5_group.create_group('brain')
        self._brain.to_h5_group(brain_group)


if __name__ == '__main__':

    # =========================================================================================
    # starting_position = (10, 10)
    # terrain_map = np.zeros((20, 20), dtype=np.uint8)
    #
    # fish = Fish()
    # fish.initialize_simulation(starting_position=starting_position, terrain_map=terrain_map)
    # print(fish.get_curr_health())
    # print(fish.get_curr_position())
    # fish.act(0, terrain_map=terrain_map)
    # =========================================================================================

    # =========================================================================================
    save_path = r"G:\little_fish_test\fish.hdf5"
    if os.path.isfile(save_path):
        os.remove(save_path)
    fish_group = h5py.File(save_path).create_group('fish')

    fish = Fish()
    fish.to_h5_group(fish_group)
    # =========================================================================================

    print('\nfor debug ...')
