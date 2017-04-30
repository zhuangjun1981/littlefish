# from __future__ import (absolute_import, division,
#                         print_function, unicode_literals)
# from builtins import *

import brain as br
import numpy as np
import pandas as pd
import h5py
import os

# unnecessary global varible
# SIMULATION_LENGTH = 100000
# FISH_MAX_HEALTH = 100.
# FISH_HEALTH_DECAY_RATE = 0.0001
# FISH_LAND_PENALTY_RATE = 0.005
# FISH_FOOD_RATE = 20

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

    def __init__(self, name=None, mother_name=None, brain=None, max_health=100., health_decay_rate=0.0001,
                 land_penalty_rate=0.005, food_rate=20.):

        """
        :param brain: a brain.Brain object
        :param max_health: float, maximum health point a fish can have
        :param health_decay_rate: float, the constant rate of health reduction, health point / time unit
        :param land_penalty_rate: float, the penalty of health point, if the fish's body covers land pixels (1s) in
                                  the terrain map, health point / (pixel * time unit)
        :param food_rate: float, the gaining of health point if fish's body covers food pixels (1s) in the food map,
                          health point / pixel. the food after taken will disappear, so no health gaining is a
                          transient event
        """

        if name is None:
            self._name = 'test_fish'
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

        if brain is None:
            self._brain = br.Brain()
        else:
            brain.check_integrity()
            self._brain = brain

        # self._curr_health = None

        print('\nFish: fish object generated successfully.')
    #
    # def get_curr_health(self):
    #     return self._curr_health

    def get_name(self):
        return self._name

    @property
    def name(self):
        return self.get_name()

    def get_max_health(self):
        return self._max_health

    def act(self, t_point, curr_position, curr_health, action_histories, psp_waveforms, terrain_map,
            food_map=None, fish_map=None):
        """
        simulate the fish's action at a given time point
        
        :param t_point: non-negative int, time point
        :param curr_position: list or tuple of two non-negative integers, coordinates of fish center position, 
                              [row, col]
        :param curr_health: positive float, health point at the beginning of t_point
        :param action_histories: data frame of lists, each list is the action history of a particular neuron, in the
                                 same order as self._neurons data frame, columns = ['action_history'], used by 
                                 self._brain.act()
        :param psp_waveforms: 2d-array of floats, psp waveforms of all neurons in the brain, each row represents one
                              neuron in the same order as self._neurons data frame, each column represents a time point,
                              used by self._brain.act()
        :param terrain_map: 2d array, with only 0s (water) and 1s (land). represents the land scape of the world
        :param food_map: 2d array, with only 0s (no food) and 1s (food). represents the distribution of food
        :param fish_map: not fully implemented right now.
        :return updated_health: float, health point at the end of t_point
                movement_attempt: list of two integers, the attempt the fish is trying to move [row_shift, col_shift].
                                  None if updated_health is below 0 (fish is dead).
        """

        updated_health = curr_health

        # evaluate food
        if food_map is not None:
            body_food_overlap = self._eval_food(food_map=food_map, curr_position=curr_position)
            updated_health = self._eat_food(body_food_overlap=body_food_overlap, curr_health=updated_health)

            # update food map
            food_map[curr_position[0] - 1: curr_position[0] + 2, curr_position[1] - 1: curr_position[1] + 2] = 0

        # evaluate the extend of how much of the fish is on the land
        body_land_overlap = self._eval_terrain(terrain_map=terrain_map, curr_position=curr_position)

        # update current health with land penalty
        updated_health = updated_health - body_land_overlap * self._land_penalty_rate

        if fish_map is not None:
            self._eval_fish(fish_map=fish_map)

        # update health
        updated_health = updated_health - self._health_decay_rate

        if updated_health > 0:  # still alive
            movement_attempt = self._brain.act(t_point=t_point, action_histories=action_histories,
                                               psp_waveforms=psp_waveforms, body_position=curr_position,
                                               terrain_map=terrain_map, food_map=food_map,
                                               fish_map=fish_map)
        else:
            movement_attempt = None

        return updated_health, movement_attempt

    def _eval_terrain(self, terrain_map, curr_position):
        """
        Evaluate the coverage of fish body on terrain map, return the sum of all terrain pixels that are covered by
        the fish body
        """

        if terrain_map is None:
            raise ValueError('Fish: _eval_terrain failure. terrain_map is None.')
        else:
            curr_body = terrain_map[curr_position[0] - 1: curr_position[0] + 2,
                                    curr_position[1] - 1: curr_position[1] + 2]
            body_land_overlap = np.sum(curr_body.flat)

        return body_land_overlap

    def _eval_food(self, food_map, curr_position):
        """
        find out how many foods are covered by the fish body

        :param food_map: 2d array, binary map of current food
        :param curr_position: tuple of two positive ints, (row, col) of current location of fish
        :return: non-negative int, number of food taken
        """

        curr_body = food_map[curr_position[0] - 1: curr_position[0] + 2,
                             curr_position[1] - 1: curr_position[1] + 2]
        body_food_overlap = np.sum(curr_body.flat)

        return body_food_overlap

    def _eat_food(self, body_food_overlap, curr_health):
        """
        count the number of food to be taken, add relevant HP to curr_health, but not exceed the maximum health
        """
        curr_health += self._food_rate * body_food_overlap
        if curr_health > self._max_health:
            curr_health = self._max_health
        return curr_health

    def _eval_fish(self, fish_map):
        # todo: finish this method
        pass

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
    # save_path = r"G:\little_fish_test\fish.hdf5"
    # if os.path.isfile(save_path):
    #     os.remove(save_path)
    # fish_group = h5py.File(save_path).create_group('fish')
    #
    # fish = Fish()
    # fish.to_h5_group(fish_group)
    # =========================================================================================

    print('\nfor debug ...')
