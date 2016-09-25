# from __future__ import (absolute_import, division,
#                         print_function, unicode_literals)
# from builtins import *

import brain as brain
import numpy as np
import pandas as pd
import h5py
import os

SIMULATION_LENGTH = 100000

FISH_MAX_HEALTH = 100.
FISH_HEALTH_DECAY_RATE = 0.0001
FISH_LAND_PENALTY_RATE = 0.005
FISH_FOOD_RATE = 20.


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

    def __init__(self, name=None, mother_name=None, input_brain=None, max_health=FISH_MAX_HEALTH,
                 health_decay_rate=FISH_HEALTH_DECAY_RATE, land_penalty_rate=FISH_LAND_PENALTY_RATE,
                 food_rate=FISH_FOOD_RATE):

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

        self._simulation_status = 0
        self._curr_position = None
        self._curr_health = None
        self._simulation_history = pd.DataFrame(columns=['t_point', 'row', 'col', 'health'])

        print('\nFish: fish object generated successfully.')

    def get_simulation_status(self):
        return self._simulation_status

    def get_curr_position(self):
        return self._curr_position

    def set_curr_position(self, position):

        if len(position) != 2:
            raise ValueError('Fish: set body position failure. position does not have 2 elements.')

        if not (isinstance(position[0], int) and isinstance(position[1], int)):
            raise ValueError('Fish: set body position failure. position does not contain 2 integers.')

        if self._simulation_status == 0:
            self.clear_simulation()
            self._curr_position = position
            print('\nFish: self._curr_position set to ' + str(position) + ' before simulation.')
        elif self._simulation_status == 1:
            raise RuntimeError('Fish: set body position failure. Still in simulation.')
        elif self._simulation_status == 2:
            print('\nFish: attempt to reset body position ..., clearing all simulation data.')
            self.clear_simulation()
            self._curr_position = position
            print('\nFish: self._curr_position set to ' + str(position))

    def get_curr_health(self):
        return self._curr_health

    def initialize_simulation(self, starting_position, terrain_map):
        """
        create all psp waveforms as internal attributes
        turn self._simulation_status to be 1
        """
        if len(starting_position) != 2:
            raise ValueError('starting_position should contain two elements.')

        if (not isinstance(starting_position[0], int)) or (not isinstance(starting_position[1], int)):
            raise ValueError('starting_position should contain two integers.')

        if len(terrain_map.shape) != 2:
            raise ValueError('terrain_map should be a 2-d array.')

        if not np.issubdtype(terrain_map.dtype, np.int):
            raise ValueError('dtype of terrain_map should be integer.')

        if np.max(terrain_map) > 1 or np.min(terrain_map) < 0:
            raise ValueError('terrain_map should only contain 0s and 1s.')

        if starting_position[0] < 1 or starting_position[0] > terrain_map.shape[0] - 2 or \
                starting_position[1] < 1 or starting_position[1] > terrain_map.shape[1] - 2:
            raise ValueError('starting_position out of the range.')

        if np.sum(terrain_map[starting_position[0] - 1: starting_position[0] + 2,
                  starting_position[1] - 1: starting_position[1] + 2, ]) > 0:
            raise ValueError('the body of fish at starting_position covers land.')

        if self._simulation_status == 0:  # has not been simulated

            self._curr_position = np.array(starting_position, dtype=np.int)
            self._curr_health = self._max_health
            self._simulation_history.append(pd.DataFrame([[0, self._curr_position[0], self._curr_position[1]]],
                                                         columns=['t_point', 'row', 'column']),
                                            ignore_index=True)
            self._simulation_status = 1
            print('Fish: Simulation initialized successfully.')
            return True
        elif self._simulation_status == 1:
            raise RuntimeError('Fish: Simulation initialization failure. Already in simulation.')
        elif self._simulation_status == 2:
            raise RuntimeError('Fish: Simulation initialization failure. Already after simulation. '
                               'Please clear simulation data first.')

    def act(self, t_point, terrain_map, food_map=None, fish_map=None):

        if self._simulation_status == 0:
            raise RuntimeError('Fish: action failure. simulation not initialized.')
        elif self._simulation_status == 2:
            raise RuntimeError('Fish: action failure. simulation already stopped.')

        if not self._simulation_status == 1:
            raise RuntimeError('Fish: action failure. self._simulation_status should be 1.')

        self._eval_terrain(terrain_map=terrain_map)
        food_taken_positions = self._eval_food(food_map=food_map)
        self._eval_fish(fish_map=fish_map)

        movement_attempt = self._brain.act(t_point, self._curr_position, terrain_map, food_map=None, fish_map=None)

        self._move(movement_attempt, terrain_map=terrain_map)

        self._curr_health += (- self._health_decay_rate)

        self._simulation_history.loc[len(self._simulation_history)] = [t_point, self._curr_position[0],
                                                                       self._curr_position[1], self._curr_health]

        return food_taken_positions

    def _eval_fish(self, fish_map):

        if fish_map is not None:

            if self._simulation_status == 1:
                # todo: add code for action here.
                pass
            elif self._simulation_status == 0:
                raise RuntimeError('Fish: cannot evaluate terrain. Simulation not started!')
            elif self._simulation_status == 2:
                raise RuntimeError('Fish: cannot evaluate terrain. Simulation already stopped!')
            else:
                raise RuntimeError('Fish: self._simulation_status should 0, 1 or 2.')

    def _eval_terrain(self, terrain_map):
        """
        Evaluate the coverage of fish body on terrain map, apply land penalty to current health accordingly
        """

        if terrain_map is None:
            raise ValueError('Fish: _eval_terrain failure. terrain_map is None.')
        else:
            if self._simulation_status == 1:

                curr_body = terrain_map[self._curr_position[0] - 1: self._curr_position[0] + 2,
                                        self._curr_position[1] - 1: self._curr_position[1] + 2]
                self._curr_health += (-1. * np.sum(curr_body[:]) * self._land_penalty_rate)

            elif self._simulation_status == 0:
                raise RuntimeError('Fish: cannot evaluate terrain. Simulation not started!')
            elif self._simulation_status == 2:
                raise RuntimeError('Fish: cannot evaluate terrain. Simulation already stopped!')
            else:
                raise RuntimeError('Fish: self._simulation_status should 0, 1 or 2.')

    def _eval_food(self, food_map):

        if food_map is not None:

            if self._simulation_status == 1:

                food_positions = zip(*np.where(food_map == 1))
                food_taken_positions = []

                for food_position in food_positions:
                    if self._curr_position[0] - 1 <= food_position[0] < self._curr_position[0] + 2 and \
                            self._curr_position[1] - 1 <= food_position[1] < self._curr_position[1] + 2:
                        self._curr_health += self._food_rate
                        food_taken_positions.append(food_position)

                return food_taken_positions

            elif self._simulation_status == 0:
                raise RuntimeError('Fish: cannot evaluate terrain. Simulation not started!')
            elif self._simulation_status == 2:
                raise RuntimeError('Fish: cannot evaluate terrain. Simulation already stopped!')
            else:
                raise RuntimeError('Fish: self._simulation_status should 0, 1 or 2.')

    def _move(self, movement_attempt, terrain_map):
        """
        update self._curr_position according to the movement_attempt but not moving out of the terrain map
        """
        if self._simulation_status == 1:
            if self._curr_position[0] + movement_attempt[0] < 1:
                self._curr_position[0] = 1
            elif self._curr_position[0] + movement_attempt[0] > terrain_map.shape[0] - 2:
                self._curr_position[0] = terrain_map.shape[0] - 2
            else:
                self._curr_position[0] += movement_attempt[0]

            if self._curr_position[1] + movement_attempt[1] < 1:
                self._curr_position[1] = 1
            elif self._curr_position[1] + movement_attempt[1] > terrain_map.shape[1] - 2:
                self._curr_position[1] = terrain_map.shape[1] - 2
            else:
                self._curr_position[1] += movement_attempt[1]
        elif self._simulation_status == 0:
            raise RuntimeError('Fish: cannot evaluate terrain. Simulation not started!')
        elif self._simulation_status == 2:
            raise RuntimeError('Fish: cannot evaluate terrain. Simulation already stopped!')
        else:
            raise RuntimeError('Fish: self._simulation_status should 0, 1 or 2.')

    def clear_simulation(self):
        """
        clear all psp waveforms, clear action history for all neurons, clear position
        """
        if self._simulation_status == 1:
            raise RuntimeError('Fish: clear simulation failure. Still in simulation.')

        elif self._simulation_status == 0:
            self._curr_position = None
            self._curr_health = None
            self._curr_time = None
            self._clear_simulation_history()
            self._brain.clear_simulation_data()
            self._simulation_status = 0
            print('Fish: simulation not started. All simulation data cleared. Simulation now can be initialized.')

        elif self._simulation_status == 2:
            self._curr_position = None
            self._curr_health = None
            self._curr_time = None
            self._clear_simulation_history()
            self._brain.clear_simulation_data()
            self._simulation_status = 0
            print('Fish: all simulation data cleared. Simulation now can be initialized.')

    def _clear_simulation_history(self):
        self._simulation_history = pd.DataFrame(columns=['t_point', 'row', 'column, ''health'])

    def to_h5_group(self, h5_group):
        if self._simulation_status == 1:
            raise RuntimeError('Fish: Simulation save failure. Still in simulation.')
        elif self._simulation_status == 0 or self._simulation_status == 2:

            h5_group.attrs['name'] = self._name
            h5_group.attrs['mother_name'] = self._mother_name
            h5_group.attrs['max_health'] = self._max_health
            h5_group.attrs['health_decay_rate_per_tu'] = self._health_decay_rate
            h5_group.attrs['land_penalty_rate_per_pixel_tu'] = self._land_penalty_rate
            h5_group.attrs['food_rate_per_pixel'] = self._food_rate
            brain_group = h5_group.create_group('brain')
            self._brain.to_h5_group(brain_group)
            simulation_status_dset = h5_group.create_dataset('simulation_status', data=self._simulation_status)
            simulation_status_dset.attrs['simulation_status_code'] = '0: before simulation; 1: during simulation; ' \
                                                                     '2: after simulation'
            if self._curr_position is None:
                h5_group.create_dataset('curr_position_row_col', data=np.nan)
            else:
                h5_group.create_dataset('curr_position_row_col', data=self._curr_position)

            if self._curr_health is None:
                h5_group.create_dataset('curr_health', data=np.nan)
            else:
                h5_group.create_dataset('curr_health', data=self._curr_health)

            simulation_history_group = h5_group.create_group('simulation_history')
            simulation_history_group.create_dataset('t_point_tu',
                                                    data=np.array(self._simulation_history['t_point'], dtype=np.uint))
            simulation_history_group.create_dataset('body_position',
                                                    data=np.array(self._simulation_history.loc[:,['row', 'col']],
                                                                  dtype=np.uint))
            simulation_history_group.create_dataset('health',
                                                    data=np.array(self._simulation_history['health'], dtype=np.float))
        else:
            raise RuntimeError('Fish: simulation save failure. Invalid self._simulation_status.')

    def stop_simulation(self, end_time):
        if self._simulation_status == 1:
            self._simulation_status = 2
            self._curr_time = end_time
            print('Fish: Simulation stopped.')
        elif self._simulation_status == 0:
            raise RuntimeError('Fish: Stop simulation failure. Not in simulation.')
        elif self._simulation_status == 2:
            print('Fish: No need to stop simulation. Already stopped.')


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
    save_path = r"D:\little_fish_test\fish.hdf5"
    if os.path.isfile(save_path):
        os.remove(save_path)
    fish_group = h5py.File(save_path).create_group('fish')

    fish = Fish()
    fish.to_h5_group(fish_group)

    # =========================================================================================

    print('\nfor debug ...')
