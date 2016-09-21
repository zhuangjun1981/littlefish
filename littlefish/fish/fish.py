# from __future__ import (absolute_import, division,
#                         print_function, unicode_literals)
# from builtins import *

import brain as brain
import numpy as np
import pandas as pd

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

    self._simulation_status: 0, has not simulated
                             1, during simulation
                             2, after simulation

    self._simulation_history: pandas dataframe, columns: ['t_point', 'row', 'column', 'health']
    """

    def __init__(self, brain=None, max_health=FISH_MAX_HEALTH, health_decay_rate=FISH_HEALTH_DECAY_RATE,
                 land_penalty_rate=FISH_LAND_PENALTY_RATE, food_rate=FISH_FOOD_RATE):


        self._max_health = float(max_health)
        self._health_decay_rate = float(health_decay_rate)
        self._land_penalty_rate = land_penalty_rate
        self._food_rate = food_rate

        if brain is None:
            self._brain = brain.Brain()
        else:
            self._brain = brain

        self._simulation_status = 0
        self._curr_position = (None, None)
        self._curr_health = None
        self._end_time = None
        self._simulation_history = pd.DataFrame(columns=['t_point', 'row', 'column', 'health'])

    def get_simulation_status(self):
        return self._simulation_status

    def get_curr_position(self):
        return self._curr_position

    def get_curr_health(self):
        return self._curr_health

    def initialize_simulation(self, curr_position=(10, 10), world_map=np.zeros((20, 20), dtype=np.uint8)):
        """
        create all psp waveforms as internal attributes
        turn self._simulation_status to be 1
        """
        if len(curr_position) != 2:
            raise (ValueError, 'curr_position should contain two elements.')

        if (not isinstance(curr_position[0], int)) or (not isinstance(curr_position[1], int)):
            raise (ValueError, 'curr_position should contain two integers.')

        if len(world_map.shape) != 2:
            raise(ValueError, 'world_map should be a 2-d array.')

        if not np.issubdtype(world_map.dtype, np.integer):
            raise(ValueError, 'dtype of world_map should be integer.')

        if np.max(world_map) > 1 or np.min(world_map) < 0:
            raise(ValueError, 'world_map should only contain 0s and 1s.')

        if curr_position[0] < 1 or curr_position[0] > world_map.shape[0] - 2 or \
            curr_position[1] < 1 or curr_position[1] > world_map.shape[1] - 2:
            raise(ValueError, 'curr_position out of the range.')

        if np.sum(world_map[curr_position[0] - 1: curr_position[0] + 2,
                            curr_position[1] - 1: curr_position[1] + 2,]) > 0:
            raise(ValueError, 'the body of fish at curr_position cover 1s.')

        if self._simulation_status == 0:  # has not been simulated

            self._curr_position = curr_position
            self._curr_health = self._max_health
            self._simulation_history.append(pd.DataFrame([[0, self._curr_position[0], self._curr_position[1]]],
                                                         columns=['t_point', 'row', 'column']),
                                            ignore_index=True)
            self._simulation_status = 1
            print('Fish: Simulation initialization successful.')
            return True
        elif self._simulation_status == 1:
            raise(RuntimeError, 'Fish: Simulation initialization fail. Already in simulation.')
        elif self._simulation_status == 2:
            raise(RuntimeError, 'Fish: Simulation initialization fail. Already after simulation. '
                                'Please clear simulation data first.')

    def act(self, t_point, terrain_map, food_map=None, fish_map=None):

        if self._simulation_status == 0:
            raise(RuntimeError, 'Fish: action failed. simulation not initialized.')
        elif self._simulation_status == 2:
            raise(RuntimeError, 'Fish: action failed. simulation already stopped.')

        if not self._simulation_status == 1:
            raise(RuntimeError, 'Fish: action failed. self._simulation_status should be 1.')

        self._eval_terrain()
        self._eval_food()
        self._eval_fish()

        movement_attempt = self._brain.act(t_point, self._curr_position, terrain_map, food_map=None, fish_map=None)

        self._move(movement_attempt, terrain_map=terrain_map)

        self._curr_health += (- self._health_decay_rate)

        self._simulation_history.loc[len(self._simulation_history)] = [t_point, self._curr_position[0],
                                                                       self._curr_position[1], self._curr_health]

    def _eval_fish(self, fish_map):

        if self._simulation_status == 1:
            # todo: add code for action here.
            pass
        elif self._simulation_status == 0:
            raise (RuntimeError, 'Fish: cannot evaluate terrain. Simulation not started!')
        elif self._simulation_status == 2:
            raise (RuntimeError, 'Fish: cannot evaluate terrain. Simulation already stopped!')
        else:
            raise (RuntimeError, 'Fish: self._simulation_status should 0, 1 or 2.')

    def _eval_terrain(self, terrain_map):
        """
        Evaluate the coverage of fish body on terrain map, apply land penalty to current health accordingly
        """

        if self._simulation_status == 1:
            curr_body = terrain_map[self._curr_position[0] - 1: self._curr_position[0] + 2,
                                    self._curr_position[1] - 1: self._curr_position[1] + 2]

            self._curr_health += (-1. * np.sum(curr_body[:]) * self._land_penalty_rate)
        elif self._simulation_status == 0:
            raise(RuntimeError, 'Fish: cannot evaluate terrain. Simulation not started!')
        elif self._simulation_status == 2:
            raise(RuntimeError, 'Fish: cannot evaluate terrain. Simulation already stopped!')
        else:
            raise(RuntimeError, 'Fish: self._simulation_status should 0, 1 or 2.')

    def _eval_food(self, food_map):

        if self._simulation_status == 1:
            # todo: add code for action here. return the location where food has been eaten.
            pass
        elif self._simulation_status == 0:
            raise (RuntimeError, 'Fish: cannot evaluate terrain. Simulation not started!')
        elif self._simulation_status == 2:
            raise (RuntimeError, 'Fish: cannot evaluate terrain. Simulation already stopped!')
        else:
            raise (RuntimeError, 'Fish: self._simulation_status should 0, 1 or 2.')

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
            raise(RuntimeError, 'Fish: cannot evaluate terrain. Simulation not started!')
        elif self._simulation_status == 2:
            raise(RuntimeError, 'Fish: cannot evaluate terrain. Simulation already stopped!')
        else:
            raise(RuntimeError, 'Fish: self._simulation_status should 0, 1 or 2.')

    def clear_simulation(self):
        """
        clear all psp waveforms, clear action history for all neurons, clear position
        """
        if self._simulation_status == 1:
            raise(RuntimeError, 'Fish: clear simulation fail. Still in simulation.')

        elif self._simulation_status == 0 or self._simulation_status == 2:
            self._curr_position = (None, None)
            self._curr_health = None
            self._end_time = None
            self._clear_simulation_history()
            self._brain._simulation_data()
            self._simulation_status = 0
            print('Fish: all simulation data cleared. Simulation now can be initialized.')

    def _clear_simulation_history(self):
        self._simulation_history = pd.DataFrame(columns=['t_point', 'row', 'column, ''health'])

    def save_simulation(self, save_path):
        if self._simulation_status == 1:
            raise(RuntimeError, 'Fish: Simulation save fail. Still in simulation.')
        if self._simulation_status == 0:
            raise(ValueError, 'Fish: Simulation save fail. No simulation data found.')
        if self._simulation_status== 2:
            # todo: save simulation data
            pass

    def stop_simulation(self, end_time):
        if self._simulation_status == 1:
            self._simulation_status = 2
            self._end_time = end_time
            print('Fish: Simulation stopped.')
        elif self._simulation_status == 0:
            raise(RuntimeError, 'Fish: Stop simulation fail. Not in simulation.')
        elif self._simulation_status == 2:
            print('Fish: No need to stop simulation. Already stopped.')


if __name__ == '__main__':

    print('for debug ...')
