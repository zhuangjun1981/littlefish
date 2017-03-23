import pandas as pd



class Simulation(object):
    """
    Simulation class takes fish(s) and terrain to run the simulation of a fish's activity during its life
    """

    def __init__(self, terrain, fish_list):
        """
        designed for only run once after creation

        :param terrain: terrain object, current terrain.terrain_2d.BinaryTerrain object
        :param fish_list: list of fish object (fish.fish.Fish class)
        :return: None
        """

        # simulation status: 0: not started yet; 1: during simulation; 2: after simulation
        self._simulation_status = 0
        self._terrain = terrain
        self._fish_list = fish_list


    def initiate_simulation(self):
        """
        initiate simulation, check simulation status, creating simulation history variables
        """

        # todo: unfinished, finish this method

        if self._simulation_status == 0:

            self._psp_waveforms_all_fish = {}
            self._action_histories_all_fish = {}

        elif self._simulation_status == 1:
            raise RuntimeError("Simulation: Can not initiate simulation. Already in simulation.")
        elif self._simulation_status == 2:
            raise RuntimeError("Simulation: Can not initiate simulation. Already stopped.")

        # self._curr_position = None
        # self._simulation_history = pd.DataFrame(columns=['t_point', 'row', 'col', 'health'])

    # def get_simulation_status(self):
    #     return self._simulation_status

    # def get_curr_position(self):
    #     return self._curr_position

    # def set_curr_position(self, position):
    #
    #     if len(position) != 2:
    #         raise ValueError('Fish: set body position failure. position does not have 2 elements.')
    #
    #     if not (isinstance(position[0], int) and isinstance(position[1], int)):
    #         raise ValueError('Fish: set body position failure. position does not contain 2 integers.')
    #
    #     if self._simulation_status == 0:
    #         self.clear_history()
    #         self._curr_position = position
    #         print('\nFish: self._curr_position set to ' + str(position) + ' before simulation.')
    #     elif self._simulation_status == 1:
    #         raise RuntimeError('Fish: set body position failure. Still in simulation.')
    #     elif self._simulation_status == 2:
    #         print('\nFish: attempt to reset body position ..., clearing all simulation data.')
    #         self.clear_history()
    #         self._curr_position = position
    #         print('\nFish: self._curr_position set to ' + str(position))

    # def initialize_simulation(self, starting_position, terrain_map):
    #     """
    #     create all psp waveforms as internal attributes
    #     turn self._simulation_status to be 1
    #     """
    #     if len(starting_position) != 2:
    #         raise ValueError('starting_position should contain two elements.')
    #
    #     if (not isinstance(starting_position[0], int)) or (not isinstance(starting_position[1], int)):
    #         raise ValueError('starting_position should contain two integers.')
    #
    #     if len(terrain_map.shape) != 2:
    #         raise ValueError('terrain_map should be a 2-d array.')
    #
    #     if not np.issubdtype(terrain_map.dtype, np.integer):
    #         raise ValueError('dtype of terrain_map should be integer.')
    #
    #     if np.max(terrain_map) > 1 or np.min(terrain_map) < 0:
    #         raise ValueError('terrain_map should only contain 0s and 1s.')
    #
    #     if starting_position[0] < 1 or starting_position[0] > terrain_map.shape[0] - 2 or \
    #             starting_position[1] < 1 or starting_position[1] > terrain_map.shape[1] - 2:
    #         raise ValueError('starting_position out of the range.')
    #
    #     if np.sum(terrain_map[starting_position[0] - 1: starting_position[0] + 2,
    #               starting_position[1] - 1: starting_position[1] + 2, ]) > 0:
    #         raise ValueError('the body of fish at starting_position covers land.')
    #
    #     if self._simulation_status == 0:  # has not been simulated
    #
    #         self._curr_position = np.array(starting_position, dtype=np.uint)
    #         self._curr_health = self._max_health
    #         self._simulation_history.append(pd.DataFrame([[0, self._curr_position[0], self._curr_position[1]]],
    #                                                      columns=['t_point', 'row', 'column']),
    #                                         ignore_index=True)
    #         self._simulation_status = 1
    #         print('Fish: Simulation initialized successfully.')
    #         return True
    #     elif self._simulation_status == 1:
    #         raise RuntimeError('Fish: Simulation initialization failure. Already in simulation.')
    #     elif self._simulation_status == 2:
    #         raise RuntimeError('Fish: Simulation initialization failure. Already after simulation. '
    #                            'Please clear simulation data first.')

    # def act()
    #
    #     self._move(movement_attempt, terrain_map=terrain_map)
    #     self._simulation_history.loc[len(self._simulation_history)] = [t_point, self._curr_position[0],
    #                                                                    self._curr_position[1], self._curr_health]

    # def _move(self, movement_attempt, terrain_map):
    #     """
    #     update self._curr_position according to the movement_attempt but not moving out of the terrain map
    #     """
    #     if self._simulation_status == 1:
    #         if self._curr_position[0] + movement_attempt[0] < 1:
    #             self._curr_position[0] = 1
    #         elif self._curr_position[0] + movement_attempt[0] > terrain_map.shape[0] - 2:
    #             self._curr_position[0] = terrain_map.shape[0] - 2
    #         else:
    #             self._curr_position[0] += movement_attempt[0]
    #
    #         if self._curr_position[1] + movement_attempt[1] < 1:
    #             self._curr_position[1] = 1
    #         elif self._curr_position[1] + movement_attempt[1] > terrain_map.shape[1] - 2:
    #             self._curr_position[1] = terrain_map.shape[1] - 2
    #         else:
    #             self._curr_position[1] += movement_attempt[1]
    #     elif self._simulation_status == 0:
    #         raise RuntimeError('Fish: cannot evaluate terrain. Simulation not started!')
    #     elif self._simulation_status == 2:
    #         raise RuntimeError('Fish: cannot evaluate terrain. Simulation already stopped!')
    #     else:
    #         raise RuntimeError('Fish: self._simulation_status should 0, 1 or 2.')

    # def _clear_simulation_history(self):
    #     self._simulation_history = pd.DataFrame(columns=['t_point', 'row', 'column, ''health'])

    # def stop_simulation(self, end_time):
    #     if self._simulation_status == 1:
    #         self._simulation_status = 2
    #         self._curr_time = end_time
    #         print('Fish: Simulation stopped.')
    #     elif self._simulation_status == 0:
    #         raise RuntimeError('Fish: Stop simulation failure. Not in simulation.')
    #     elif self._simulation_status == 2:
    #         print('Fish: No need to stop simulation. Already stopped.')