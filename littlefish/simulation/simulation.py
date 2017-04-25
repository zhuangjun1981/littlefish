import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Simulation(object):
    """
    Simulation class takes fish(s) and terrain to run the simulation of a fish's activity during its life
    """

    def __init__(self, terrain, fish_list, simulation_length=1000, food_num=5):
        """
        designed to run only once after creation

        :param terrain: terrain object, current terrain.terrain_2d.BinaryTerrain object
        :param fish_list: list of fish object (fish.fish.Fish class)
        :param simulation_length: positive integer, number of time points of the simulation 
        :param food_num: non-negative integer, number of food pixels in food map
        :return: None
        """

        # simulation status: 0: not initialized yet;
        #                    1: started initialization
        #                    2: initialization finished
        #                    3: started simulation
        #                    4: after simulation
        self._simulation_status = 0
        self._terrain = terrain
        self._fish_list = fish_list
        self._simulation_length = simulation_length
        self._food_num = food_num

    def initiate_simulation(self):
        """
        initiate simulation, check simulation status, creating simulation history variables
        
        :param simulation_length: int, number of time points of the simulation
        """

        if self._simulation_status == 0:

            self._simulation_status = 1

            self._simulation_histories = {}

            #  generate food map, food position history and initial food positions
            self._food_map = np.zeros(self._terrain.get_terrain_shape(), dtype=np.uint8)
            self._food_map, food_pos_array = self._terrain.update_food_map(food_num=self._food_num,
                                                                           food_map=self._food_map)
            food_pos_history = pd.DataFrame([[np.empty((self._food_num, 2))]] * self._simulation_length,
                                            columns=['food_pos'])
            # food_pos_array = np.arange(10).reshape((5, 2))
            food_pos_history.loc[0, 'food_pos'] = food_pos_array
            self._simulation_histories.update({'food_pos_history': food_pos_history})

            #  generate non-overlapping positions for all fish
            fish_start_positions = self._terrain.generate_fish_starting_position(len(self._fish_list))

            #  for each fish generate action histories, psp waveforms and life history
            for i, fish in enumerate(self._fish_list):
                simulation_history_curr_fish = {}
                simulation_history_curr_fish.update({'action_histories': fish._brain.generate_empty_action_histories()})
                simulation_history_curr_fish.update({'psp_waveforms': fish._brain.generate_empty_psp_waveforms(self._simulation_length)})

                life_history = pd.DataFrame(np.zeros((self._simulation_length,), dtype=[('pos_row', np.uint16),
                                                                                        ('pos_col', np.uint16),
                                                                                        ('health', np.float32)]))

                start_position = fish_start_positions[i]
                life_history.loc[0, 'pos_row'] = start_position[0]
                life_history.loc[0, 'pos_col'] = start_position[1]
                life_history.loc[0, 'health'] = fish._max_health

                simulation_history_curr_fish.update({'life_history': life_history})
                self._simulation_histories.update({fish.name: simulation_history_curr_fish})

            #  simulation initialization finished, change simulation status
            self._simulation_status = 2

        else:
            raise RuntimeError("Simulation: Cannot initiate simulation. Already initialized.")

    def get_simulation_status(self):
        return self._simulation_status

    @property
    def terrain_shape(self):
        return self._terrain.get_terrain_shape()

    @property
    def simulation_length(self):
        return self._simulation_length

    @property
    def fish_names(self):
        return [f.get_name() for f in self._fish_list]

    def generating_fish_map(self, time_point, is_plot=False):
        """
        generating fish map containing all fishes according to self._fish_list at time_point
        
        :param time_point: non-negative int, time_point in the simulation
        :param is_plot: bool
        
        return: fish_map, 2d array, uint16, 
        """

        if self._simulation_status in [2, 3, 4]:
            fish_map = np.zeros(self.terrain_shape, dtype=np.uint16)
            for fish_name, fish_history in self._simulation_histories.items():
                curr_health = fish_history['life_history'].loc[time_point, 'health']
                if curr_health > 0:
                    curr_row, curr_col = tuple(fish_history['life_history'].loc[time_point, ['pos_row', 'pos_col']])
                    curr_row = int(curr_row); curr_col = int(curr_col)
                    if (curr_row - 1 < 0) or (curr_row + 2 > self.terrain_shape[0]) or \
                        (curr_col - 1 < 0) or (curr_col + 2 > self.terrain_shape[1]):
                        print('Simulation: cannot plot fish {}. Position out of terrain boundary'.format(fish_name))
                        return None
                    fish_map[curr_row-1: curr_row+2, curr_col-1: curr_col+2] += 1

            if is_plot:
                f = plt.figure(figsize=(8, 7))
                ax = f.add_subplot(111)
                fig = ax.imshow(fish_map, cmap='magma', interpolation='nearest')
                f.colorbar(fig)
                ax.set_title('fish map at time: {}'.format(time_point))
                plt.show()

            return fish_map
        else:
            raise RuntimeError("Simulation: Cannot generate fish_map. Simulation not initialized properly.")

    def get_fish_health_status(self, t_point, verbose=False):
        """
        check the health status of all the fish in the simulation at the time point, t_point.
        if the time point is not simulated (at least one fish position is [0, 0]), an error will raise.
        :param t_point: non-negative integer, time point to evaluate
        :param verbose: bool, if True, print the health_status
        :return: dataframe, with index = fish_name, column=['health']
        """
        health_status = pd.DataFrame(index=self.fish_names, columns=['health'])

        for fish_n in self.fish_names:

            fish_life_his = self._simulation_histories[fish_n]['life_history']

            if fish_life_his.loc[t_point, 'pos_row'] == 0 and fish_life_his.loc[t_point, 'pos_col'] == 0:
                raise RuntimeError("Simulation: Try to get fish health status at an unsimulated time point.")

            health_status.loc[fish_n, 'health'] = float(fish_life_his.loc[t_point, 'health'])

        if verbose:
            print(health_status)

        return health_status

    def is_all_fish_dead(self, t_point):
        """
        check if the health of all fish are no higher than 0. at t_point in the simulation
        if the time point is not simulated (at least one fish position is [0, 0]), an error will raise.
        :param t_point: non-negative integer, time point to evaluate
        :return: bool
        """
        health_status = self.get_fish_health_status(t_point=t_point)
        return np.all(health_status['health'] <= 0.)

    def _is_dead(self, fish_n, t_point):
        """
        check if the health of a single fish is no higher than 0. in the simulation
        if the time point is not simulated (at least one fish position is [0, 0]), an error will raise.
        :param fish_n: str, name of the fish
        :param t_point: non-negative integer, time point to evaluate
        :return: bool
        """

        if self._simulation_histories[fish_n]['life_history'].loc[t_point, 'pos_row'] == 0 and \
                        self._simulation_histories[fish_n]['life_history'].loc[t_point, 'pos_col'] == 0:
            raise RuntimeError("Simulation: Try to get fish health status at an unsimulated time point.")
        return self._simulation_histories[fish_n]['life_history'].loc[t_point, 'health'] <= 0.

    def run(self):

        if self._simulation_status == 2:

            self._simulation_status = 3

            # todo: finish this simulation

            alive_fish_list = list(self._fish_list)

            curr_t = 0

            while len(alive_fish_list) > 0 and curr_t < self.simulation_length:

                for curr_fish in alive_fish_list:

                    curr_fish_history = self._simulation_histories[curr_fish.name]
                    curr_position = [curr_fish_history['life_history'].loc[curr_t, 'pos_row'],
                                     curr_fish_history['life_history'].loc[curr_t, 'pos_col']]
                    curr_health = self._simulation_histories[curr_fish.name]['life_history'].loc[curr_t, 'health']

                    curr_fish.act(t_point=curr_t, curr_position=curr_position, curr_health=curr_health,
                                  action_histories=curr_fish_history['action_histories'],
                                  psp_waveforms=curr_fish_history['psp_waveforms'],
                                  terrain_map=self._terrain._terrain_map,
                                  food_map=None, fish_map=None)

                curr_t += 1

            else:
                if len(alive_fish_list) == 0:
                    print("Simulation: End of Simulation. All fish are dead. "
                          "Last simulated time point: {}".format(curr_t))
                if curr_t == self.simulation_length:
                    print("Simulation: End of Simulation. Prespecified simulation length reached.")

            self._simulation_status = 4

        else:
            raise RuntimeError("Simulation: Cannot run simulation. Simulation not initialized properly.")


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


if __name__ == '__main__':

    # -------------------------------------------------------------------------
    import littlefish.terrain.terrain_2d as tr
    import littlefish.fish.fish as fi
    tg = tr.TerrainGenerator()
    terrain_map = tg.generate_binary_map()
    terrain = tr.BinaryTerrain(terrain_map)
    fish = fi.Fish()
    sim = Simulation(terrain, [fish])
    sim.initiate_simulation()

    print(sim._simulation_histories)
    # sim.generating_fish_map(time_point=0, is_plot=True)
    # sim.get_fish_health_status(0, verbose=True)
    # print(sim.is_all_fish_dead(0))
    # sim.run()
    # -------------------------------------------------------------------------



    print 'for debugging...'