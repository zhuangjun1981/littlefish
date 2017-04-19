import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Simulation(object):
    """
    Simulation class takes fish(s) and terrain to run the simulation of a fish's activity during its life
    """

    def __init__(self, terrain, fish_list, simulation_length=1000):
        """
        designed to run only once after creation

        :param terrain: terrain object, current terrain.terrain_2d.BinaryTerrain object
        :param fish_list: list of fish object (fish.fish.Fish class)
        :param simulation_length: positive integer, number of time points of the simulation 
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


    def initiate_simulation(self):
        """
        initiate simulation, check simulation status, creating simulation history variables
        
        :param simulation_length: int, number of time points of the simulation
        """

        if self._simulation_status == 0:

            self._simulation_status = 1

            self._simulation_histories = {}

            for fish in self._fish_list:
                simulation_history_curr_fish = {}
                simulation_history_curr_fish.update({'action_histories': fish._brain.generate_empty_action_histories()})
                simulation_history_curr_fish.update({'psp_waveforms': fish._brain.generate_empty_psp_waveforms(self._simulation_length)})

                life_history = pd.DataFrame(np.zeros((self._simulation_length,), dtype=[('pos_row', np.uint16),
                                                                                        ('pos_col', np.uint16),
                                                                                        ('health', np.float32)]))

                start_position = self._terrain.generate_fish_starting_position()
                life_history.loc[0, 'pos_row'] = start_position[0]
                life_history.loc[0, 'pos_col'] = start_position[1]
                life_history.loc[0, 'health'] = fish._max_health

                simulation_history_curr_fish.update({'life_history': life_history})
                self._simulation_histories.update({fish.name: simulation_history_curr_fish})

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

    def run(self):

        if self._simulation_status == 2:

            self._simulation_status = 3

            # todo: finish this simulation

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

    # print(sim._simulation_histories)
    sim.generating_fish_map(time_point=0, is_plot=True)
    # -------------------------------------------------------------------------



    print 'for debugging...'