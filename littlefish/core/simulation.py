import datetime
import os
import time

import h5py
import littlefish.core.utilities as util
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Simulation(object):
    """
    Simulation class takes fish(s) and terrain to run the simulation of a fish's activity during its life
    """

    def __init__(self, terrain, fish_list, simulation_length=100, food_num=5):
        """
        designed to run only once after creation

        :param terrain: terrain object, current terrain.terrain_2d.BinaryTerrain object
        :param fish_list: list of fish object (fish.fish.Fish class)
        :param simulation_length: positive integer, number of time points of the simulation 
        :param food_num: positive integer, number of food pixels in food map
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

        if food_num < 1 or not util.is_integer(food_num):
            raise ValueError("Simulation: food_num should be a positive integer.")
        else:
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

            self._simulation_histories.update({'message': ''})

            fish_movement_num = {}

            self._simulation_histories.update({'food_pos_history':
                                                   np.zeros((self._simulation_length, self._food_num, 2),
                                                            dtype=np.uint32)})

            #  generate non-overlapping positions for all fish
            fish_start_positions = self._terrain.generate_fish_starting_position(len(self._fish_list))

            #  for each fish generate action histories, psp waveforms and life history
            for i, fish in enumerate(self._fish_list):
                simulation_history_curr_fish = {}
                simulation_history_curr_fish.update({'action_histories': fish._brain.generate_empty_action_histories()})
                simulation_history_curr_fish.update({'psp_waveforms': fish._brain.generate_empty_psp_waveforms(self._simulation_length)})
                simulation_history_curr_fish.update({'total_moves': 0})

                life_history = pd.DataFrame(np.zeros((self._simulation_length,), dtype=[('pos_row', np.uint16),
                                                                                        ('pos_col', np.uint16),
                                                                                        ('health', np.float64)]))

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

    @property
    def food_num(self):
        return self._food_num

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

    def _get_fish_status(self, t_point, fish_history):

        curr_position = [int(fish_history['life_history'].loc[t_point, 'pos_row']),
                         int(fish_history['life_history'].loc[t_point, 'pos_col'])]
        curr_health = fish_history['life_history'].loc[t_point, 'health']
        return curr_position, curr_health

    def _move_fish(self, curr_fish, curr_t, curr_pos, curr_health, movement_attempt):

        msg = ''

        # no movement attempt
        if np.array_equal(movement_attempt, np.array([0, 0], np.int8)):
            new_pos = curr_pos
        else:
            # update fish's body center postion at curr_t + 1
            new_pos_row = curr_pos[0] + movement_attempt[0]
            new_pos_row = max([1, new_pos_row])
            new_pos_row = min([new_pos_row, self.terrain_shape[0] - 2])

            new_pos_col = curr_pos[1] + movement_attempt[1]
            new_pos_col = max([1, new_pos_col])
            new_pos_col = min([new_pos_col, self.terrain_shape[1] - 2])

            new_pos = [new_pos_row, new_pos_col]

            if new_pos != curr_pos:
                msg += 'Time: {:08d}; Fish: {}; Health: {:3.4f}; move from [{:3d}, {:3d}] to [{:3d}, {:3d}].'.\
                    format(curr_t, curr_fish.name, curr_health, curr_pos[0], curr_pos[1], new_pos[0], new_pos[1])

        return new_pos, msg

    def run(self, verbose=1):
        """
        
        :param verbose: 
        :return: print out message as a string
        """

        if self._simulation_status == 2:

            # at t0
            self._simulation_status = 3
            t0 = time.time()

            curr_msg = '\nstart of simulation. start time: {}'.\
                format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print(curr_msg)
            self._simulation_histories['message'] += (curr_msg)

            alive_fish_list = list(self._fish_list)
            curr_t = 0
            curr_progress = -1  # percentage finished, for printing the simulation progress

            while len(alive_fish_list) > 0 and curr_t < self.simulation_length:

                # print simulation progress
                if (verbose > 0) and (self._simulation_length > 100):
                    if curr_t // (self.simulation_length // 10) > curr_progress:
                        curr_msg = '\n{:09.2f} second: {:2d} %'.\
                            format(time.time() - t0, (curr_t // (self.simulation_length // 10)) * 10)
                        print(curr_msg)
                        self._simulation_histories['message'] += ('\n' + curr_msg)
                        curr_progress = curr_t // (self.simulation_length // 10)

                # update food position
                curr_food_positions = self._terrain.update_food_map(self.food_num, self._food_map)
                self._simulation_histories['food_pos_history'][curr_t] = curr_food_positions

                dead_fish_list = []  # list of fish is going to die at curr_t

                for curr_fish in alive_fish_list:  # loop through current live fish

                    # get current fish status
                    curr_fish_history = self._simulation_histories[curr_fish.name]
                    curr_position, curr_health = self._get_fish_status(curr_t, curr_fish_history)

                    _ = curr_fish.act(t_point=curr_t,
                                      curr_position=curr_position,
                                      curr_health=curr_health,
                                      action_histories=curr_fish_history['action_histories'],
                                      psp_waveforms=curr_fish_history['psp_waveforms'],
                                      terrain_map=self._terrain._terrain_map,
                                      food_map=self._food_map, fish_map=None)

                    updated_health, movement_attempt, food_eated = _

                    # print eat food message
                    if (food_eated > 0) and (verbose > 0):
                        curr_msg = "Time: {:08d}; Fish: {}; eated {} food pellet(s). HP: {:3.4f} -> {:3.4f}."\
                            .format(curr_t, curr_fish.name, food_eated, curr_health,
                                                                updated_health)
                        print(curr_msg)
                        self._simulation_histories['message'] += ('\n' + curr_msg)

                    if curr_t < self.simulation_length - 1:  # if not at the end of simulation

                        # update fish's health at curr_t + 1
                        curr_fish_history['life_history'].loc[curr_t + 1, 'health'] = updated_health

                        if updated_health > 0:  # if not dead
                            new_pos, curr_msg = self._move_fish(curr_fish, curr_t, curr_position, curr_health,
                                                                movement_attempt=movement_attempt)

                            if curr_msg:
                                self._simulation_histories['message'] += ('\n' + curr_msg)
                                self._simulation_histories[curr_fish.name]['total_moves'] += 1

                                if verbose > 1:
                                    print curr_msg

                            curr_fish_history['life_history'].loc[curr_t + 1, 'pos_row'] = new_pos[0]
                            curr_fish_history['life_history'].loc[curr_t + 1, 'pos_col'] = new_pos[1]

                            if verbose > 1:
                                curr_msg = "Time: {:08d}; Fish: {}; health: {:3.4f}; position:[{:3d},{:3d}]".\
                                    format(curr_t + 1, curr_fish.name, updated_health, new_pos[0], new_pos[1])
                                print(curr_msg)
                                self._simulation_histories['message'] += ('\n' + curr_msg)

                        else:  # if dead
                            curr_msg = 'Time: {:08d}; Fish: {} died. Total number of movements: {}'\
                                .format(curr_t, curr_fish.name,
                                        self._simulation_histories[curr_fish.name]['total_moves'])
                            print('\n' + curr_msg)
                            self._simulation_histories['message'] += ('\n' + curr_msg)
                            dead_fish_list.append(curr_fish)
                            self._simulation_histories[curr_fish.name]['life_history'] = \
                                self._simulation_histories[curr_fish.name]['life_history'][0: curr_t + 1]
                            self._simulation_histories[curr_fish.name]['psp_waveforms'] = \
                                self._simulation_histories[curr_fish.name]['psp_waveforms'][:, 0: curr_t + 1]
                    else:
                        pass

                for dead_fish in dead_fish_list:
                    alive_fish_list.remove(dead_fish)

                curr_t += 1

            else:

                self._end_t = curr_t

                if curr_t == self.simulation_length:
                    curr_msg = "Simulation: End of Simulation. Prespecified simulation length reached.\n"

                    for sur_fish in alive_fish_list:
                        add_msg = "Survived fish: {}. Final health: {}. Total number of movements: {}".\
                            format(sur_fish.name,
                                   self._simulation_histories[sur_fish.name]['life_history'].loc[curr_t - 1, 'health'],
                                   self._simulation_histories[sur_fish.name]['total_moves'])
                        curr_msg += '\n' + add_msg
                    print(curr_msg)
                    self._simulation_histories['message'] += ('\n' + curr_msg)

                elif len(alive_fish_list) == 0:
                    self._simulation_histories['food_pos_history'] = \
                        self._simulation_histories['food_pos_history'][0: curr_t + 1]
                    curr_msg = "\nSimulation: End of Simulation. All fish are dead. " \
                               "Last simulated time point: {}".format(curr_t)
                    print(curr_msg)
                    self._simulation_histories['message'] += ('\n' + curr_msg)

            self._simulation_status = 4

        else:
            raise RuntimeError("Simulation: Cannot run simulation. Simulation not initialized properly.")

        return

    def save_log(self, log_folder, is_save_psp_waveforms=False):
        """
        save simulation results into a hdf5 file
        :param log_folder: directory path to save save_log
        :param msg: str, print out string
        :param is_save_psp_waveforms: 
        :return: None
        """

        if self._simulation_status == 4:
            save_name = 'simulation_' + datetime.datetime.now().strftime('%y%m%d_%H_%M_%S') + '.hdf5'
            if not os.path.isdir(log_folder):
                os.makedirs(log_folder)
            log_f = h5py.File(os.path.join(log_folder, save_name))
            log_f['terrain_map'] = self._terrain._terrain_map
            log_f['message'] = self._simulation_histories['message']

            end_t_dset = log_f.create_dataset('last_time_point', data=self._end_t)
            end_t_dset.attrs['description'] = 'the last time point simulated.'

            food_pos_dset = log_f.create_dataset('food_pos_history',
                                                 data=self._simulation_histories['food_pos_history'])
            food_pos_dset.attrs['data_format'] = 'time_points x food_num x food position [row, col]'

            fish_list_grp = log_f.create_group('fish_list')
            for curr_fish in self._fish_list:

                curr_fish_grp = fish_list_grp.create_group(curr_fish.name)

                # save data of current fish
                curr_fish_fish_grp = curr_fish_grp.create_group('fish')
                curr_fish.to_h5_group(curr_fish_fish_grp)
                curr_fish_fish_grp.attrs['description'] = 'Data of the little.fish.fish.Fish object. The object can ' \
                                                          'be loaded by Fish.from_h5_group() method.'

                # create group to save simulation history of current fish
                curr_fish_sim_grp = curr_fish_grp.create_group('sim_history')
                curr_sim_history = self._simulation_histories[curr_fish.name]

                # save total moves
                curr_fish_grp['total_moves'] = curr_sim_history['total_moves']

                # save action histories of every neuron in current fish
                curr_fish_ah_grp = curr_fish_sim_grp.create_group('action_histories')
                curr_fish_ah = curr_sim_history['action_histories']
                for neuron_ind, ah in curr_fish_ah.iterrows():
                    curr_fish_ah_grp['neuron_' + util.int2str(neuron_ind, 4)] = ah.iloc[0]

                if is_save_psp_waveforms:
                    # save psp waveforms of all neurons in current fish
                    curr_fish_psp_wf_dset = curr_fish_sim_grp.create_dataset('psp_waveforms',
                                                                             data=curr_sim_history['psp_waveforms'])
                    curr_fish_psp_wf_dset.attrs['data_format'] = 'neuron_ind x time_point'

                # save life history of fish
                curr_life_his = self._simulation_histories[curr_fish.name]['life_history']
                curr_pos_arr = np.array(curr_life_his.loc[:, ['pos_row', 'pos_col']])
                curr_fish_pos_dset = curr_fish_sim_grp.create_dataset('position_history', data=curr_pos_arr)
                curr_fish_pos_dset.attrs['data_format'] = 'time_point x center position [row, col]'
                curr_health_arr = np.array(curr_life_his.loc[:, 'health'])
                curr_fish_sim_grp['health'] = curr_health_arr
        else:
            raise RuntimeError("Simulation: Cannot save save_log. Simulation has not run yet.")

    def save_log_to_h5_grp(self, h5_grp, is_save_psp_waveforms=False):
        """
        save simulation results into a hdf5 group
        :param h5_grp, hdf5 group object
        :param msg: str, print out string
        :param is_save_psp_waveforms:
        :return: None
        """
        if len(self._fish_list) > 1:
            raise IOError('Simulation: Cannot save log to a hdf5 group. More than one fish in self._fish_list. '
                          'The save_log_to_h5_grp() function only designed to save log of simulation contain only '
                          'one fish.')

        if self._simulation_status == 4:

            curr_fish = self._fish_list[0]
            curr_sim_history = self._simulation_histories[curr_fish.name]

            sim_grp = h5_grp.create_group('simulation_log')
            sim_grp['terrain_map'] = self._terrain._terrain_map
            sim_grp['message'] = self._simulation_histories['message']

            sim_grp['total_moves'] = curr_sim_history['total_moves']

            food_pos_dset = sim_grp.create_dataset('food_pos_history',
                                                  data=self._simulation_histories['food_pos_history'], dtype=np.uint32,
                                                  compression='gzip')
            food_pos_dset.attrs['data_format'] = 'time_points x food_num x food position [row, col]'

            end_t_dset = sim_grp.create_dataset('last_time_point', data=self._end_t)
            end_t_dset.attrs['description'] = 'the last time point simulated.'

            # save action histories of every neuron in current fish
            curr_fish_ah_grp = sim_grp.create_group('action_histories')
            curr_fish_ah = curr_sim_history['action_histories']
            for neuron_ind, ah in curr_fish_ah.iterrows():
                curr_fish_ah_grp['neuron_' + util.int2str(neuron_ind, 4)] = ah.iloc[0]

            if is_save_psp_waveforms:
                # save psp waveforms of all neurons in current fish
                curr_fish_psp_wf_dset = sim_grp.create_dataset('psp_waveforms',
                                                              data=curr_sim_history['psp_waveforms'])
                curr_fish_psp_wf_dset.attrs['data_format'] = 'neuron_ind x time_point'

            # save life history of fish
            curr_life_his = self._simulation_histories[curr_fish.name]['life_history']

            curr_pos_arr = np.array(curr_life_his.loc[:, ['pos_row', 'pos_col']])
            curr_fish_pos_dset = sim_grp.create_dataset('position_history', data=curr_pos_arr, dtype=np.uint32,
                                                       compression='gzip')
            curr_fish_pos_dset.attrs['data_format'] = 'time_point x center position [row, col]'

            curr_health_arr = np.array(curr_life_his.loc[:, 'health'])
            sim_grp['health'] = curr_health_arr

        else:
            raise RuntimeError("Simulation: Cannot save save_log. Simulation has not run yet.")


if __name__ == '__main__':

    # -------------------------------------------------------------------------
    import littlefish.core.terrain as tr
    import littlefish.core.fish as fi
    import random
    random.seed(50)
    np.random.seed(50)
    tg = tr.TerrainGenerator()
    terrain_map = tg.generate_binary_map(sigma=3., is_plot=True)
    terrain = tr.BinaryTerrain(terrain_map)
    fish = fi.Fish()
    sim = Simulation(terrain=terrain, fish_list=[fish],
                     simulation_length=5000, food_num=5)
    sim.initiate_simulation()
    # print(sim._simulation_histories)
    # sim.generating_fish_map(time_point=0, is_plot=True)
    # sim.get_fish_health_status(0, verbose=True)
    # print(sim.is_all_fish_dead(0))
    sim.run(verbose=1)
    # -------------------------------------------------------------------------

    print 'for debugging...'