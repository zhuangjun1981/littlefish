# from __future__ import (absolute_import, division,
#                         print_function, unicode_literals)
# from builtins import *

import os
import h5py
import random
import numpy as np
import littlefish.terrain.terrain_2d as tr
import littlefish.fish.fish as fi
import littlefish.simulation.simulation as si
import unittest

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

class TestSimulation(unittest.TestCase):

    def test_simulation(self):

        # clean saved test simulation logs, if there is any
        sim_his_fns = [f for f in os.listdir(curr_folder) if f[0: 11] == 'simulation_' and f[-5:] == '.hdf5']
        for sim_his_fn in sim_his_fns:
            os.remove(sim_his_fn)

        random.seed(111)
        np.random.seed(50)
        simulation_length = 5
        terrain_map = np.array([[0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1],
                                [0, 0, 1, 0, 0],
                                [0, 1, 0, 1, 0]], dtype=np.uint8)
        terrain = tr.BinaryTerrain(terrain_map)
        fish = fi.Fish()
        simulation = si.Simulation(terrain=terrain, fish_list=[fish],
                                   simulation_length=simulation_length, food_num=2)
        simulation.initiate_simulation()
        simulation.run(verbose=2)

        simulation.save_log(curr_folder)
        sim_his_fn = [f for f in os.listdir(curr_folder) if f[0: 11] == 'simulation_' and f[-5:] == '.hdf5'][0]
        sim_his_f = h5py.File(sim_his_fn, 'r')
        assert (np.array_equal(sim_his_f['terrain_map'].value, terrain_map))
        assert (np.array_equal(sim_his_f['food_pos_history'].value, np.array([[[3, 3], [1, 4]],
                                                                              [[1, 4], [3, 3]],
                                                                              [[1, 4], [3, 3]],
                                                                              [[1, 4], [3, 3]],
                                                                              [[1, 4], [3, 3]]])))
        sim_his_f.close()
        os.remove(sim_his_fn)

