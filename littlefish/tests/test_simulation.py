# from __future__ import (absolute_import, division,
#                         print_function, unicode_literals)
# from builtins import *

import os
import random
import unittest

import h5py
import littlefish.core.fish as fi
import littlefish.core.simulation as si
import littlefish.core.terrain as tr
import numpy as np


class TestSimulation(unittest.TestCase):
    def test_simulation(self):
        curr_folder = os.path.dirname(os.path.realpath(__file__))

        # clean saved test simulation logs, if there is any
        log_path = os.path.join(curr_folder, "simulation_log.hdf5")
        if os.path.isfile(log_path):
            os.remove(log_path)

        random.seed(111)
        np.random.seed(50)
        simulation_length = 5
        terrain_map = np.array(
            [
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
            ],
            dtype=int,
        )
        terrain = tr.BinaryTerrain(terrain_map)
        fish = fi.Fish()
        simulation = si.Simulation(
            terrain=terrain,
            fish_list=[fish],
            simulation_length=simulation_length,
            food_num=2,
        )
        simulation.initiate_simulation()
        simulation.run(verbose=2)

        # print(simulation.simulation_cache)
        # print(vars(simulation).keys())
        # print(simulation.simulation_cache.keys())
        # print(simulation.simulation_cache["message"])
        # print(simulation.simulation_cache["food_pos_history"])

        sim_log_f = h5py.File(log_path, "a")
        fish.to_h5_group(sim_log_f, should_save_cache=False)
        simulation.to_h5_group(sim_log_f, should_save_psp_waveforms=True)
        sim_name = [k for k in sim_log_f.keys() if k.startswith("simulation_")][0]

        assert np.array_equal(sim_log_f[sim_name]["terrain_map"][()], terrain_map)
        assert np.array_equal(
            sim_log_f[sim_name]["simulation_cache/food_pos_history"][()],
            np.array(
                [
                    [[3, 3], [1, 4]],
                    [[1, 4], [3, 3]],
                    [[1, 4], [3, 3]],
                    [[1, 4], [3, 3]],
                    [[1, 4], [3, 3]],
                ]
            ),
        )
        sim_log_f.close()
        os.remove(log_path)


if __name__ == "__main__":
    test_simulation = TestSimulation()
    test_simulation.test_simulation()
