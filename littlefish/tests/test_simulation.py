# from __future__ import (absolute_import, division,
#                         print_function, unicode_literals)
# from builtins import *

import os
import random
import unittest

import h5py
import numpy as np
import littlefish.core.fish as fi
import littlefish.core.simulation as si
import littlefish.core.terrain as tr
from littlefish.brain.functional import generate_brain_from_brain_config


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

    def test_simulate_one_fish(self):
        curr_folder = os.path.dirname(os.path.realpath(__file__))

        tg = tr.TerrainGenerator(size=[64, 64], sea_portion=0.7)
        ter = tr.BinaryTerrain(tg.generate_binary_map(sigma=3.0))

        # get brain

        brain_config_path = os.path.join(
            os.path.dirname(curr_folder),
            "configs",
            "brain_config_4eyes_feedforward.yml",
        )
        brain = generate_brain_from_brain_config(brain_config_path=brain_config_path)
        # get fish
        fish = fi.Fish(
            name="test_fish",
            mother_name="from_config",
            brain=brain,
            max_health=20.0,
            health_decay_rate=0.01,
            land_penalty_rate=0.5,
            food_rate=20.0,
            move_penalty_rate=0.001,
            action_potential_penalty_rate=0.00001,
            generations=[0],
        )
        log_path = os.path.join(curr_folder, "test_log.h5")

        if os.path.isfile(log_path):
            os.remove(log_path)

        log_f = h5py.File(log_path, "a")
        fish.to_h5_group(log_f)
        log_f.close()

        si.simulate_one_fish(
            fish_path=log_path,
            simulation_length=100,
            simulation_num=1,
            terrain=ter,
            food_num=50,
            hard_thr=0,
            fish_ind=0,
            fish_num=1,
            verbose=True,
        )

        # os.remove(log_path)


if __name__ == "__main__":
    test_simulation = TestSimulation()
    # test_simulation.test_simulation()
    test_simulation.test_simulate_one_fish()
