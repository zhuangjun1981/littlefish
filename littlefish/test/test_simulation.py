# from __future__ import (absolute_import, division,
#                         print_function, unicode_literals)
# from builtins import *

import os
import random
import numpy as np
import littlefish.terrain.terrain_2d as tr
import littlefish.fish.fish as fi
import littlefish.simulation.simulation as si
import unittest

class TestSimulation(unittest.TestCase):

    def setup(self):
        pass

    def test_simulation(self):
        random.seed(111)
        simulation_length = 50

        terrain_map = np.array([[0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1],
                                [0, 0, 1, 0, 0],
                                [0, 1, 0, 1, 0]], dtype=np.uint8)
        terrain = tr.BinaryTerrain(terrain_map)
        fish = fi.Fish()
        simulation = si.Simulation(terrain=terrain, fish_list=[fish],
                                   simulation_length=simulation_length, food_num=5)
        simulation.initiate_simulation()
        simulation.run(verbose=2)