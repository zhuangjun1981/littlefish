# from __future__ import (absolute_import, division,
#                         print_function, unicode_literals)
# from builtins import *

import unittest

import littlefish.core.terrain as tr
import numpy as np


class TestTerrain(unittest.TestCase):
    def setup(self):
        pass

    def test_terrain_generator(self):
        tr_gen = tr.TerrainGenerator(size=(64, 64), sea_portion=0.5)
        bm = tr_gen.generate_binary_map(sigma=3, step_size=0.01, is_plot=False)
        assert (256.0 - np.sum(bm[:])) / 256.0 < 0.5

    def test_update_food_map(self):
        food_map = np.zeros((5, 5), dtype=int)
        terrain_map = np.zeros((5, 5), dtype=int)
        terrain_map[(2, 0, 1, 4, 4), (3, 4, 2, 3, 1)] = 1
        terrain = tr.BinaryTerrain(terrain_map)
        food_pos_list = terrain.update_food_map(food_num=5, food_map=food_map)
        # f = plt.figure(figsize=(10, 4))
        # ax1 = f.add_subplot(121)
        # ax1.imshow(terrain_map, interpolation='nearest')
        # ax1.set_title('terrain map')
        # ax2 = f.add_subplot(122)
        # ax2.imshow(food_map, interpolation='nearest')
        # ax2.set_title('food map')
        # plt.show()
        food_pos_array = np.array([np.array(pos) for pos in food_pos_list])
        assert np.max(np.logical_and(terrain_map, food_map)) == 0
        assert np.sum(food_map.flat) == 5
        assert food_pos_array.shape == (5, 2)

        food_pos_list = terrain.update_food_map(food_num=3, food_map=food_map)
        # f = plt.figure(figsize=(10, 4))
        # ax1 = f.add_subplot(121)
        # ax1.imshow(terrain_map, interpolation='nearest')
        # ax1.set_title('terrain map')
        # ax2 = f.add_subplot(122)
        # ax2.imshow(food_map, interpolation='nearest')
        # ax2.set_title('food map')
        # plt.show()
        food_pos_array = np.array([np.array(pos) for pos in food_pos_list])
        assert np.max(np.logical_and(terrain_map, food_map)) == 0
        assert np.sum(food_map.flat) == 3
        assert food_pos_array.shape == (3, 2)

        food_pos_list = terrain.update_food_map(food_num=5, food_map=food_map)
        # f = plt.figure(figsize=(10, 4))
        # ax1 = f.add_subplot(121)
        # ax1.imshow(terrain_map, interpolation='nearest')
        # ax1.set_title('terrain map')
        # ax2 = f.add_subplot(122)
        # ax2.imshow(food_map, interpolation='nearest')
        # ax2.set_title('food map')
        # plt.show()
        food_pos_array = np.array([np.array(pos) for pos in food_pos_list])
        assert np.max(np.logical_and(terrain_map, food_map)) == 0
        assert np.sum(food_map.flat) == 5
        assert food_pos_array.shape == (5, 2)

        food_pos_list = terrain.update_food_map(food_num=5, food_map=food_map)
        food_pos_array = np.array([np.array(pos) for pos in food_pos_list])
        assert np.max(np.logical_and(terrain_map, food_map)) == 0
        assert np.sum(food_map.flat) == 5
        assert food_pos_array.shape == (5, 2)

    def test_generate_fish_starting_position(self):
        ter = tr.BinaryTerrain(
            np.array(
                [
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1],
                    [0, 0, 1, 0, 0],
                    [0, 1, 0, 1, 0],
                ],
                dtype=int,
            )
        )
        pos = ter.generate_fish_starting_position(2)
        assert pos[0] == (1, 1)
        assert pos[1] == (1, 1)


if __name__ == "__main__":
    tu = TestTerrain()
    # tu.test_terrain_generator()
    tu.test_update_food_map()
    # tu.test_generate_fish_starting_position()
