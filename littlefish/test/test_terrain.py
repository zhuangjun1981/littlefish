# from __future__ import (absolute_import, division,
#                         print_function, unicode_literals)
# from builtins import *

import os
import sys
package_path, _ = os.path.split(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(package_path)
import terrain.terrain_2d as ter
import numpy as np
import unittest
import matplotlib.pyplot as plt


class TestUtilities(unittest.TestCase):

    def setup(self):
        pass

    def test_update_food_map(self):
        food_map = np.zeros((5, 5), dtype=np.uint8)
        terrain_map = np.zeros((5, 5), dtype=np.uint8)
        terrain_map[(2, 0, 1, 4, 4), (3, 4, 2, 3, 1)] = 1
        terrain = ter.BinaryTerrain(terrain_map)
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
        assert (np.max(np.logical_and(terrain_map, food_map)) == 0)
        assert (np.sum(food_map.flat) == 5)
        assert (food_pos_array.shape == (5, 2))

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
        assert (np.max(np.logical_and(terrain_map, food_map)) == 0)
        assert (np.sum(food_map.flat) == 3)
        assert (food_pos_array.shape == (3, 2))

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
        assert (np.max(np.logical_and(terrain_map, food_map)) == 0)
        assert (np.sum(food_map.flat) == 5)
        assert (food_pos_array.shape == (5, 2))

        food_pos_list = terrain.update_food_map(food_num=5, food_map=food_map)
        food_pos_array = np.array([np.array(pos) for pos in food_pos_list])
        assert (np.max(np.logical_and(terrain_map, food_map)) == 0)
        assert (np.sum(food_map.flat) == 5)
        assert (food_pos_array.shape == (5, 2))



if __name__ == '__main__':
    tu = TestUtilities()
    tu.test_update_food_map()
