# the 2-d terrain generator

# from __future__ import (absolute_import, division,
#                         print_function, unicode_literals)
# from builtins import *

import littlefish.core.utilities as util
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ni

plt.ioff()


class TerrainGenerator(object):
    """
    terrain object, a square matrix containing altitude
    """

    def __init__(self, size=(128, 128), sea_portion=0.6):
        """

        :param size: size of world_map (height, width)
        :param sea_portion: float, (0, 1), the portion of sea in the generated maps
        """

        self.size = size

        if sea_portion < 0.0 or sea_portion > 1.0:
            raise ValueError(
                "TerrainGenerator: sea_portion should be larger than 0. and smaller than"
                "1."
            )

        self.sea_portion = sea_portion

    def generate_float_map(self, sigma: float = 0.0):
        """
        generate a world_map with floating point with elevation [0., 1.]

        :param sigma: filter sigma to filter the world_map
        :return:
        """

        float_map = np.random.random(self.size)
        float_map = ni.filters.gaussian_filter(float_map, sigma)
        float_map = util.array_nor(float_map)

        return float_map

    def generate_binary_map(
        self, sigma: float = 0.5, step_size: float = 0.001, is_plot: bool = False
    ):
        """
        generate binary map, 1: land, 0: sea, with the proportion of sea is roughly same as
        self.sea_portion

        :param sigma: positive float, filter sigma to filter the world_map
        :param step_size: positive float, (0, 1), step size to try different sea level to reach
            self.sea_portion
        :param is_plot: if True, pop a plot of binary world_map
        :return: a binary world_map with defined size, 0 means under water. 1 means above water
        """

        float_map = self.generate_float_map(sigma=sigma)
        binary_map = np.zeros(float_map.shape, dtype=np.bool)

        if self.sea_portion == 0:
            binary_map = np.ones(self.size, dtype=int)
        elif self.sea_portion == 1:
            binary_map = np.zeros(self.size, dtype=int)
        else:
            total_area = float(self.size[0] * self.size[1])

            sea_level = step_size
            binary_map[float_map >= sea_level] = 1
            curr_sea_portion = 1 - (np.sum(binary_map[:]) / total_area)

            while curr_sea_portion < self.sea_portion:
                sea_level += step_size
                binary_map[:] = 0
                binary_map[float_map >= sea_level] = 1
                curr_sea_portion = 1 - (np.sum(binary_map[:]) / total_area)

        if is_plot:
            f = plt.figure(figsize=(20, 8))
            ax1 = f.add_subplot(121)
            ax1.imshow(float_map, vmin=0, vmax=1, cmap="gray", interpolation="nearest")
            ax2 = f.add_subplot(122)
            ax2.imshow(binary_map, vmin=0, vmax=1, cmap="gray", interpolation="nearest")
            ax2.set_title(
                "sea level: {}; sea portion: {}".format(
                    sea_level, np.sum(binary_map.flat) / total_area
                )
            )
            plt.show()

        return binary_map.astype(int)


class BinaryTerrain(object):
    """
    2 dimensional binary terrain object.
    """

    def __init__(self, input_array: np.ndarray):
        """

        :param input_array:
        """

        if util.check_binary_2d_array(input_array):
            self.terrain_map = input_array
        else:
            raise ValueError(
                "BinaryTerrain: input array should be binary 2d numpy array, with _dtype integer."
            )

    def generate_fish_starting_position(self, fish_num: int = 1):
        """
        return randomized fish starting position. Note: fishes can be completely or partially overlapping

        :param fish_num: positive integer, number of fish positions to return
        :return: list of tuple, each tuple contains 2 positive integers (row, col) of a random position for a 3x3 fish,
            where its body will not cover 1s in self.terrain_map
        """

        position_maps = np.array(self.terrain_map)
        position_maps = ni.binary_dilation(
            position_maps, structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        )
        position_maps[:, 0] = 1
        position_maps[:, -1] = 1
        position_maps[0, :] = 1
        position_maps[-1, :] = 1
        possible_positions = np.array(list(zip(*np.where(position_maps == 0))))
        fish_positions = possible_positions[
            np.random.choice(
                range(len(possible_positions)), size=fish_num, replace=True
            )
        ]
        return [tuple(p) for p in fish_positions]

    def update_food_map(self, food_num: int, food_map: np.ndarray):
        """
        update the input food_map and food_pos_array, to generate new food map and food_pos_array so that the terrain
        contains food_num of food pixels (each food only occupies one pixel). If the number of food is less than
        food_num, new food will be added, if the number of food is more than food_num, extra food will be removed.

        :param food_map: 2-d binary array, 0: non-food, 1: food
        :param food_num: positive integer, number of food pixels after update
        :return: food_map: updated food map
        :return: food_positions: 2d array, food_num x 2, _dtype: uint16, array of food pixel coordinates,
            each row is a food position, columns: [row, col]
        """

        curr_food_pos_list = np.where(food_map == 1)
        curr_food_pos_list = list(zip(*curr_food_pos_list))

        if len(curr_food_pos_list) == food_num:  # current number of food equal food_num
            food_pos_list = curr_food_pos_list

        elif (
            len(curr_food_pos_list) > food_num
        ):  # curren number of food more than food_num
            # get food positions to retain
            food_retain_index = np.random.choice(
                range(len(curr_food_pos_list)), size=food_num, replace=False
            )
            food_pos_list = [curr_food_pos_list[ind] for ind in food_retain_index]

            # get food positions to remove
            food_remove_index = np.array(
                list(set(range(len(curr_food_pos_list))) - set(food_retain_index))
            )
            food_remove_list = [curr_food_pos_list[ind] for ind in food_remove_index]

            # update food_map
            food_map[
                [pos[0] for pos in food_remove_list],
                [pos[1] for pos in food_remove_list],
            ] = 0

        elif (
            len(curr_food_pos_list) < food_num
        ):  # current number of food less than food_num
            # get positions to add food
            possible_positions = list(
                zip(*np.where(np.logical_or(self.terrain_map, food_map) == 0))
            )
            food_add_index = np.random.choice(
                range(len(possible_positions)),
                size=food_num - len(curr_food_pos_list),
                replace=False,
            )
            food_add_list = [possible_positions[ind] for ind in food_add_index]

            # update food_pos_array
            food_pos_list = curr_food_pos_list + food_add_list

            # update food_map
            food_map[
                [pos[0] for pos in food_add_list], [pos[1] for pos in food_add_list]
            ] = 1

        else:
            raise ValueError(
                "Terrain: cannot update food map. Do not understand current food position list."
            )

        food_positions = np.array(food_pos_list)
        return food_positions

    def plot_terrain(self, plot_axis=None):
        # todo: finish this method
        pass

    def get_sea_portion(self):
        return float(np.sum(self.terrain_map.flat)) / float(len(self.terrain_map.flat))


if __name__ == "__main__":
    # =============================================================
    # terrain_generator = TerrainGenerator(sea_level=0.6)
    # terrain_map = terrain_generator.generate_binary_map(sigma=5., is_plot=True)
    # binary_terrain = BinaryTerrain(terrain_map)
    # fish_poss = binary_terrain.generate_fish_starting_position(5)
    # fish_map = np.zeros(binary_terrain.get_terrain_shape(), _dtype=int)
    #
    # for fish_pos in fish_poss:
    #     fish_map[fish_pos[0] - 1: fish_pos[0] + 2, fish_pos[1] - 1: fish_pos[1] + 2] = 1
    #
    # f = plt.figure(figsize=(10, 10))
    # ax = f.add_subplot(111)
    # ax.imshow(binary_terrain.get_terrain_map(), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    # util.plot_mask(fish_map, plot_axis=ax, color='#ff0000')
    # plt.show()
    # =============================================================

    # =============================================================
    # food_map = np.zeros((5, 5), _dtype=int)
    # terrain_map = np.zeros((5, 5), _dtype=int)
    # terrain_map[(2, 0, 1, 4, 4), (3, 4, 2, 3, 1)] = 1
    # terrain = BinaryTerrain(terrain_map)
    # food_pos_list = terrain.update_food_map(food_num=5, food_map=food_map)
    # f = plt.figure(figsize=(10, 4))
    # ax1 = f.add_subplot(121)
    # ax1.imshow(terrain_map, interpolation='nearest')
    # ax1.set_title('terrain map')
    # ax2 = f.add_subplot(122)
    # ax2.imshow(food_map, interpolation='nearest')
    # ax2.set_title('food map')
    # plt.show()
    # food_pos_array = np.array([np.array(pos) for pos in food_pos_list])
    # assert (np.max(np.logical_and(terrain_map, food_map)) == 0)
    # assert (np.sum(food_map.flat) == 5)
    # assert (food_pos_array.shape == (5, 2))
    #
    # food_pos_list = terrain.update_food_map(food_num=3, food_map=food_map)
    # f = plt.figure(figsize=(10, 4))
    # ax1 = f.add_subplot(121)
    # ax1.imshow(terrain_map, interpolation='nearest')
    # ax1.set_title('terrain map')
    # ax2 = f.add_subplot(122)
    # ax2.imshow(food_map, interpolation='nearest')
    # ax2.set_title('food map')
    # plt.show()
    # food_pos_array = np.array([np.array(pos) for pos in food_pos_list])
    # assert (np.max(np.logical_and(terrain_map, food_map)) == 0)
    # assert (np.sum(food_map.flat) == 3)
    # assert (food_pos_array.shape == (3, 2))
    #
    # food_pos_list = terrain.update_food_map(food_num=5, food_map=food_map)
    # f = plt.figure(figsize=(10, 4))
    # ax1 = f.add_subplot(121)
    # ax1.imshow(terrain_map, interpolation='nearest')
    # ax1.set_title('terrain map')
    # ax2 = f.add_subplot(122)
    # ax2.imshow(food_map, interpolation='nearest')
    # ax2.set_title('food map')
    # plt.show()
    # food_pos_array = np.array([np.array(pos) for pos in food_pos_list])
    # assert (np.max(np.logical_and(terrain_map, food_map)) == 0)
    # assert (np.sum(food_map.flat) == 5)
    # assert (food_pos_array.shape == (5, 2))
    #
    # food_pos_list = terrain.update_food_map(food_num=5, food_map=food_map)
    # food_pos_array = np.array([np.array(pos) for pos in food_pos_list])
    # assert (np.max(np.logical_and(terrain_map, food_map)) == 0)
    # assert (np.sum(food_map.flat) == 5)
    # assert (food_pos_array.shape == (5, 2))
    # =============================================================

    print("for debug")
