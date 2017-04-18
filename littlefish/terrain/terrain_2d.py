# the 2-d terrain generator

# from __future__ import (absolute_import, division,
#                         print_function, unicode_literals)
# from builtins import *

import littlefish.utilities as util
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ni
import random




plt.ioff()

class TerrainGenerator(object):
    '''
    terrain object, a square matrix containing altitude
    '''

    def __init__(self,size=(256,256),sea_level=0.6):
        '''
        :param size: size of world_map (height, width)
        :param sea_level: a simple threshold
        '''
        self._size = size
        self._sea_level = sea_level

    def get_size(self):
        '''
        :return: size of world_map (height, width)
        '''
        return self._size

    def get_sea_level(self):
        '''
        :return: sea level
        '''
        return self._sea_level

    def generate_float_map(self, sigma=0.):
        '''
        generate a world_map with floating point with elevation [0., 1.]
        :param sigma: filter sigma to filter the world_map
        :return:
        '''
        float_map = np.random.random(self._size)
        float_map = ni.filters.gaussian_filter(float_map,sigma)
        float_map = util.array_nor(float_map)

        return float_map

    def generate_binary_map(self, sigma=0.5, is_plot=False):
        '''
        :param sigma: filter sigma to filter the world_map
        :param is_plot: if True, pop a plot of binary world_map
        :return: a binary world_map with defined size, 0 means under water. 1 means above water
        '''
        float_map = self.generate_float_map(sigma)
        binary_map = np.zeros(float_map.shape, dtype = np.bool)
        binary_map[float_map > self._sea_level] = 1
        # print(binary_map.dtype)

        if is_plot:
            f = plt.figure(figsize=(20, 8))
            ax1 = f.add_subplot(121)
            ax1.imshow(float_map, vmin=0, vmax=1, cmap='gray', interpolation='nearest')
            ax2 = f.add_subplot(122)
            ax2.imshow(binary_map, vmin=0, vmax=1, cmap='gray', interpolation='nearest')
            plt.show()

        return binary_map.astype(np.uint8)


class BinaryTerrain(object):
    """
    2 dimensional binary terrain object.
    """

    def __init__(self, input_array):
        """

        :param input_array:
        """

        if util.check_binary_2d_array(input_array):
            self._terrain_map = input_array
        else:
            raise ValueError('BinaryTerrain: input array should be binary 2d numpy array, with dtype np.int.')

        self._curr_food_positions = []
        self._food_position_history = []

    def get_terrain_shape(self):
        return self._terrain_map.shape

    def get_terrain_map(self):
        return self._terrain_map

    def generate_fish_starting_position(self):
        """
        :return: a possible position for a 3x3 fish, where its body will not cover 1s in self._terrain_map
        """
        dilated_terrain = ni.binary_dilation(self._terrain_map, structure=[[1,1,1], [1,1,1], [1,1,1]])
        possible_positions = zip(*np.where(dilated_terrain == 0))
        return list(random.choice(possible_positions))

    def generate_next_food_position(self):
        # todo: finish this method
        pass

    def generate_curr_food_map(self):
        # tood: finish this method
        pass

    def _update_food_positions(self):
        # todo: finish this method
        pass

    def plot_terrain(self, plot_axis=None):
        # todo: finish this method
        pass


if __name__ == '__main__':

    #=============================================================
    terrain_generator = TerrainGenerator(sea_level=0.6)
    terrain_map = terrain_generator.generate_binary_map(sigma=5., is_plot=True)
    binary_terrain = BinaryTerrain(terrain_map)
    fish_pos = binary_terrain.generate_fish_starting_position()
    fish_map = np.zeros(binary_terrain.get_terrain_shape(), dtype=np.uint8)
    fish_map[fish_pos[0] - 1: fish_pos[0] + 2, fish_pos[1] - 1: fish_pos[1] + 2] = 1
    f = plt.figure(figsize=(10, 10))
    ax = f.add_subplot(111)
    ax.imshow(binary_terrain.get_terrain_map(), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    util.plot_mask(fish_map, plot_axis=ax, color='#ff0000')
    plt.show()
    #=============================================================


    # =============================================================

    # =============================================================

    print 'for debug'
