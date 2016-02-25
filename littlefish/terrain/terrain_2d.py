# the 2-d terrain generator

import littlefish.utilities as util
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ni

plt.ioff()

class TerrainBase(object):
    '''
    terrain object, a square matrix containing altitude
    '''

    def __init__(self,size=(256,256),sea_level=0.5):
        '''
        :param size: size of map (height, width)
        :param sea_level: a simple threshold
        '''
        self._size = size
        self._sea_level = sea_level

    def get_size(self):
        '''
        :return: size of map (height, width)
        '''
        return self._size

    def get_sea_level(self):
        '''
        :return: sea level
        '''
        return self._sea_level

    def generate_float_map(self, sigma=0.):
        '''
        generate a map with floating point with elevation [0., 1.]
        :param sigma: filter sigma to filter the map
        :return:
        '''
        float_map = np.random.random(self._size)
        float_map = ni.filters.gaussian_filter(float_map,sigma)
        float_map = util.array_nor(float_map)

        return float_map

    def generate_binary_map(self, sigma=0., is_plot=False):
        '''
        :param sigma: filter sigma to filter the map
        :param is_plot: if True, pop a plot of binary map
        :return: a binary map with defined size, 0 means under water. 1 means above water
        '''
        float_map = self.generate_float_map(sigma)
        binary_map = np.zeros(float_map.shape, dtype = np.bool)
        binary_map[float_map > self._sea_level] = 1
        print binary_map.dtype

        if is_plot:
            f = plt.figure(figsize=(20, 8))
            ax1 = f.add_subplot(121)
            ax1.imshow(float_map, vmin=0, vmax=1, cmap='gray', interpolation='nearest')
            ax2 = f.add_subplot(122)
            ax2.imshow(binary_map, vmin=0, vmax=1, cmap='gray', interpolation='nearest')
            plt.show()

        return binary_map.astype(np.bool)



if __name__ == '__main__':

    #=============================================================
    terr = TerrainBase(sea_level=0.6)
    _ = terr.generate_binary_map(sigma=5., is_plot=True)
    #=============================================================
