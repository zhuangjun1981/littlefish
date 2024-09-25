# from __future__ import (absolute_import, division,
#                         print_function, unicode_literals)
# from builtins import *

import unittest
import littlefish.core.utilities as util
import numpy as np


class TestUtilities(unittest.TestCase):
    def setup(self):
        pass

    def test_array_nor(self):
        arr = np.array([1, 2, 3])
        assert np.array_equal(util.array_nor(arr), np.array([0, 0.5, 1]))

    def test_discreat_crosscorrelation(self):
        ts_trigger = np.arange(5)
        ts_reference = np.arange(5) + 0.1001
        ccg, t = util.discreat_crosscorrelation(
            ts_trigger, ts_reference, t_range=(-0.2, 0.5), bin_width=0.1
        )
        assert np.array_equal(ccg, [0, 0, 0, 5, 0, 0, 0])

    def test_get_color_str(self):
        assert util.get_color_str(0, 0, 0) == "#000000"
        assert util.get_color_str(255, 0, 0) == "#ff0000"
        assert util.get_color_str(0, 255, 0) == "#00ff00"
        assert util.get_color_str(0, 0, 255) == "#0000ff"

    def test_normalized_to_range(self):
        assert (
            round(
                util.normalized_to_range(
                    var=0.5, input_range=(0.0, 1.0), output_range=(0.0, 2.0)
                ),
                10,
            )
            == 1.0
        )
        assert (
            round(
                util.normalized_to_range(
                    var=0.8, input_range=(0.0, 1.0), output_range=(0.0, 2.0)
                ),
                10,
            )
            == 1.6
        )
        assert (
            round(
                util.normalized_to_range(
                    var=0.8, input_range=(0.0, 1.0), output_range=(1.0, 4.0)
                ),
                10,
            )
            == 3.4
        )

    # def test_value_2_rgb(self):
    #     import matplotlib.pyplot as plt
    #     import matplotlib
    #     plt.ioff()
    #     cmap = 'RdBu_r'
    #     norm = matplotlib.colors.Normalize(vmin=0., vmax=99.)
    #     f = plt.figure(figsize=(8, 4))
    #     ax1 = f.add_subplot(121)
    #     ax2 = f.add_subplot(122)
    #     plot_arr = np.arange(100).reshape((10, 10))
    #     ax1.imshow(plot_arr, vmin=0., vmax=99., cmap=cmap, interpolation='nearest')
    #     for i in range(10):
    #         for j in range(10):
    #             # curr_c = util.normalized_to_range(plot_arr[i, j], (0., 99.), (0., 1.))
    #             curr_c = norm(plot_arr[i, j])
    #             curr_cstr = util.value_2_rgb(curr_c, cmap)
    #             ax2.plot(j, i, 'o', mfc=curr_cstr, lw=0, mec='none', ms=20)
    #     ax2.set_xlim([-0.5, 9.5])
    #     ax2.set_ylim([9.5, -0.5])
    #     plt.show()

    def test_distrube_number(self):
        buckets = util.distrube_number(
            possibilities=np.arange(1, 6), population_size=1500
        )
        # print ('\n{}'.format(buckets))
        assert np.sum(buckets) == 1500


if __name__ == "__main__":
    tu = TestUtilities()
    tu.test_array_nor()
    tu.test_discreat_crosscorrelation()
