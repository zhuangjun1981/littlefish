# from __future__ import (absolute_import, division,
#                         print_function, unicode_literals)
# from builtins import *

import os
import sys
import numpy as np
import littlefish.utilities as util
import unittest


class TestUtilities(unittest.TestCase):

    def setup(self):
        pass

    def test_array_nor(self):
        arr = np.array([1, 2, 3])
        assert(np.array_equal(util.array_nor(arr), np.array([0, 0.5, 1])))

    def test_discreat_crosscorrelation(self):
        ts_trigger = np.arange(5)
        ts_reference = np.arange(5) + 0.1001
        ccg, t = util.discreat_crosscorrelation(ts_trigger, ts_reference, t_range=(-0.2, 0.5), bin_width=0.1)
        assert(np.array_equal(ccg, [0, 0, 0, 5, 0, 0, 0]))


if __name__ == '__main__':
    tu = TestUtilities()
    tu.test_array_nor()
    tu.test_discreat_crosscorrelation()

