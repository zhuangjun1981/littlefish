from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import littlefish.utilities as util
import numpy as np

def test_array_nor():
    arr = np.array([1, 2, 3])
    assert(np.array_equal(util.array_nor(arr), np.array([0, 0.5, 1])))

def run():
    test_array_nor()

if __name__ == '__main__':
    run()

