from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import numpy as np

def array_nor(arr):
    arr = arr.astype(np.float32)
    return (arr - np.amin(arr)) / (np.amax(arr) - np.amin(arr))
