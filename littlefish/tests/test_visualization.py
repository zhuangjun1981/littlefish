# from __future__ import (absolute_import, division,
#                         print_function, unicode_literals)
# from builtins import *

import os
import unittest

import littlefish.core.plotting as vis


class TestUtilities(unittest.TestCase):
    def setup(self):
        pass

    def test_plot_brain(self):
        import littlefish.core.fish as fi
        import h5py

        curr_folder = os.path.dirname(os.path.realpath(__file__))
        log_path = os.path.join(curr_folder, "real_simulation_log.hdf5")
        log_f = h5py.File(log_path, "r")
        input_brain = fi.Brain.from_h5_group(log_f["fish/brain"])
        log_f.close()
        vis.plot_brain(input_brain)
