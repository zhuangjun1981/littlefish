# from __future__ import (absolute_import, division,
#                         print_function, unicode_literals)
# from builtins import *

import os
import sys
package_path, _ = os.path.split(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(package_path)

import fish.brain as brain
import numpy as np

SIMULATION_LENGTH = 50


def test_connection_psp():
    connection = brain.Connection(amplitude=10, latency=5, rise_time=5, decay_time=10)
    assert(np.array_equal(connection.get_psp(), np.array([0.,0.,0.,0.,0.,2.,4.,6.,8.,10.,9.,8.,7.,6.,5.,4.,3.,2.,1.,0.])))


def test_connection_act():
    connection = brain.Connection(amplitude=10, latency=5, rise_time=5, decay_time=10)
    postsynaptic_input = np.zeros(SIMULATION_LENGTH)
    connection.act(2, postsynaptic_input)
    assert(np.array_equal(postsynaptic_input,
                          np.array([0.,0.,0.,0.,0.,0.,0.,2.,4.,6.,8.,10.,9.,8.,7.,6.,5.,4.,3.,2.,1.,0.,0.,0.,0.,0.,0.,
                                    0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])))
    connection.act(4, postsynaptic_input)
    assert (np.array_equal(postsynaptic_input,
                           np.array([0.,0.,0.,0.,0.,0.,0.,2.,4.,8.,12.,16.,17.,18.,16.,14.,12.,10.,8.,6.,4.,2.,1.,0.,0.,
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])))
    connection.act(40, postsynaptic_input)
    assert (np.array_equal(postsynaptic_input,
                           np.array([0.,0.,0.,0.,0.,0.,0.,2.,4.,8.,12.,16.,17.,18.,16.,14.,12.,10.,8.,6.,4.,2.,1.,0.,0.,
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,2.,4.,6.,8.,10.])))
    connection.act(50, postsynaptic_input)
    assert (np.array_equal(postsynaptic_input,
                           np.array([0.,0.,0.,0.,0.,0.,0.,2.,4.,8.,12.,16.,17.,18.,16.,14.,12.,10.,8.,6.,4.,2.,1.,0.,0.,
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,2.,4.,6.,8.,10.])))


def test_eye_get_input_pixels():

    map = np.zeros((5, 5), dtype=np.uint8)
    map[3, 3] = 1
    map[2, 2] = 1
    map[2, 1] = 1

    eye = brain.Eye(position=(2, 3), direction='south')
    assert(np.array_equal(eye._get_input_pixels(map), [0, 1, 0]))
    eye = brain.Eye(position=(2, 3), direction='southeast')
    assert (np.array_equal(eye._get_input_pixels(map), [1, 0, 0]))
    eye = brain.Eye(position=(3, 2), direction='east')
    assert (np.array_equal(eye._get_input_pixels(map), [0, 1, 0]))
    eye = brain.Eye(position=(3, 2), direction='northeast')
    assert (np.array_equal(eye._get_input_pixels(map), [1, 0, 1]))
    eye = brain.Eye(position=(3, 2), direction='north')
    assert (np.array_equal(eye._get_input_pixels(map), [0, 1, 1]))
    eye = brain.Eye(position=(3, 2), direction='northwest')
    assert (np.array_equal(eye._get_input_pixels(map), [1, 1, 0]))
    eye = brain.Eye(position=(3, 2), direction='west')
    assert (np.array_equal(eye._get_input_pixels(map), [1, 0, 0]))
    eye = brain.Eye(position=(3, 2), direction='southwest')
    assert (np.array_equal(eye._get_input_pixels(map), [0, 0, 0]))
    eye = brain.Eye(position=(0, 0), direction='north')
    assert (np.array_equal(eye._get_input_pixels(map), [1, 1, 1]))
    eye = brain.Eye(position=(0, 0), direction='northeast')
    assert (np.array_equal(eye._get_input_pixels(map), [0, 1, 1]))
    eye = brain.Eye(position=(0, 0), direction='east')
    assert (np.array_equal(eye._get_input_pixels(map), [0, 0, 1]))


def run():
    test_connection_psp()
    test_connection_act()
    test_eye_get_input_pixels()

if __name__ == '__main__':
    run()