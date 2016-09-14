# from __future__ import (absolute_import, division,
#                         print_function, unicode_literals)
# from builtins import *

import numpy as np
import matplotlib.pyplot as plt


def array_nor(arr):
    arr = arr.astype(np.float32)
    return (arr - np.amin(arr)) / (np.amax(arr) - np.amin(arr))


def discreat_crosscorrelation(ts_trigger, ts_reference, t_range=(-10., 20.), bin_width=1.):
    """
    generate crosscorrelogram between two timestamp train

    :param ts_trigger: the triggering time stamps, 1d array, monotonically increasing
    :param ts_reference: the reference time stamps, 1d array, monotonically increasing
    :param t_range: para-trigger time range, tuple with two floats
    :param bin_with: duration of a single bin
    :return: crosscorrelogram of counts, 1-d array, int
             t_axis, time axis of crosscorrelogram, 1-d array, float
    """

    if len(ts_trigger.shape) != 1:
        raise(ValueError, 'ts_trigger should be a 1-d array.')

    if len(ts_reference.shape) != 1:
        raise(ValueError, 'ts_reference should be a 1-d array.')

    t_axis = np.arange(t_range[0], t_range[1], bin_width)
    ccg = np.zeros(t_axis.shape, dtype=np.int)
    for trigger in ts_trigger:
        # get the chunk of ts_reference for this particular trigger
        ref_ts_chunk = ts_reference[np.logical_and(ts_reference >= trigger + t_axis[0],
                                                   ts_reference < trigger + t_axis[-1]+bin_width)]

        # calculate relative timing around the trigger
        ref_ts_chunk = ref_ts_chunk - (float(trigger) + t_range[0])

        for ref_ts in ref_ts_chunk:
            ccg[int(ref_ts//bin_width)] += 1

    return ccg, t_axis


def get_random_number(distribution, shape):
    """
    get a random number from given distribution
    :param distribution: tuple in the format, (distribution type, parameter1, parameter2, ...)
                        supported: ('flat', mean, range)
                                   ('gaussian, mean, sigma)
                                   ('exponential', mean)
    :param shape: output shape
    :return: a random number
    """
    if distribution is None:
        output = np.zeros(shape, dtype=np.float64)
    elif distribution[0] == 'flat':
        output = np.random.rand(*shape) * float(distribution[2]) - 0.5 * distribution[2] + distribution[1]
    elif distribution[0] == 'gaussian':
        output = np.random.randn(*shape) * float(distribution[2]) + float(distribution[1])
    elif distribution[0] == 'exponential':
        if distribution[1] <= 0:
            raise(ValueError, 'The mean of the exponential distribution should be larger than 0!')
        output = np.random.exponential(float(distribution[1]), shape)
    else:
        raise (LookupError, 'the first element of "noise" should be "gaussian", "flat" or "exponential"!')


if __name__ == '__main__':

    #==================================================
    ts_trigger = np.arange(5)
    ts_reference = np.arange(5) + 0.1001

    ccg, t = discreat_crosscorrelation(ts_trigger, ts_reference, t_range=(-0.2, 0.5), bin_width=0.1)

    print(t)
    print(ccg)

    plt.plot(t, ccg)
    plt.show()
    # ==================================================

    print('for debug ...')

