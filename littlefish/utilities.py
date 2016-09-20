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


def int2str(num,length=None):
    '''
    generate a string representation for a integer with a given length
    :param num: input number
    :param length: length of the string
    :return: string represetation of the integer
    '''
    rawstr = str(int(num))
    if length is None or length == len(rawstr):return rawstr
    elif length < len(rawstr): raise ValueError, 'Length of the number is longer then defined display length!'
    elif length > len(rawstr): return '0'*(length-len(rawstr)) + rawstr


def check_df_index(df):
    """
    check if the indices of a pandas dataframe is a series in an ascending order with increment as 1 and starts at 0
    :return: bool
    """
    return np.array_equal(df.index, np.arange(len(df)))


def plot_spike_ticks(spike_history, y=0., plot_axis=None, color='#ff0000', **kwargs):
    """
    plot spike ticks as separate dots
    :param spike_history: list of ints, spike timestamps on time unit axis
    :param y: float, vertical locations of spike line
    :param plot_axis: plotting axis, matplotlib.pyplot.axis object
    :param color: plotting color
    :param kwargs: other inputs to matplotlib.pyplot.plot function
    :return: plot_axis
    """

    if plot_axis is None:
        f = plt.figure(figsize=(10, 5))
        plot_axis = f.add_subplot(111)

    spk_num = len(spike_history)

    plot_axis.plot(spike_history, [float(y)] * spk_num, '.', mfc=color, **kwargs)


def short(str):
    """
    retrun abbreviation of a string
    """

    if str in ['north', 'south', 'east', 'west']:
        return str[0] * 2
    elif str in ['northwest', 'northeast', 'southwest', 'southeast']:
        return str[0] + str[5]
    elif str in ['hidden', 'eye', 'muscle']:
        return str[0]
    elif str in ['terrain', 'food', 'fish']:
        return str[0:4]


if __name__ == '__main__':

    #==================================================
    # ts_trigger = np.arange(5)
    # ts_reference = np.arange(5) + 0.1001
    #
    # ccg, t = discreat_crosscorrelation(ts_trigger, ts_reference, t_range=(-0.2, 0.5), bin_width=0.1)
    #
    # print(t)
    # print(ccg)
    #
    # plt.plot(t, ccg)
    # plt.show()
    # ==================================================

    # ==================================================
    plot_spike_ticks(range(5))
    plt.show()
    # ==================================================

    print('for debug ...')

