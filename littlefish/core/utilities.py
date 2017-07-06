# from __future__ import (absolute_import, division,
#                         print_function, unicode_literals)
# from builtins import *

import numpy as np
import numbers
import matplotlib.pyplot as plt
import scipy.ndimage as ni


def is_integer(var):
    return isinstance(var, numbers.Integral)


def array_nor(arr):
    arr = arr.astype(np.float32)
    return (arr - np.amin(arr)) / (np.amax(arr) - np.amin(arr))


def normalized_to_range(var, input_range=(0., 1.), output_range=(0., 1.)):
    """
    map a single value to the output_range according to it's position in the input_range.

    :param var: int or float, input value
    :param input_range: tuple of two floats, the reference range
    :param output_range: tuple of two floats, the mapping range
    :return: float, mapped value
    """

    if len(input_range) != 2:
        raise ValueError('input_range should contain two and only two numbers.')

    if input_range[1] <= input_range[0]:
        raise ValueError('input_range[1] should be larger than range[0].')

    if len(output_range) != 2:
        raise ValueError('output_range should contain two and only two numbers.')

    if output_range[1] <= output_range[0]:
        raise ValueError('output_range[1] should be larger than range[0].')

    var = float(var)
    input_range = (float(input_range[0]), float(input_range[1]))
    output_range = (float(output_range[0]), float(output_range[1]))
    if var < input_range[0] or var > input_range[1]:
        raise ValueError('input variable is out of the input range.')

    return ((var - input_range[0]) / (input_range[1] - input_range[0])) * \
           (output_range[1] - output_range[0]) + output_range[0]


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
                         supported: ('flat', mean, range), ('gaussian, mean, sigma), ('exponential', mean)
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
    """
    generate a string representation for a integer with a given length

    :param num: input number
    :param length: length of the string
    :return: string represetation of the integer
    """

    rawstr = str(int(num))
    if length is None or length == len(rawstr):return rawstr
    elif length < len(rawstr): raise ValueError('Length of the number is longer then defined display length!')
    elif length > len(rawstr): return '0'*(length-len(rawstr)) + rawstr


def check_df_index(df):
    """
    check if the indices of a pandas dataframe is a series in an ascending order with increment as 1 and starts at 0

    :return: bool
    """

    return np.array_equal(df.index, np.arange(len(df)))


def check_arithmetic_progression(seq):
    """
    check a 1-d list of numbers is arithmetic progression sequence

    :return bool
    """

    step = seq[1] - seq[0]

    return np.array_equal(seq, np.arange(len(seq)) * step + seq[0])


def check_binary_2d_array(array):
    """
    check if an array is 2 dimensional and _dtype is int and only contains 0s and 1s
    """

    if not isinstance(array, np.ndarray):
        return False

    if len(array.shape) != 2:
        return False

    if not np.issubdtype(array.dtype, np.integer):
        return False

    if np.min(array[:]) < 0 or np.max(array[:]) > 1:
        return False

    return True


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


def short(input_str):
    """
    retrun abbreviation of a string
    """

    if input_str in ['north', 'south', 'east', 'west']:
        output_str =  input_str[0] * 2
    elif input_str in ['northwest', 'northeast', 'southwest', 'southeast']:
        output_str = input_str[0] + input_str[5]
    elif input_str in ['neuron', 'hidden', 'eye', 'muscle']:
        output_str = input_str[0]
    elif input_str in ['terrain', 'food', 'fish']:
        output_str = input_str[0:4]
    else:
        raise ValueError('littlefish.core.utilities.short(): do not understand input string.')

    return output_str.upper()


def plot_mask_borders(mask, plot_axis=None, color='#ff0000', border_width=2, closing_iteration=None, **kwargs):
    """
    plot mask (ROI) borders by using pyplot.contour function. all the 0s and Nans in the input mask will be considered
    as background, and non-zero, non-nan pixel will be considered in ROI.
    """

    if not check_binary_2d_array(mask):
        raise(ValueError, 'input mask should be a 2d binary numpy.ndarray with _dtype as integer and contains '
                          'only 0s and 1s.')

    if not plot_axis:
        f = plt.figure()
        plot_axis = f.add_subplot(111)

    if closing_iteration is not None:
        ploting_mask = ni.binary_closing(mask, iterations=closing_iteration).astype(np.uint8)
    else:
        ploting_mask = mask

    currfig = plot_axis.contour(ploting_mask, levels=[0.5], colors=color, linewidths=border_width, **kwargs)

    # put y axis in decreasing order
    y_lim = list(plot_axis.get_ylim())
    y_lim.sort()
    plot_axis.set_ylim(y_lim[::-1])

    plot_axis.set_aspect('equal')

    return currfig


def plot_mask(mask, plot_axis=None, color='#ff0000', closing_iteration=None, **kwargs):
    """
    plot mask (ROI) borders by using pyplot.contour function. all the 0s and Nans in the input mask will be considered
    as background, and non-zero, non-nan pixel will be considered in ROI.
    """

    if not check_binary_2d_array(mask):
        raise(ValueError, 'input mask should be a 2d binary numpy.ndarray with _dtype as integer and contains '
                          'only 0s and 1s.')

    if not plot_axis:
        f = plt.figure()
        plot_axis = f.add_subplot(111)

    if closing_iteration is not None:
        ploting_mask = ni.binary_closing(mask, iterations=closing_iteration).astype(np.uint8)
    else:
        ploting_mask = mask

    currfig = plot_axis.contourf(ploting_mask, levels=[0.5, 1], colors=color, **kwargs)

    # put y axis in decreasing order
    y_lim = list(plot_axis.get_ylim())
    y_lim.sort()
    plot_axis.set_ylim(y_lim[::-1])

    plot_axis.set_aspect('equal')

    return currfig


def check_monotonicity(arr, direction='increasing'):
    """
    check monotonicity of a 1-d array, usually a time series

    :param arr: input array, should be 1 dimensional
    :param direction: 'increasing', 'decreasing', 'non-increasing', 'non-decreasing'
    :return: True or False
    """

    if len(arr.shape) != 1:
        raise ValueError('Input array should be one dimensional!')

    if arr.shape[0] < 2:
        raise ValueError('Input array should have at least two elements!')

    diff = np.diff(arr)
    min_diff = np.min(diff)
    max_diff = np.max(diff)

    if direction == 'increasing':
        if min_diff > 0:
            return True
        else:
            return False

    elif direction == 'decreasing':
        if max_diff < 0:
            return True
        else:
            return False

    elif direction == 'non-increasing':
        if max_diff <= 0:
            return True
        else:
            return False

    elif direction == 'non-decreasing':
        if min_diff >= 0:
            return True
        else:
            return False

    else:
        raise LookupError('direction should one of the following: "increasing", "decreasing", '
                          '"non-increasing", "non-decreasing"!')


def decode(str_like, code='UTF-8'):
    """
    if a string like object is actually a 'bytes' type, decode it by 'UTF-8', and return the string. This is to deal
    with the hdf5 string format madness.

    :param str_like:
    :param code:
    :return: str
    """

    if isinstance(str_like, str):
        return str_like
    elif isinstance(str_like, bytes):
        return str_like.decode(code)
    else:
        raise ValueError('Utility: decode function do not understand the input type. Should be "str" or "bytes".')


def get_rgb(color_str):
    """
    get R,G,B int value from a hex color string
    """
    return int(color_str[1:3], 16), int(color_str[3:5], 16), int(color_str[5:7], 16)


def get_color_str(r, g, b):
    """
    get hex color string from r,g,b value (integer with uint8 format)
    """
    if not (is_integer(r) and is_integer(g) and is_integer(b)):
        raise TypeError('Input r, g and b should be integer!')

    if not ((0 <= r <= 255) and (0 <= g <= 255) and (0 <= b <= 255)):
        raise ValueError('Input r, g and b should between 0 and 255!')
    return '#{:0<2}{:0<2}{:0<2}'.format(hex(r)[2:], hex(g)[2:], hex(b)[2:])


def value_2_rgb(value, cmap):
    """
    get the RGB value as format as hex string from the decimal ratio of a given colormap (from 0 to 1)
    """
    cmap = plt.get_cmap(cmap)
    color = cmap(value)[0:3]
    color = [int(x * 255) for x in color]
    return get_color_str(*color)


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
    # plot_spike_ticks(range(5))
    # plt.show()
    # ==================================================

    # ==================================================
    # array = np.zeros((100, 100), _dtype=np.uint8)
    # array[35: 38, 40: 43] = 1
    # array[35, 40] = 1
    # plot_mask(array)
    # plt.show()
    # ==================================================

    # ==================================================
    print(check_arithmetic_progression(range(5)))
    print(check_arithmetic_progression(np.arange(3, 8, 0.05)))
    seq = np.arange(3, 8, 0.05)
    seq[10] += 0.01
    print(check_arithmetic_progression(seq))
    # ==================================================

    print('for debug ...')

