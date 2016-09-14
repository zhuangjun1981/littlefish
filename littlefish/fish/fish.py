# from __future__ import (absolute_import, division,
#                         print_function, unicode_literals)
# from builtins import *

import brain as brain
import numpy as np
import pandas as pd

class Fish(object):
    """
    the main fish class

    a 'fish' has a body occupies a 3x3 space.

    a 'fish' has a couple of sets of 8 eyes (brain.Eye object, each at each border pixel of the body). each set of eyes
    are receiving inputs from different objects. i.e. one set of eyes will look at land/water, another set of eyes will
    look for food, another set of eyes will look for other fish.

    a 'fish' has 4 invisible muscles (brain.Muscle object, each controlling the movement in each direction), each
    muscle receive inputs from one motor neuron (so the output layer of the fish's brain will be 4 motor neurons)

    between eyes and muscles are a neural network consists of neurons (brain.Neuron object) and connections
    (brain.Connections object). Number of layers and number of neurons can be specified. fish will have health point
    (going down over time and increase after eating a food). fish moves around in a 2d landscape (2-d binary map,
    0 represents water, 1 represents land), when hits land, health point will go down quickly.

    the purpose of simulation is to train the fish use its eyes, brain and muscles to avoid land and look for food.

    the simulation works on "real-time" basis on a time unit axis (consider one time unit is equivalent to 0.1
    millisecond.
    """

    def __init__(self, position, health=100., eyes=None, hidden_layers=None, muscles=None,
                 health_decay_rate=0.001):


        if len(position) != 2:
            raise(ValueError, 'position should contain two elements.')

        if (not isinstance(position[0], int)) or (not isinstance(position[1], int)):
            raise(ValueError, 'position should contain two integers.')

        self._position = position
        self._health = float(health)
        self._health_decay_rate = float(health_decay_rate)

        if eyes is None:
            self._eyes = {'eye_set_terrain': self._get_default_eye_set(),
                          'eye_set_food': self._get_default_eye_set()}
        else:
            self._eyes = eyes

        if hidden_layers is None:
            self._hidden_layers = {'hidden_layers_001': self._get_default_hidden_layer()}
        else:
            self._hidden_layers = hidden_layers

        if muscles is None:
            # todo: function to get default muscles
            pass
        else:
            self._muscles = muscles

    def _get_default_eye_set(self):
        """
        :return: a data frame, representing a full default set of eight eyes
                 columns: ['baseline_rate', 'direction', 'gain', 'input_filter', 'pos_col',
                           'pos_raw', 'refractory_period']
        """
        if len(self._position) != 2:
            raise(ValueError, 'position should contain two elements.')

        if (not isinstance(self._position[0], int)) or (not isinstance(self._position[1], int)):
            raise(ValueError, 'position should contain two integers.')

        row_list = np.array([0, -1, -1, -1, 0, 1, 1, 1]) + self._position[0]
        col_list = np.array([1, 1, 0, -1, -1, -1, 0, 1]) + self._position[1]
        direction_list = ['east', 'northeast', 'north', 'northwest', 'west', 'southwest', 'south', 'southeast']
        input_filter = np.array([[0.2, 0.6, 0.2]])

        eye_set = pd.DataFrame({'pos_raw': row_list,
                                'pos_col': col_list,
                                'direction': direction_list,
                                'input_filter': list(np.repeat(input_filter, 8, axis=0)),
                                'gain': 0.001,
                                'baseline_rate': 0.,
                                'refractory_period': 0})
        return eye_set

    def _get_default_hidden_layer(self):
        """
        :return: a data frame, representing a default hidden layer
        """
        hidden_layer = pd.DataFrame({'baseline_rate': [0.0001] * 16,
                                     'refractory_period': [10] *16})
        return hidden_layer

    def _get_default_motor_neuron_layer(self):
        """
        :return: a data frame, representing a default motor neuron layer
        """
        motor_neuron_layer = pd.DataFrame({'direction': ['east', 'north', 'west', 'south'],
                                           'baseline_rate': [0.0001] * 4,
                                           'refractory_period': [10] * 4})

    def _get_default_muscles(self):
        pass


if __name__ == '__main__':

    fish = Fish((5,6))
    fish._get_default_eye_set()
    fish._get_default_hidden_layer()





