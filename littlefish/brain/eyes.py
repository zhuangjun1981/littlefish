import numpy as np
from littlefish.core import utilities as util
from littlefish.brain.base import Neuron


class Eye(Neuron):
    """
    base class of eye, subclass of Neuron with additional attributes:
        gain: float, the overall gain of the weighted sum input
        input_type: str, from what entity does the eye receive input, could be "terrain", "food", or "fish" ("fish" is currently not implemented)
        eye_position: 1d array, shape=(2,), row and col of the position of the eye relative to the center of the fish body
        rf_positions: 2d array, shape=(2, n), n is the number of pixels in the receptive field.
            the first row is the row idx of each rf pixel relative to the center of the fish body
            the second row is the col idx of each rf pixel relative to the center of the fish body
        rf_weights: 1d array, shape=(n,), weights of each rf pixel relative to the center of the fish body
    """

    def __init__(
        self,
        eye_position=np.array([-1, 0]),
        rf_positions=np.array([[-2], [0]]),
        rf_weights=np.array([1.0]),
        gain=1.0,
        input_type="terrain",
        baseline_rate=0.001,
        refractory_period=1.2,
    ):
        super().__init__(
            baseline_rate=baseline_rate, refractory_period=refractory_period
        )
        self.type = "littlefish.brain.eyes.Eye"
        self.eye_position = eye_position
        self.rf_positions = rf_positions
        self.rf_weights = rf_weights
        self.gain = float(gain)
        self.input_type = input_type

    def get_input(self):
        pass

    def act(self):
        pass


# def get_eye_type(ind, dir_num=4):
#     """
#     given the neuron_ind in the eye layer return direction and input type of a specific eye

#     :param ind: non-negative int, index of the eye in eye layer
#     :param dir_num: 4 or 8, if 4, eye directions iterate through 4 cardinal directions; if 8, eye directions will
#                     iterate through 8 directions
#     :return: two strings, (direction, type)
#     """

#     if dir_num == 8:
#         eye_directions = [
#             "east",
#             "northeast",
#             "north",
#             "northwest",
#             "west",
#             "southwest",
#             "south",
#             "southeast",
#         ]
#     elif dir_num == 4:
#         eye_directions = ["east", "north", "west", "south"]
#     else:
#         raise ValueError("cannot understand dir_num, should be 4 or 8.")

#     eye_types = ["terrain", "food", "fish"]
#     direction_num = len(eye_directions)
#     type_num = len(eye_types)
#     return (
#         eye_directions[ind % direction_num],
#         eye_types[(ind // direction_num) % type_num],
#     )


# class WideEye(Neuron):
#     """
#     Eye class to observe the environment and the inside of fish body, subclass of Neuron, has eye sight as 2 pixels
#     """

#     def __init__(
#         self,
#         direction,
#         input_filter=None,
#         gain=0.001,
#         input_type="terrain",
#         baseline_rate=0.0,
#         refractory_period=1.2,
#     ):
#         """
#         for a fish occupies 3x3 space, consider the eyes are in the outer rim of the body (the 4 pixels surrounding the
#         central pixel in 4 cardinal direction. Each eye receives the inputs from the given direction in the environment.
#         for example:

#         fish (1) in the environment(0):
#         0 0 0 0 0 0 0
#         0 0 0 0 0 0 0
#         0 0 1 1 1 0 0
#         0 0 1 1 1 0 0
#         0 0 1 1 1 0 0
#         0 0 0 0 0 0 0
#         0 0 0 0 0 0 0

#         eye (2) in the east direction is:
#         0 0 0 0 0 0 0
#         0 0 0 0 0 0 0
#         0 0 1 1 1 0 0
#         0 0 1 1 2 0 0
#         0 0 1 1 1 0 0
#         0 0 0 0 0 0 0
#         0 0 0 0 0 0 0

#         it receive inputs from pixels labelled as 3 in the environment:
#         0 0 0 0 0 0 3
#         0 0 0 0 0 3 3
#         0 0 0 0 3 3 3
#         0 0 0 3 3 3 3
#         0 0 0 0 3 3 3
#         0 0 0 0 0 3 3
#         0 0 0 0 0 0 3

#         the inputs from the environment (1x16 array) will be filtered by a array with same size, to generate
#         a single value as the base of its input. this value will be multiplied by a float number gain to generate
#         final input probability.

#         self._get_input_pixels will pick pixels within the eye's receptive field in the following order:
#         0 0 0 0 0 0 10
#         0 0 0 0 0 5 11
#         0 0 0 0 2 6 12
#         0 0 0 1 3 7 13
#         0 0 0 0 4 8 14
#         0 0 0 0 0 9 15
#         0 0 0 0 0 0 16

#         :param direction: str, the aim of the eye, should be one of the following, 'east', 'north', 'west' and 'south'
#         :param input_filter: 1d array, shape: (16,), filter to transform input pixel values to a single number.
#                              analog of linear receptive field of a retinal ganglion cell. default input_filter is
#                              set as the following:

#                              input_filter[pixel_i] = exp(-1 * cartesian_distance(pixel_i, pixel_body_center)
#         :param baseline_rate: float, probablity of a action per time unit.
#         :param refractory_period: float, refractory_period in time unit
#         :param input_type: str, type of the input the eye receives, should be one of 'terrain', 'food', 'fish',
#                default: 'terrain'
#         """

#         self._direction = direction
#         self._rf_positions = self._get_rf_positions()
#         self._gain = float(gain)

#         if input_filter is None:
#             self._input_filter = np.array(
#                 [
#                     1,
#                     0.243116734,
#                     0.367879441,
#                     0.243116734,
#                     0.059105747,
#                     0.106877926,
#                     0.135335283,
#                     0.106877926,
#                     0.059105747,
#                     0.014369596,
#                     0.027172461,
#                     0.04232922,
#                     0.049787068,
#                     0.04232922,
#                     0.027172461,
#                     0.014369596,
#                 ]
#             )
#         else:
#             self._input_filter = input_filter.astype(np.float32)

#         if input_type in ["terrain", "food", "fish"]:
#             self._input_type = input_type
#         else:
#             raise ValueError(
#                 'Eye: type should be one of the following: "terrain", "food", "fish".'
#             )

#         curr_baseline_rate = float(baseline_rate)
#         curr_refractory_period = float(refractory_period)

#         super(WideEye, self).__init__(
#             baseline_rate=curr_baseline_rate, refractory_period=curr_refractory_period
#         )

#         self.type = "littlefish.brain.WideEye"

#     def copy(self):
#         """

#         :return: a copy of self for i.e. mutation
#         """

#         return WideEye(
#             direction=self.get_direction(),
#             input_filter=self.get_input_filter(),
#             gain=self.get_gain(),
#             input_type=self.get_input_type(),
#             baseline_rate=self.get_baseline_rate(),
#             refractory_period=self.get_refractory_period(),
#         )

#     def get_direction(self):
#         return self._direction

#     def get_input_filter(self):
#         return self._input_filter

#     def get_gain(self):
#         return self._gain

#     def get_input_type(self):
#         return self._input_type

#     def get_neuron_type(self):
#         return "eye"

#     def _get_rf_positions(self):
#         """
#         get pixel coordinate of receptive field

#         :return rf_positions: 2d array, shape: (16, 2), _dtype: int64. each row is a pixel in the receptive field,
#                               [row, col] relative to the position of the body center pixel
#         """

#         array1 = np.array(
#             [0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3], dtype=np.int32
#         )
#         array2 = np.array(
#             [0, -1, 0, 1, -2, -1, 0, 1, 2, -3, -2, -1, 0, 1, 2, 3], dtype=np.int32
#         )

#         if self._direction == "east":
#             rf_pos = np.array([array2, array1]).transpose()
#         elif self._direction == "north":
#             rf_pos = np.array([array1 * -1, array2]).transpose()
#         elif self._direction == "west":
#             rf_pos = np.array([array2, array1 * -1]).transpose()
#         elif self._direction == "south":
#             rf_pos = np.array([array1, array2]).transpose()
#         else:
#             raise ValueError(
#                 "Eye: direction should be one of the following: ['east', 'north', 'west', 'south']."
#             )

#         return rf_pos

#     def _get_input_pixels(self, input_map, body_position, border_value=1):
#         """
#         get a 1d array of all pixels within the eye's receptive field

#         :param input_map: 2d array, the map the eye is looking
#         :param body_position: list of two non-negative integers, [row, col] location of body center pixel
#         :param border_value: float, values for pixels outside the border of input_map
#         :return: the 1d array with the values of the 16 pixels in the eye's receptive field. pixels out of the
#         input_map range will be returned as border_value
#         """

#         body_pos = np.array(body_position, dtype=np.int32)
#         input_pixels = np.zeros(16, dtype=np.float32)

#         for ind, curr_pos in enumerate(self._rf_positions):
#             curr_rf_pos = body_pos + curr_pos
#             try:
#                 if curr_rf_pos[0] < 0 or curr_rf_pos[1] < 0:
#                     input_pixels[ind] = border_value
#                 else:
#                     input_pixels[ind] = input_map[curr_rf_pos[0], curr_rf_pos[1]]
#             except IndexError:
#                 input_pixels[ind] = border_value

#         return input_pixels

#     def _get_input(self, input_map, body_position, border_value=1):
#         """

#         :return: float, calculate real time input from the visual field
#         """

#         input_pixels = self._get_input_pixels(
#             input_map, body_position, border_value=border_value
#         )
#         probability_input = self._gain * np.sum(input_pixels * self._input_filter)

#         return probability_input

#     def act(
#         self,
#         t_point,
#         body_position,
#         input_map,
#         action_history=[],
#         border_value=1,
#         probability_base=None,
#     ):
#         """
#         evaluate if the eye neuron will fire at given time point

#         :param t_point: int, current time point as the index of time unit axis
#         :param body_position: tuple of two ints, (row, col),  position of the body center pixel
#         :param input_map: binary 2-d map, for now it should only contain 0s and 1s
#         :param action_history: list of positive integers, list of time stamps of actions of this neuron,
#                                should be monotonically increasing
#         :param border_value: int, default 1, value for pixels outside the terrain_map
#         :param probability_base: float, a random number no less than 0 and less than 1, to determine if the neuron
#                                  is going to act or not, if None, a random number will be generated by
#                                  random.random()
#         :return: bool, True: fire; False: quite
#         """

#         probability_input = self._get_input(
#             input_map=input_map, body_position=body_position, border_value=border_value
#         )

#         return super(Eye, self).act(
#             t_point,
#             action_history=action_history,
#             probability_input=probability_input,
#             probability_base=probability_base,
#         )

#     def to_h5_group(self, h5_group, additional_kv_pairs):

#         additional_kv_pairs.update(
#             {
#                 "direction": self._direction,
#                 "input_filter": self._input_filter,
#                 "gain": self._gain,
#                 "input_type": self._input_type,
#                 "rf_positions": self._rf_positions,
#                 "rf_data_format": "16 (pixel_num) x 2 ([row, col])",
#                 "rf_description": "coordinates of each pixel of this eye's receptive field relative to body center position",
#             }
#         )
#         self.super().to_h5_group(h5_group, additional_kv_pairs)

#     @staticmethod
#     def from_h5_group(h5_group):
#         if util.decode(h5_group.attrs["neuron_type"]) != "eye":
#             raise ValueError(
#                 'Eye: loading from h5 file failed. "neuron_type" attribute should be "eye".'
#             )

#         direction = util.decode(h5_group["direction"][()])
#         input_type = util.decode(h5_group["input_type"][()])

#         eye = WideEye(
#             direction=direction,
#             input_filter=h5_group["input_filter"][()],
#             gain=h5_group["gain"][()],
#             input_type=input_type,
#             baseline_rate=h5_group["baseline_rate"][()],
#             refractory_period=h5_group["refractory_period"][()],
#         )
#         return eye
