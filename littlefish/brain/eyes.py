import numpy as np
from littlefish.core import utilities as util
from littlefish.brain.base import Neuron


class Eye(Neuron):
    """
    base class of eye, subclass of Neuron with additional attributes:
        gain: float, the overall gain of the weighted sum input
        input_type: str, from what entity does the eye receive input, could be "terrain", "food", or "fish" ("fish" is currently not implemented)
        eye_direction: str, in which direction the eye is looking at, can be "north", "south", "northeast", etc.
        rf_positions: 2d array, shape=(2, n), n is the number of pixels in the receptive field.
            the first row is the row idx of each rf pixel relative to the center of the fish body
            the second row is the col idx of each rf pixel relative to the center of the fish body
        rf_weights: 1d array, shape=(n,), weights of each rf pixel relative to the center of the fish body
    """

    def __init__(
        self,
        eye_direction: str = "north",
        rf_positions: np.ndarray = np.array([[-2], [0]]),
        rf_weights: np.ndarray = np.array([1.0]),
        gain: float = 1.0,
        input_type: str = "terrain",
        baseline_rate: float = 0.001,
        refractory_period: float = 1.2,
    ):
        super().__init__(
            baseline_rate=baseline_rate, refractory_period=refractory_period
        )

        self.type = "littlefish.brain.eyes.Eye"
        self.eye_direction = eye_direction
        self.rf_positions = np.array(rf_positions)
        self.rf_weights = np.array(rf_weights)

        assert self.rf_positions.ndim == 2
        assert self.rf_weights.ndim == 1
        assert self.rf_positions.shape[0] == 2
        assert self.rf_positions.shape[1] == rf_weights.shape[0]

        self.gain = float(gain)
        self.input_type = input_type

    def _get_input_pixel_values(self, input_map, body_position, border_value):
        def isin(r, c):
            return 0 <= r < n_rows and 0 <= c < n_cols

        n_rows, n_cols = input_map.shape
        input_positions = self.rf_positions + np.array([body_position]).transpose()
        input_values = np.array(
            [
                input_map[r, c] if isin(r, c) else border_value
                for r, c in zip(*input_positions)
            ],
            dtype=input_map.dtype,
        )

        return input_values

    def get_input(self, input_map, body_position, border_value):
        """
        given the input_map and body_position, return the value to the eye.
        The values of pixels in the eye's receptive field will be extracted
        timed with the self.rf_weights and return the sum

        :param input_map: 2d ndarray, the map of the eye's input type
        :param body_position: ndarray, (2,), row and col of the center of the fish body
        :param border_value: float, the value of pixels outside the map

        :return: float, the input value to the eye

        """
        input_values = self._get_input_pixel_values(
            input_map, body_position, border_value
        )
        input_value = self.gain * sum(input_values * self.rf_weights)
        return input_value

    def act(
        self,
        input_map: np.ndarray,
        body_position: np.ndarray,
        border_value: float,
        t_point: int,
        action_history: list,
        probability_base: None,
    ):
        input_value = self.get_input(
            input_map=input_map, body_position=body_position, border_value=border_value
        )

        return super().act(
            t_point=t_point,
            action_history=action_history,
            probability_input=input_value,
            probability_base=probability_base,
        )


EIGHT_EYES = {
    "southeast": {
        "rf_positions": [[1, 1], [1, 2], [2, 2], [2, 1], [2, 3], [3, 3], [3, 2]],
        "rf_weights": [0.1, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
    },
    "northwest": {
        "rf_positions": [
            [-1, -1],
            [-1, -2],
            [-2, -2],
            [-2, -1],
            [-2, -3],
            [-3, -3],
            [-3, -2],
        ],
        "rf_weights": [0.1, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
    },
    "northeast": {
        "rf_positions": [[-1, 1], [-1, 2], [-2, 2], [-2, 1], [-2, 3], [-3, 3], [-3, 2]],
        "rf_weights": [0.1, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
    },
    "southwest": {
        "rf_positions": [[1, -1], [1, -2], [2, -2], [2, -1], [2, -3], [3, -3], [3, -2]],
        "rf_weights": [0.1, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
    },
    "north": {
        "rf_positions": [
            [-1, 0],
            [-2, -1],
            [-2, 0],
            [-2, 1],
            [-3, -1],
            [-3, 0],
            [-3, 1],
        ],
        "rf_weights": [0.1, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
    },
    "south": {
        "rf_positions": [[1, 0], [2, -1], [2, 0], [2, 1], [3, -1], [3, 0], [3, 1]],
        "rf_weights": [0.1, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
    },
    "east": {
        "rf_positions": [[0, 1], [-1, 2], [0, 2], [1, 2], [-1, 3], [0, 3], [1, 3]],
        "rf_weights": [0.1, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
    },
    "west": {
        "rf_positions": [
            [0, -1],
            [-1, -2],
            [0, -2],
            [1, -2],
            [-1, -3],
            [0, -3],
            [1, -3],
        ],
        "rf_weights": [0.1, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
    },
}


FOUR_EYES = {
    "north": {
        "rf_positions": [
            [0, 0],
            [-1, -1],
            [-1, 0],
            [-1, 1],
            [-2, -2],
            [-2, -1],
            [-2, 0],
            [-2, 1],
            [-2, 2],
            [-3, -3],
            [-3, -2],
            [-3, -1],
            [-3, 0],
            [-3, 1],
            [-3, 2],
            [-3, 3],
        ],
        "rf_weights": [
            1,
            0.243116734,
            0.367879441,
            0.243116734,
            0.059105747,
            0.106877926,
            0.135335283,
            0.106877926,
            0.059105747,
            0.014369596,
            0.027172461,
            0.04232922,
            0.049787068,
            0.04232922,
            0.027172461,
            0.014369596,
        ],
    },
    "south": {
        "rf_positions": [
            [0, 0],
            [1, -1],
            [1, 0],
            [1, 1],
            [2, -2],
            [2, -1],
            [2, 0],
            [2, 1],
            [2, 2],
            [3, -3],
            [3, -2],
            [3, -1],
            [3, 0],
            [3, 1],
            [3, 2],
            [3, 3],
        ],
        "rf_weights": [
            1,
            0.243116734,
            0.367879441,
            0.243116734,
            0.059105747,
            0.106877926,
            0.135335283,
            0.106877926,
            0.059105747,
            0.014369596,
            0.027172461,
            0.04232922,
            0.049787068,
            0.04232922,
            0.027172461,
            0.014369596,
        ],
    },
    "east": {
        "rf_positions": [
            [0, 0],
            [-1, 1],
            [0, 1],
            [1, 1],
            [-2, 2],
            [-1, 2],
            [0, 2],
            [1, 2],
            [2, 2],
            [-3, 3],
            [-2, 3],
            [-1, 3],
            [0, 3],
            [1, 3],
            [2, 3],
            [3, 3],
        ],
        "rf_weights": [
            1,
            0.243116734,
            0.367879441,
            0.243116734,
            0.059105747,
            0.106877926,
            0.135335283,
            0.106877926,
            0.059105747,
            0.014369596,
            0.027172461,
            0.04232922,
            0.049787068,
            0.04232922,
            0.027172461,
            0.014369596,
        ],
    },
    "west": {
        "rf_positions": [
            [0, 0],
            [-1, -1],
            [0, -1],
            [1, -1],
            [-2, -2],
            [-1, -2],
            [0, -2],
            [1, -2],
            [2, -2],
            [-3, -3],
            [-2, -3],
            [-1, -3],
            [0, -3],
            [1, -3],
            [2, -3],
            [3, -3],
        ],
        "rf_weights": [
            1,
            0.243116734,
            0.367879441,
            0.243116734,
            0.059105747,
            0.106877926,
            0.135335283,
            0.106877926,
            0.059105747,
            0.014369596,
            0.027172461,
            0.04232922,
            0.049787068,
            0.04232922,
            0.027172461,
            0.014369596,
        ],
    },
}
