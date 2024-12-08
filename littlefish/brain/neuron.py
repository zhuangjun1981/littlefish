import random
import numpy as np


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

# direction, step_motion pair
FOUR_MUSCLES = [
    ("north", np.array([-1, 0], dtype=np.int8)),
    ("south", np.array([1, 0], dtype=np.int8)),
    ("east", np.array([0, -1], dtype=np.int8)),
    ("west", np.array([0, 1], dtype=np.int8)),
]


class Neuron:
    """
    a very simple neuron class
    """

    def __init__(self, baseline_rate=0.001, refractory_period=1.2):
        """
        action is the equivalent of action potential in biology, and consider one time unit is 0.1 milisecond

        :param baseline_rate: float, probablity of a action per time unit.
        :param refractory_period: float, refractory_period in time unit
        """

        self.baseline_rate = float(baseline_rate)
        self.refractory_period = float(refractory_period)
        self.type = "littlefish.brain.neuron.Neuron"

    def __str__(self):
        return f"{self.type} object"

    def set_baseline_rate(self, new_baseline_rate):
        self.baseline_rate = float(new_baseline_rate)

    def set_refractory_period(self, new_refractory_period):
        self.refractory_period = new_refractory_period

    def act(
        self, t_point, action_history=[], probability_input=0.0, probability_base=None
    ):
        """
        evaluate if the neuron will fire at given time point

        :param t_point: int, current time point as the index of time unit axis
        :param action_history: list of positive integers, list of time stamps of actions of this neuron,
                               should be monotonically increasing
        :param probability_input: float, summed connection inputs, as add on to baseline_rate
        :param probability_base: float, a random number no less than 0 and less than 1, to determine if the neuron
                                 is going to act or not, if None, a random number will be generated by
                                 random.random()
        :return: bool, True: fire; False: quite
        """

        if probability_base is None:
            probability_base = random.random()
        probability_base = np.clip(probability_base, 0, 1)

        # this block of code is commented out to speed up simulation
        # else:
        #     if probability_base < 0. or probability_base >= 1.:
        #         raise ValueError('Neuron: probability_base should be no less than 0 and less than 1.')

        # this block of code is commented out to speed up simulation
        # if len(action_history) >= 2:
        #     if not util.check_monotonicity(np.array(action_history), direction='increasing'):
        #         raise ValueError('Neuron: action history should be monotonically increasing.')

        if (
            len(action_history) > 0
            and t_point - action_history[-1] < self.refractory_period
        ):
            return False
        else:
            curr_rate = np.clip(self.baseline_rate + probability_input, 0, 1 + 1e-6)
            if probability_base < curr_rate:
                action_history.append(t_point)
                return True
            else:
                return False

    def to_h5_group(self, h5_group, additional_kv_pairs={}):
        attributes = vars(self)
        attributes.update(additional_kv_pairs)

        for k, v in attributes.items():
            dset = h5_group.create_dataset(k, data=v)
            if k == "baseline_rate":
                dset.attrs["unit"] = "action_per_time_unit"
            if k == "refractory_period":
                dset.attrs["unit"] = "time_unit"


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

        self.type = "littlefish.brain.neuron.Eye"
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
        probability_base: float = None,
    ) -> bool:
        if input_map is None:
            return False

        input_value = self.get_input(
            input_map=input_map, body_position=body_position, border_value=border_value
        )

        return super().act(
            t_point=t_point,
            action_history=action_history,
            probability_input=input_value,
            probability_base=probability_base,
        )


class Muscle(Neuron):
    """
    muscle class for determining the motion of the fish. Subclass of Neuron class
    """

    def __init__(
        self,
        direction="south",
        step_motion=np.array([1, 0], dtype=np.int8),
        baseline_rate=0.001,
        refractory_period=500.0,
    ):
        self.direction = direction
        self.step_motion = step_motion

        super().__init__(
            baseline_rate=baseline_rate, refractory_period=refractory_period
        )

        self.type = "littlefish.brain.neuron.Muscle"

    def act(
        self, t_point, action_history=[], probability_input=0.0, probability_base=None
    ):
        """
        evaluate if the muscle will try to move the fish or not

        :param t_point: int, current time point as the index of time unit axis
        :param probability_input: float, summed connection inputs, as add on to baseline_rate
        :param action_history: list of positive integers, list of time stamps of actions of this neuron,
                               should be monotonically increasing
        :param probability_base: float, a random number no less than 0 and less than 1, to determine if the neuron
                                 is going to act or not, if None, a random number will be generated by
                                 random.random()
        :return: no attempt: False
                 attempt: movement vector, 1d array with two ints, [row_update, col_update]
        """

        is_act = super().act(
            t_point,
            action_history=action_history,
            probability_input=probability_input,
            probability_base=probability_base,
        )

        if is_act:
            return self.step_motion
        else:
            return is_act
