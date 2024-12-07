import numpy as np
from littlefish.brain.base import Neuron


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

        self.type = "littlefish.brain.Muscle"

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


# direction, step_motion pair
FOUR_MUSCLES = [
    ("north", np.array([-1, 0], dtype=np.int8)),
    ("south", np.array([1, 0], dtype=np.int8)),
    ("east", np.array([0, -1], dtype=np.int8)),
    ("west", np.array([0, 1], dtype=np.int8)),
]
