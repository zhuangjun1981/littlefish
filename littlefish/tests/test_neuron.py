import os
import h5py
import unittest
import random
import numpy as np
from littlefish.brain.base import Neuron
from littlefish.brain.eyes import Eye
from littlefish.brain.muscles import Muscle
from littlefish.brain.functional import load_neuron_from_h5_group


class TestNeuron(unittest.TestCase):
    def setup(self):
        random.seed(42)
        np.random.seed(42)

    def test_neuron_io(self):
        curr_folder = os.path.dirname(os.path.abspath(__file__))
        temp_path = os.path.join(curr_folder, "temp_file.h5")

        neuron = Neuron()
        f_temp = h5py.File(temp_path, "a")
        h5_grp = f_temp.create_group("neuron")
        neuron.to_h5_group(h5_grp)
        neuron2 = load_neuron_from_h5_group(h5_grp)
        f_temp.close()

        assert neuron.baseline_rate == neuron2.baseline_rate
        assert neuron.type == neuron2.type
        assert neuron.refractory_period == neuron2.refractory_period

        os.remove(temp_path)

    def test_neuron_act(self):
        total_t = 10
        neuron = Neuron(baseline_rate=0.5, refractory_period=1.2)
        action_history = []
        for t_point in range(total_t):
            neuron.act(
                t_point=t_point,
                action_history=action_history,
                probability_input=0.5,
            )
        assert action_history == [0, 2, 4, 6, 8]

        neuron1 = Neuron(baseline_rate=0.0, refractory_period=1.2)
        action_history = []
        for t_point in range(total_t):
            if t_point in [0, 1, 5, 9]:
                probility_input = 1.0
            else:
                probility_input = 0.0

            neuron1.act(
                t_point=t_point,
                action_history=action_history,
                probability_input=probility_input,
            )
        assert action_history == [0, 5, 9]

        neuron2 = Neuron(baseline_rate=0.2, refractory_period=1.2)
        action_history = []
        for t_point in range(total_t):
            if t_point in [0, 1, 5, 9]:
                probility_input = 0.5
            else:
                probility_input = 0.0

            if t_point in [2, 4, 5, 8, 9]:
                probility_base = 0.3
            else:
                probility_base = 0.0

            neuron2.act(
                t_point=t_point,
                action_history=action_history,
                probability_input=probility_input,
                probability_base=probility_base,
            )
        assert action_history == [0, 3, 5, 7, 9]

    def test_eye_io(self):
        curr_folder = os.path.dirname(os.path.abspath(__file__))
        temp_path = os.path.join(curr_folder, "temp_file.h5")

        eye = Eye()
        f_temp = h5py.File(temp_path, "a")
        h5_grp = f_temp.create_group("eye")
        eye.to_h5_group(h5_grp)

        eye2 = load_neuron_from_h5_group(h5_grp)
        f_temp.close()

        assert eye.type == eye2.type
        assert eye.baseline_rate == eye2.baseline_rate
        assert eye.refractory_period == eye2.refractory_period
        assert eye.gain == eye2.gain
        assert eye.input_type == eye2.input_type
        assert eye.eye_direction == eye2.eye_direction
        assert np.array_equal(eye.rf_positions, eye2.rf_positions)
        assert np.array_equal(eye.rf_weights, eye2.rf_weights)

        os.remove(temp_path)

    def test_eye_get_input(self):
        input_map = np.arange(20).reshape(4, 5)
        eye = Eye(
            rf_positions=np.array([[0, -1, -2, -3, -1], [0, 0, 0, 0, 1]]),
            rf_weights=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        )

        input_pixel_values = eye._get_input_pixel_values(
            input_map=input_map,
            body_position=(2, 2),
            border_value=0.0,
        )

        assert np.array_equal(input_pixel_values, np.array([12, 7, 2, 0, 8]))

        input_value = eye.get_input(
            input_map=input_map, body_position=[2, 2], border_value=3.0
        )

        assert input_value == 8.4

    def test_eye_act(self):
        input_map = np.arange(20).reshape(4, 5)
        eye = Eye(
            rf_positions=np.array([[0, -1, -2, -3, -1], [0, 0, 0, 0, 1]]),
            rf_weights=np.array([0.01, 0.02, 0.03, 0.04, 0.05]),
            baseline_rate=0.0,
        )
        input_value = eye.get_input(
            input_map=input_map, body_position=[2, 2], border_value=1.0
        )
        assert input_value == 0.76

        action_history = []
        eye.act(
            input_map=input_map,
            body_position=[2, 2],
            border_value=1.0,
            t_point=3,
            action_history=action_history,
            probability_base=0.7,
        )
        eye.act(
            input_map=input_map,
            body_position=[2, 2],
            border_value=1.0,
            t_point=5,
            action_history=action_history,
            probability_base=0.9,
        )
        assert action_history == [3]

    def test_muscle_io(self):
        curr_folder = os.path.dirname(os.path.abspath(__file__))
        temp_path = os.path.join(curr_folder, "temp_file.h5")

        muscle = Muscle()
        f_temp = h5py.File(temp_path, "a")
        h5_grp = f_temp.create_group("muscle")
        muscle.to_h5_group(h5_grp)

        muscle2 = load_neuron_from_h5_group(h5_grp)
        f_temp.close()

        assert muscle.type == muscle2.type
        assert muscle.baseline_rate == muscle2.baseline_rate
        assert muscle.refractory_period == muscle2.refractory_period
        assert muscle.direction == muscle2.direction
        os.remove(temp_path)

    def test_muscle_action(self):
        simulation_length = 20000
        muscle = Muscle(
            direction="east",
            step_motion=np.array([0, 1]),
            baseline_rate=0.0,
            refractory_period=5000,
        )
        action_history = []
        movements = []
        for i in range(simulation_length):
            movement = muscle.act(
                i,
                action_history=action_history,
                probability_input=0.5,
                probability_base=0.1,
            )
            if movement is not False:
                movements.append(movement)
        target_movements = [
            np.array([0, 1], dtype=np.uint8),
            np.array([0, 1], dtype=np.uint8),
            np.array([0, 1], dtype=np.uint8),
            np.array([0, 1], dtype=np.uint8),
        ]
        assert all(
            [np.array_equal(movements[i], target_movements[i]) for i in range(4)]
        )
        assert action_history == [0, 5000, 10000, 15000]


if __name__ == "__main__":
    neuron_tests = TestNeuron()
    neuron_tests.test_neuron_io()
    neuron_tests.test_neuron_act()
    neuron_tests.test_eye_io()
    neuron_tests.test_eye_get_input()
    neuron_tests.test_eye_act()
    neuron_tests.test_muscle_io()
    neuron_tests.test_muscle_action()
