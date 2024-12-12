import unittest
import random
import numpy as np
import littlefish.brain.brain as brain
from littlefish.brain.functional import (
    load_brain_from_h5_group,
    generate_brain_from_brain_config,
    plot_brain_connections,
)


class TestBrain(unittest.TestCase):
    def setup(self):
        pass

    def test_generate_minimal_brain(self):
        min_brain = brain.generate_minimal_brain()
        assert min_brain.get_postsynaptic_indices(0) == [1, 2]
        assert min_brain.get_presynaptic_indices(3) == [1, 2]

    def test_neuron_fire(self):
        min_brain = brain.generate_minimal_brain()
        min_brain.initiate_simulation(max_simulation_length=20)
        min_brain.neuron_fire(presynaptic_idx=0, t_point=2)
        assert np.allclose(
            min_brain.simulation_cache["psp_waveforms"],
            np.array(
                [
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0002,
                        0.0004,
                        0.0006,
                        0.0008,
                        0.001,
                        0.0009,
                        0.0008,
                        0.0007,
                        0.0006,
                        0.0005,
                        0.0004,
                        0.0003,
                        0.0002,
                        0.0001,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0002,
                        0.0004,
                        0.0006,
                        0.0008,
                        0.001,
                        0.0009,
                        0.0008,
                        0.0007,
                        0.0006,
                        0.0005,
                        0.0004,
                        0.0003,
                        0.0002,
                        0.0001,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                ]
            ),
            atol=1e-10,
        )

    def test_act(self):
        random.seed(42)
        np.random.seed(42)
        # np.set_printoptions(precision=4, suppress=True)
        input_map = np.arange(20).reshape(4, 5)
        body_position = [2, 3]
        max_simulation_length = 20
        min_brain = brain.generate_minimal_brain()
        min_brain.neurons.loc[0, "neuron"].baseline_rate = 0.1
        min_brain.neurons.loc[1, "neuron"].baseline_rate = 0.0
        min_brain.neurons.loc[2, "neuron"].baseline_rate = 0.0
        min_brain.neurons.loc[3, "neuron"].baseline_rate = 0.0

        min_brain.neurons.loc[0, "neuron"].gain = 0.3

        min_brain.connections.loc[0, "connection"].set_amplitude(1.0)
        min_brain.connections.loc[2, "connection"].set_amplitude(1.0)

        # print(min_brain.neurons)
        # print(min_brain.connections)
        # print(f'{min_brain.neurons.loc[0, "neuron"].eye_direction=}')
        # print(f'{min_brain.neurons.loc[0, "neuron"].rf_positions=}')
        # print(f'{min_brain.neurons.loc[0, "neuron"].rf_weights=}')
        # print(f'{min_brain.neurons.loc[0, "neuron"].gain=}')
        # print(f'{min_brain.neurons.loc[0, "neuron"].input_type=}')
        # print(f'{min_brain.neurons.loc[0, "neuron"].baseline_rate=}')

        min_brain.initiate_simulation(max_simulation_length=max_simulation_length)
        movement_attempt, action_potential_num = min_brain.act(
            t_point=2, body_position=body_position, terrain_map=input_map
        )
        self.assertEqual(
            min_brain.simulation_cache["action_histories"], [[2], [], [], []]
        )
        assert np.array_equal(movement_attempt, [0, 0])
        assert action_potential_num == 1
        assert np.allclose(
            min_brain.simulation_cache["psp_waveforms"],
            np.array(
                [
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.2,
                        0.4,
                        0.6,
                        0.8,
                        1,
                        0.9,
                        0.8,
                        0.7,
                        0.6,
                        0.5,
                        0.4,
                        0.3,
                        0.2,
                        0.1,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0002,
                        0.0004,
                        0.0006,
                        0.0008,
                        0.001,
                        0.0009,
                        0.0008,
                        0.0007,
                        0.0006,
                        0.0005,
                        0.0004,
                        0.0003,
                        0.0002,
                        0.0001,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                ]
            ),
            atol=1e-10,
        )

        movement_attempt, action_potential_num = min_brain.act(
            t_point=9, body_position=body_position, terrain_map=input_map
        )
        self.assertEqual(
            min_brain.simulation_cache["action_histories"], [[2, 9], [9], [], []]
        )
        assert np.array_equal(movement_attempt, [0, 0])
        assert action_potential_num == 2

        movement_attempt, action_potential_num = min_brain.act(
            t_point=19, body_position=body_position, terrain_map=input_map
        )
        self.assertEqual(
            min_brain.simulation_cache["action_histories"],
            [[2, 9, 19], [9, 19], [], [19]],
        )
        assert np.array_equal(movement_attempt, [1, 0])
        assert action_potential_num == 3

    def test_io(self):
        import os
        import h5py

        random.seed(42)
        np.random.seed(42)
        min_brain = brain.generate_minimal_brain()

        curr_folder = os.path.dirname(os.path.abspath(__file__))
        temp_path = os.path.join(curr_folder, "temp_file.h5")

        if os.path.isfile(temp_path):
            os.remove(temp_path)

        f_temp = h5py.File(temp_path, "a")
        h5_grp = f_temp.create_group("brain")
        min_brain.to_h5_group(h5_group=h5_grp, should_save_cache=False)

        min_brain_2 = load_brain_from_h5_group(
            h5_grp, should_load_simulation_cache=False
        )

        assert np.array_equal(min_brain.neurons["layer"], min_brain_2.neurons["layer"])
        assert np.array_equal(
            min_brain.connections["pre_idx"], min_brain_2.connections["pre_idx"]
        )
        assert np.array_equal(
            min_brain.connections["post_idx"], min_brain_2.connections["post_idx"]
        )

        assert (
            min_brain.neurons.loc[0, "neuron"].type
            == min_brain_2.neurons.loc[0, "neuron"].type
        )
        assert (
            min_brain.neurons.loc[0, "neuron"].eye_direction
            == min_brain_2.neurons.loc[0, "neuron"].eye_direction
        )
        assert np.array_equal(
            min_brain.neurons.loc[0, "neuron"].rf_positions,
            min_brain_2.neurons.loc[0, "neuron"].rf_positions,
        )
        assert np.array_equal(
            min_brain.neurons.loc[0, "neuron"].rf_weights,
            min_brain_2.neurons.loc[0, "neuron"].rf_weights,
        )
        assert (
            min_brain.neurons.loc[0, "neuron"].gain
            == min_brain_2.neurons.loc[0, "neuron"].gain
        )
        assert (
            min_brain.neurons.loc[0, "neuron"].input_type
            == min_brain_2.neurons.loc[0, "neuron"].input_type
        )
        assert (
            min_brain.neurons.loc[0, "neuron"].baseline_rate
            == min_brain_2.neurons.loc[0, "neuron"].baseline_rate
        )
        assert (
            min_brain.neurons.loc[0, "neuron"].refractory_period
            == min_brain_2.neurons.loc[0, "neuron"].refractory_period
        )

        assert (
            min_brain.neurons.loc[1, "neuron"].type
            == min_brain_2.neurons.loc[1, "neuron"].type
        )
        assert (
            min_brain.neurons.loc[1, "neuron"].baseline_rate
            == min_brain_2.neurons.loc[1, "neuron"].baseline_rate
        )
        assert (
            min_brain.neurons.loc[1, "neuron"].refractory_period
            == min_brain_2.neurons.loc[1, "neuron"].refractory_period
        )

        assert (
            min_brain.neurons.loc[2, "neuron"].type
            == min_brain_2.neurons.loc[2, "neuron"].type
        )
        assert (
            min_brain.neurons.loc[2, "neuron"].baseline_rate
            == min_brain_2.neurons.loc[2, "neuron"].baseline_rate
        )
        assert (
            min_brain.neurons.loc[2, "neuron"].refractory_period
            == min_brain_2.neurons.loc[2, "neuron"].refractory_period
        )

        assert (
            min_brain.neurons.loc[3, "neuron"].type
            == min_brain_2.neurons.loc[3, "neuron"].type
        )
        assert (
            min_brain.neurons.loc[3, "neuron"].direction
            == min_brain_2.neurons.loc[3, "neuron"].direction
        )
        assert np.array_equal(
            min_brain.neurons.loc[3, "neuron"].step_motion,
            min_brain_2.neurons.loc[3, "neuron"].step_motion,
        )
        assert (
            min_brain.neurons.loc[3, "neuron"].baseline_rate
            == min_brain_2.neurons.loc[3, "neuron"].baseline_rate
        )
        assert (
            min_brain.neurons.loc[3, "neuron"].refractory_period
            == min_brain_2.neurons.loc[3, "neuron"].refractory_period
        )

        for i in range(4):
            assert (
                min_brain.connections.loc[i, "connection"].type
                == min_brain_2.connections.loc[i, "connection"].type
            )
            assert (
                min_brain.connections.loc[i, "connection"].amplitude
                == min_brain_2.connections.loc[i, "connection"].amplitude
            )
            assert (
                min_brain.connections.loc[i, "connection"].latency
                == min_brain_2.connections.loc[i, "connection"].latency
            )
            assert (
                min_brain.connections.loc[i, "connection"].rise_time
                == min_brain_2.connections.loc[i, "connection"].rise_time
            )
            assert (
                min_brain.connections.loc[i, "connection"].decay_time
                == min_brain_2.connections.loc[i, "connection"].decay_time
            )

        f_temp.close()
        os.remove(temp_path)

    def test_generate_brain_from_brain_config(self):
        import os

        curr_folder = os.path.dirname(os.path.realpath(__file__))

        brain_config_path_min = os.path.join(curr_folder, "brain_config_minimal.yml")
        brain_min = generate_brain_from_brain_config(
            brain_config_path=brain_config_path_min
        )

        brain_config_path_4eyes_ff = os.path.join(
            curr_folder, "brain_config_4eyes_feedforward.yml"
        )
        brain_4eyes_ff = generate_brain_from_brain_config(
            brain_config_path=brain_config_path_4eyes_ff
        )

        brain_config_path_8eyes_recur = os.path.join(
            curr_folder, "brain_config_8eyes_recurrent.yml"
        )
        brain_8eyes_recur = generate_brain_from_brain_config(
            brain_config_path=brain_config_path_8eyes_recur
        )

    def test_plot_brain_connections(self):
        import os

        curr_folder = os.path.dirname(os.path.realpath(__file__))
        brain_config_path_8eyes_recur = os.path.join(
            curr_folder, "brain_config_8eyes_recurrent.yml"
        )
        brain_8eyes_recur = generate_brain_from_brain_config(
            brain_config_path=brain_config_path_8eyes_recur
        )
        f = plot_brain_connections(brain=brain_8eyes_recur)


if __name__ == "__main__":
    test_brain = TestBrain()
    test_brain.test_generate_minimal_brain()
    test_brain.test_neuron_fire()
    test_brain.test_act()
    test_brain.test_io()
    test_brain.test_generate_brain_from_brain_config()
    test_brain.test_plot_brain_connections()
