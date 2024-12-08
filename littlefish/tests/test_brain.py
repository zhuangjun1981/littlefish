import unittest
import random
import numpy as np
import littlefish.brain.brain as brain


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
        np.set_printoptions(precision=4, suppress=True)
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


if __name__ == "__main__":
    test_brain = TestBrain()
    # test_brain.test_generate_minimal_brain()
    # test_brain.test_neuron_fire()
    test_brain.test_act()
