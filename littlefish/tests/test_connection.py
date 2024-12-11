import unittest
import numpy as np
import littlefish.core.utilities as util
from littlefish.brain.neuron import Neuron
from littlefish.brain.brain import Connection


class TestConnection(unittest.TestCase):
    def setup(self):
        pass

    def test_connection_psp(self):
        connection = Connection(amplitude=10, latency=5, rise_time=5, decay_time=10)
        assert np.array_equal(
            connection.psp,
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.0,
                    4.0,
                    6.0,
                    8.0,
                    10.0,
                    9.0,
                    8.0,
                    7.0,
                    6.0,
                    5.0,
                    4.0,
                    3.0,
                    2.0,
                    1.0,
                    0.0,
                ]
            ),
        )

    def test_connection_act(self):
        simulation_length = 50
        connection = Connection(amplitude=10, latency=5, rise_time=5, decay_time=10)
        psp_waveforms = np.zeros((1, simulation_length))
        connection.act(t_point=2, postsynaptic_index=0, psp_waveforms=psp_waveforms)
        assert np.array_equal(
            psp_waveforms[0],
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.0,
                    4.0,
                    6.0,
                    8.0,
                    10.0,
                    9.0,
                    8.0,
                    7.0,
                    6.0,
                    5.0,
                    4.0,
                    3.0,
                    2.0,
                    1.0,
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
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        )
        connection.act(t_point=4, postsynaptic_index=0, psp_waveforms=psp_waveforms)
        assert np.array_equal(
            psp_waveforms[0],
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.0,
                    4.0,
                    8.0,
                    12.0,
                    16.0,
                    17.0,
                    18.0,
                    16.0,
                    14.0,
                    12.0,
                    10.0,
                    8.0,
                    6.0,
                    4.0,
                    2.0,
                    1.0,
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
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        )
        connection.act(t_point=40, postsynaptic_index=0, psp_waveforms=psp_waveforms)
        assert np.array_equal(
            psp_waveforms[0],
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.0,
                    4.0,
                    8.0,
                    12.0,
                    16.0,
                    17.0,
                    18.0,
                    16.0,
                    14.0,
                    12.0,
                    10.0,
                    8.0,
                    6.0,
                    4.0,
                    2.0,
                    1.0,
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
                    0.0,
                    0.0,
                    2.0,
                    4.0,
                    6.0,
                    8.0,
                    10.0,
                ]
            ),
        )
        connection.act(t_point=50, postsynaptic_index=0, psp_waveforms=psp_waveforms)
        assert np.array_equal(
            psp_waveforms[0],
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.0,
                    4.0,
                    8.0,
                    12.0,
                    16.0,
                    17.0,
                    18.0,
                    16.0,
                    14.0,
                    12.0,
                    10.0,
                    8.0,
                    6.0,
                    4.0,
                    2.0,
                    1.0,
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
                    0.0,
                    0.0,
                    2.0,
                    4.0,
                    6.0,
                    8.0,
                    10.0,
                ]
            ),
        )

    def test_neuron_connection(self):
        simulation_length = 5000
        neuron_pre = Neuron(baseline_rate=0.005)
        neuron_post = Neuron(baseline_rate=0.000)
        connection = Connection(amplitude=1, latency=5, rise_time=1, decay_time=1)

        psp_waveforms = np.zeros((1, simulation_length))
        action_history_pre = []
        action_history_post = []

        for i in range(simulation_length):
            is_firing = neuron_pre.act(i, action_history=action_history_pre)
            if is_firing:
                connection.act(
                    t_point=i, postsynaptic_index=0, psp_waveforms=psp_waveforms
                )
            neuron_post.act(
                i,
                action_history=action_history_post,
                probability_input=psp_waveforms[0, i],
            )

        ccg, t = util.discrete_crosscorrelation(
            np.array(action_history_pre), np.array(action_history_post)
        )
        assert np.argmax(ccg) == 15


if __name__ == "__main__":
    connection_tests = TestConnection()
    connection_tests.test_connection_psp()
    connection_tests.test_connection_act()
    connection_tests.test_neuron_connection()
