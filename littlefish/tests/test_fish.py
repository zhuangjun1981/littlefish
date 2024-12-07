import os
import h5py
import random
import unittest
import littlefish.core.fish as fi
import littlefish.core.utilities as util
import numpy as np


class TestFish(unittest.TestCase):
    def setup(self):
        pass

    def test_neuron_connection(self):
        simulation_length = 5000
        neuron_pre = fi.Neuron(baseline_rate=0.005)
        neuron_post = fi.Neuron(baseline_rate=0.000)
        connection = fi.Connection(amplitude=1, latency=5, rise_time=1, decay_time=1)

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

        ccg, t = util.discreat_crosscorrelation(
            np.array(action_history_pre), np.array(action_history_post)
        )
        assert np.argmax(ccg) == 15

    def test_brain_minimum_brain(self):
        simulation_length = int(1e2)
        random.seed(111)

        minimum_brain = fi.generate_minimal_brain()
        terrain_map = np.zeros((10, 10), dtype=np.uint8)
        terrain_map[2:4, 4:6] = 1

        action_histories = minimum_brain.generate_empty_action_histories()
        psp_waveforms = minimum_brain.generate_empty_psp_waveforms(
            simulation_length=simulation_length
        )
        body_position = np.array([3, 2], dtype=np.uint8)

        for i in range(simulation_length):
            movement = minimum_brain.act(
                t_point=i,
                body_position=body_position,
                action_histories=action_histories,
                psp_waveforms=psp_waveforms,
                terrain_map=terrain_map,
            )
            if not np.array_equal(movement, [0, 0]):
                body_position = body_position + movement
        assert action_histories.iloc[0, 0] == [53, 75, 93]
        assert action_histories.iloc[3, 0] == [6]

    def test_brain_generate_empty_action_histories(self):
        db = fi.Brain()
        eah = db.generate_empty_action_histories()
        eah.loc[3, "action_history"].append(10)
        assert len(eah.loc[0, "action_history"]) == 0
        assert len(eah.loc[3, "action_history"]) == 1

    def test_generate_standard_fish(self):
        fi.generate_standard_fish()

    def test_load_brain(self):
        curr_folder = os.path.dirname(os.path.realpath(__file__))
        example_log_f = h5py.File(
            os.path.join(curr_folder, "example_simulation_log.hdf5"), "r"
        )
        brain_grp = example_log_f["fish_test_fish/brain"]
        curr_brain = fi.Brain.from_h5_group(brain_grp)
        assert curr_brain.get_neurons().loc[0, "neuron"]._gain == 0.005
        assert curr_brain.get_neurons().loc[0, "neuron"]._baseline_rate == 0.0
        assert curr_brain.get_neurons().loc[0, "neuron"]._refractory_period == 1.2
        assert curr_brain._connections["L001_L002"].loc[16, 8]._amplitude == 0.001
        assert curr_brain._connections["L001_L002"].loc[16, 8]._latency == 3
        assert curr_brain._connections["L001_L002"].loc[16, 8]._rise_time == 2
        assert curr_brain._connections["L001_L002"].loc[16, 8]._decay_time == 5
        example_log_f.close()

    def test_load_fish(self):
        curr_folder = os.path.dirname(os.path.realpath(__file__))
        example_log_f = h5py.File(
            os.path.join(curr_folder, "example_simulation_log.hdf5"), "r"
        )
        fish_grp = example_log_f["fish_test_fish"]
        curr_fish = fi.Fish.from_h5_group(fish_grp)
        assert curr_fish._food_rate == 10.0
        assert curr_fish._health_decay_rate == 0.001
        assert curr_fish._land_penalty_rate == 0.01
        assert curr_fish._max_health == 100.0
        assert curr_fish._mother_name == ""
        assert curr_fish._name == "test_fish"
        example_log_f.close()


if __name__ == "__main__":
    tfb = TestFish()
    tfb.test_neuron_connection()
    tfb.test_brain_minimum_brain()
    tfb.test_brain_generate_empty_action_histories()
    tfb.test_generate_standard_fish()
    tfb.test_load_brain()
    tfb.test_load_fish()
