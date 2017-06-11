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
    
    def test_neuron1(self):
        total_t = 10
        neuron = fi.Neuron(baseline_rate=0.5, refractory_period=1.2)
        action_history = []
        for t_point in range(total_t):
            neuron.act(t_point=t_point, action_history=action_history, probability_input=0.5)
        assert (action_history == [0, 2, 4, 6, 8])

    def test_connection_psp(self):
        connection = fi.Connection(amplitude=10, latency=5, rise_time=5, decay_time=10)
        assert(np.array_equal(connection.get_psp(), np.array([0., 0., 0., 0., 0., 2., 4., 6., 8., 10., 9., 8.,
                                                              7., 6., 5., 4., 3., 2., 1., 0.])))

    def test_connection_act(self):
        simulation_length = 50
        connection = fi.Connection(amplitude=10, latency=5, rise_time=5, decay_time=10)
        psp_waveforms = np.zeros((1, simulation_length))
        connection.act(t_point=2, postsynaptic_index=0, psp_waveforms=psp_waveforms)
        assert(np.array_equal(psp_waveforms[0],
                              np.array([0., 0., 0., 0., 0., 0., 0., 2., 4., 6., 8., 10., 9., 8.,
                                        7., 6., 5., 4., 3., 2., 1., 0., 0., 0., 0., 0., 0., 0.,
                                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                        0., 0., 0., 0., 0., 0., 0., 0.])))
        connection.act(t_point=4, postsynaptic_index=0, psp_waveforms=psp_waveforms)
        assert (np.array_equal(psp_waveforms[0],
                               np.array([0., 0., 0., 0., 0., 0., 0., 2., 4., 8., 12., 16., 17., 18.,
                                         16., 14., 12., 10., 8., 6., 4., 2., 1., 0., 0.,
                                         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])))
        connection.act(t_point=40, postsynaptic_index=0, psp_waveforms=psp_waveforms)
        assert (np.array_equal(psp_waveforms[0],
                               np.array([0., 0., 0., 0., 0., 0., 0., 2., 4., 8., 12., 16., 17.,
                                         18., 16., 14., 12., 10., 8., 6., 4., 2., 1., 0., 0.,
                                         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                         0., 0., 0., 0., 0., 0., 0., 2., 4., 6., 8., 10.])))
        connection.act(t_point=50, postsynaptic_index=0, psp_waveforms=psp_waveforms)
        assert (np.array_equal(psp_waveforms[0],
                               np.array([0., 0., 0., 0., 0., 0., 0., 2., 4., 8., 12., 16., 17.,
                                         18., 16., 14., 12., 10., 8., 6., 4., 2., 1., 0., 0.,
                                         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                         0., 0., 0., 0., 0., 0., 0., 2., 4., 6., 8., 10.])))

    def test_eye_get_input_pixels(self):
        world_map = np.zeros((7, 8), dtype=np.uint8)
        world_map[3, 3] = 1
        world_map[2, 2] = 1
        world_map[2, 1] = 1
        world_map[5, 2] = 1
        world_map[3, 4] = 1
        world_map[4, 3] = 1
        world_map[1, 5] = 1
        world_map[0, 3] = 1
        world_map[4:6, 5:8] = 1

        eye = fi.Eye(direction='south')
        assert(np.array_equal(eye._get_input_pixels(body_position=(3, 3), input_map=world_map, border_value=1),
                              [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]))
        eye = fi.Eye(direction='east')
        assert (np.array_equal(eye._get_input_pixels(body_position=(3, 3), input_map=world_map, border_value=1),
                              [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0]))
        eye = fi.Eye(direction='north')
        assert (np.array_equal(eye._get_input_pixels(body_position=(3, 3), input_map=world_map, border_value=1),
                               [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]))
        eye = fi.Eye(direction='west')
        assert (np.array_equal(eye._get_input_pixels(body_position=(3, 3), input_map=world_map, border_value=1),
                               [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        eye = fi.Eye(direction='north')
        assert (np.array_equal(eye._get_input_pixels(body_position=(1, 1), input_map=world_map, border_value=1),
                               [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
        eye = fi.Eye(direction='south')
        assert (np.array_equal(eye._get_input_pixels(body_position=(1, 1), input_map=world_map, border_value=1),
                               [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0]))
        eye = fi.Eye(direction='east')
        assert (np.array_equal(eye._get_input_pixels(body_position=(1, 1), input_map=world_map, border_value=1),
                               [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0]))

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
                connection.act(t_point=i, postsynaptic_index=0, psp_waveforms=psp_waveforms)
            neuron_post.act(i, action_history=action_history_post, probability_input=psp_waveforms[0, i])

        ccg, t = util.discreat_crosscorrelation(np.array(action_history_pre), np.array(action_history_post))
        assert(np.argmax(ccg) == 15)

    def test_muscle_action(self):
        simulation_length = 20000
        muscle = fi.Muscle(direction='east', baseline_rate=0., refractory_period=5000)
        action_history_muscle = []
        movements = []
        for i in range(simulation_length):
            movement = muscle.act(i, action_history=action_history_muscle, probability_input=0.5, probability_base=0.1)
            if movement is not False:
                movements.append(movement)
        target_movements = [np.array([0, 1], dtype=np.uint8), np.array([0, 1], dtype=np.uint8),
                            np.array([0, 1], dtype=np.uint8), np.array([0, 1], dtype=np.uint8)]
        assert(all([np.array_equal(movements[i], target_movements[i]) for i in range(4)]))
        assert(action_history_muscle == [0, 5000, 10000, 15000])

    def test_brain_minimum_brain(self):

        simulation_length = int(1e2)
        random.seed(111)

        minimum_brain = fi.generate_minimal_brain()
        terrain_map = np.zeros((10, 10), dtype=np.uint8)
        terrain_map[2:4, 4:6] = 1

        action_histories = minimum_brain.generate_empty_action_histories()
        psp_waveforms = minimum_brain.generate_empty_psp_waveforms(simulation_length=simulation_length)
        body_position = np.array([3, 2], dtype=np.uint8)

        for i in range(simulation_length):

            movement = minimum_brain.act(t_point=i, body_position=body_position, action_histories=action_histories,
                                         psp_waveforms=psp_waveforms, terrain_map=terrain_map)
            if not np.array_equal(movement, [0, 0]):
                body_position = body_position + movement
        assert(action_histories.iloc[0, 0] == [53, 75, 93])
        assert(action_histories.iloc[3, 0] == [6])

    def test_brain_generate_empty_action_histories(self):
        db = fi.Brain()
        eah = db.generate_empty_action_histories()
        eah.loc[3, 'action_history'].append(10)
        assert(len(eah.loc[0, 'action_history']) == 0)
        assert(len(eah.loc[3, 'action_history']) == 1)

    def test_get_eye_type(self):
        assert (fi.get_eye_type(0, 8) == ('east', 'terrain'))
        assert (fi.get_eye_type(1, 8) == ('northeast', 'terrain'))
        assert (fi.get_eye_type(2, 8) == ('north', 'terrain'))
        assert (fi.get_eye_type(3, 8) == ('northwest', 'terrain'))
        assert (fi.get_eye_type(4, 8) == ('west', 'terrain'))
        assert (fi.get_eye_type(5, 8) == ('southwest', 'terrain'))
        assert (fi.get_eye_type(6, 8) == ('south', 'terrain'))
        assert (fi.get_eye_type(7, 8) == ('southeast', 'terrain'))
        assert (fi.get_eye_type(8, 8) == ('east', 'food'))
        assert (fi.get_eye_type(9, 8) == ('northeast', 'food'))
        assert (fi.get_eye_type(10, 8) == ('north', 'food'))
        assert (fi.get_eye_type(11, 8) == ('northwest', 'food'))
        assert (fi.get_eye_type(12, 8) == ('west', 'food'))
        assert (fi.get_eye_type(13, 8) == ('southwest', 'food'))
        assert (fi.get_eye_type(14, 8) == ('south', 'food'))
        assert (fi.get_eye_type(15, 8) == ('southeast', 'food'))
        assert (fi.get_eye_type(16, 8) == ('east', 'fish'))
        assert (fi.get_eye_type(17, 8) == ('northeast', 'fish'))
        assert (fi.get_eye_type(18, 8) == ('north', 'fish'))
        assert (fi.get_eye_type(19, 8) == ('northwest', 'fish'))
        assert (fi.get_eye_type(20, 8) == ('west', 'fish'))
        assert (fi.get_eye_type(21, 8) == ('southwest', 'fish'))
        assert (fi.get_eye_type(22, 8) == ('south', 'fish'))
        assert (fi.get_eye_type(23, 8) == ('southeast', 'fish'))
        assert (fi.get_eye_type(24, 8) == ('east', 'terrain'))
        assert (fi.get_eye_type(0, 4) == ('east', 'terrain'))
        assert (fi.get_eye_type(1, 4) == ('north', 'terrain'))
        assert (fi.get_eye_type(2, 4) == ('west', 'terrain'))
        assert (fi.get_eye_type(3, 4) == ('south', 'terrain'))
        assert (fi.get_eye_type(4, 4) == ('east', 'food'))
        assert (fi.get_eye_type(5, 4) == ('north', 'food'))
        assert (fi.get_eye_type(6, 4) == ('west', 'food'))
        assert (fi.get_eye_type(7, 4) == ('south', 'food'))
        assert (fi.get_eye_type(8, 4) == ('east', 'fish'))
        assert (fi.get_eye_type(9, 4) == ('north', 'fish'))
        assert (fi.get_eye_type(10, 4) == ('west', 'fish'))
        assert (fi.get_eye_type(11, 4) == ('south', 'fish'))
        assert (fi.get_eye_type(12, 4) == ('east', 'terrain'))

    def test_get_muscle_direction(self):
        assert (fi.get_muscle_direction(0) == 'east')
        assert (fi.get_muscle_direction(1) == 'north')
        assert (fi.get_muscle_direction(2) == 'west')
        assert (fi.get_muscle_direction(3) == 'south')
        assert (fi.get_muscle_direction(4) == 'east')

    def test_generate_standard_fish(self):
        fi.generate_standard_fish()

    def test_load_brain(self):
        curr_folder = os.path.dirname(os.path.realpath(__file__))
        example_log_f = h5py.File(os.path.join(curr_folder, 'example_simulation_log.hdf5'))
        brain_grp = example_log_f['fish_test_fish/brain']
        curr_brain = fi.Brain.from_h5_group(brain_grp)
        assert (curr_brain.get_neurons().loc[0, 'neuron']._gain == 0.005)
        assert (curr_brain.get_neurons().loc[0, 'neuron']._baseline_rate == 0.)
        assert (curr_brain.get_neurons().loc[0, 'neuron']._refractory_period == 1.2)
        assert (curr_brain._connections['L001_L002'].loc[16, 8]._amplitude == 0.001)
        assert (curr_brain._connections['L001_L002'].loc[16, 8]._latency == 3)
        assert (curr_brain._connections['L001_L002'].loc[16, 8]._rise_time == 2)
        assert (curr_brain._connections['L001_L002'].loc[16, 8]._decay_time == 5)
        example_log_f.close()

    def test_load_fish(self):
        curr_folder = os.path.dirname(os.path.realpath(__file__))
        example_log_f = h5py.File(os.path.join(curr_folder, 'example_simulation_log.hdf5'))
        fish_grp = example_log_f['fish_test_fish']
        curr_fish = fi.Fish.from_h5_group(fish_grp)
        assert (curr_fish._food_rate == 10.)
        assert (curr_fish._health_decay_rate == 0.001)
        assert (curr_fish._land_penalty_rate == 0.01)
        assert (curr_fish._max_health == 100.)
        assert (curr_fish._mother_name == '')
        assert (curr_fish._name == 'test_fish')
        example_log_f.close()


if __name__ == '__main__':

    tfb = TestFish()

    tfb.test_neuron1()
    tfb.test_connection_psp()
    tfb.test_connection_act()
    tfb.test_eye_get_input_pixels()
    tfb.test_neuron_connection()
    tfb.test_muscle_action()
    tfb.test_brain_minimum_brain()
    tfb.test_brain_generate_empty_action_histories()
    tfb.test_get_eye_type()
    tfb.test_get_muscle_direction()
    tfb.test_generate_standard_fish()
    tfb.test_load_brain()
    tfb.test_load_fish()
