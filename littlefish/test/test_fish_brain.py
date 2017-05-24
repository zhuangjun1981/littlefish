import random
import unittest
import littlefish.core.fish as fi
import littlefish.core.utilities as util
import numpy as np


class TestFishBrain(unittest.TestCase):

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
        world_map = np.zeros((5, 5), dtype=np.uint8)
        world_map[3, 3] = 1
        world_map[2, 2] = 1
        world_map[2, 1] = 1

        eye = fi.Eye(direction='south')
        assert(np.array_equal(eye._get_input_pixels(position=(2, 3), input_map=world_map), [0, 1, 0]))
        eye = fi.Eye(direction='southeast')
        assert (np.array_equal(eye._get_input_pixels(position=(2, 3), input_map=world_map), [1, 0, 0]))
        eye = fi.Eye(direction='east')
        assert (np.array_equal(eye._get_input_pixels(position=(3, 2), input_map=world_map), [0, 1, 0]))
        eye = fi.Eye(direction='northeast')
        assert (np.array_equal(eye._get_input_pixels(position=(3, 2), input_map=world_map), [1, 0, 1]))
        eye = fi.Eye(direction='north')
        assert (np.array_equal(eye._get_input_pixels(position=(3, 2), input_map=world_map), [0, 1, 1]))
        eye = fi.Eye(direction='northwest')
        assert (np.array_equal(eye._get_input_pixels(position=(3, 2), input_map=world_map), [1, 1, 0]))
        eye = fi.Eye(direction='west')
        assert (np.array_equal(eye._get_input_pixels(position=(3, 2), input_map=world_map), [1, 0, 0]))
        eye = fi.Eye(direction='southwest')
        assert (np.array_equal(eye._get_input_pixels(position=(3, 2), input_map=world_map), [0, 0, 0]))
        eye = fi.Eye(direction='north')
        assert (np.array_equal(eye._get_input_pixels(position=(0, 0), input_map=world_map), [1, 1, 1]))
        eye = fi.Eye(direction='northeast')
        assert (np.array_equal(eye._get_input_pixels(position=(0, 0), input_map=world_map), [0, 1, 1]))
        eye = fi.Eye(direction='east')
        assert (np.array_equal(eye._get_input_pixels(position=(0, 0), input_map=world_map), [0, 0, 1]))

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

    def test_brain_default(self):
        brain1 = fi.Brain()
        assert(len(brain1.get_neurons()) == 20)
        assert(len(brain1.get_connections()) == 2)
        assert(brain1.get_connections()['L000_L001'].shape[0] == 8)
        assert(brain1.get_connections()['L000_L001'].shape[1] == 8)
        assert(brain1.get_connections()['L001_L002'].shape[0] == 4)
        assert(brain1.get_connections()['L001_L002'].shape[1] == 8)

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
        assert(action_histories.iloc[0, 0] == [53])
        assert(action_histories.iloc[3, 0] == [6])

    def test_brain_default(self):
        default_brain = fi.Brain()

    def test_brain_generate_empty_action_histories(self):
        db = fi.Brain()
        eah = db.generate_empty_action_histories()
        eah.loc[3, 'action_history'].append(10)
        assert(len(eah.loc[4, 'action_history']) == 0)
        assert(len(eah.loc[3, 'action_history']) == 1)

    def test_brain_get_eye_type(self):
        assert (fi.Brain.get_eye_type(0) == ('east', 'terrain'))
        assert (fi.Brain.get_eye_type(1) == ('northeast', 'terrain'))
        assert (fi.Brain.get_eye_type(2) == ('north', 'terrain'))
        assert (fi.Brain.get_eye_type(3) == ('northwest', 'terrain'))
        assert (fi.Brain.get_eye_type(4) == ('west', 'terrain'))
        assert (fi.Brain.get_eye_type(5) == ('southwest', 'terrain'))
        assert (fi.Brain.get_eye_type(6) == ('south', 'terrain'))
        assert (fi.Brain.get_eye_type(7) == ('southeast', 'terrain'))
        assert (fi.Brain.get_eye_type(8) == ('east', 'food'))
        assert (fi.Brain.get_eye_type(9) == ('northeast', 'food'))
        assert (fi.Brain.get_eye_type(10) == ('north', 'food'))
        assert (fi.Brain.get_eye_type(11) == ('northwest', 'food'))
        assert (fi.Brain.get_eye_type(12) == ('west', 'food'))
        assert (fi.Brain.get_eye_type(13) == ('southwest', 'food'))
        assert (fi.Brain.get_eye_type(14) == ('south', 'food'))
        assert (fi.Brain.get_eye_type(15) == ('southeast', 'food'))
        assert (fi.Brain.get_eye_type(16) == ('east', 'fish'))
        assert (fi.Brain.get_eye_type(17) == ('northeast', 'fish'))
        assert (fi.Brain.get_eye_type(18) == ('north', 'fish'))
        assert (fi.Brain.get_eye_type(19) == ('northwest', 'fish'))
        assert (fi.Brain.get_eye_type(20) == ('west', 'fish'))
        assert (fi.Brain.get_eye_type(21) == ('southwest', 'fish'))
        assert (fi.Brain.get_eye_type(22) == ('south', 'fish'))
        assert (fi.Brain.get_eye_type(23) == ('southeast', 'fish'))
        assert (fi.Brain.get_eye_type(24) == ('east', 'terrain'))



if __name__ == '__main__':

    tfb = TestFishBrain()

    tfb.test_connection_psp()
    tfb.test_connection_act()
    tfb.test_eye_get_input_pixels()
    tfb.test_eye2_get_input_pixels()
    tfb.test_neuron_connection()
    tfb.test_muscle_action()
    tfb.test_brain_default()
    tfb.test_brain_minimum_brain()
