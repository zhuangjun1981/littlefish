import unittest
import random
import littlefish.core.fish as fi
import littlefish.core.evolution as evo


class TestEvolution(unittest.TestCase):
    def setup(self):
        pass

    def test_UniformMutation(self):
        um = evo.UniformMutation(value_range=(5, 10), dtype='float')
        assert (isinstance(um.get_value(), float))
        assert (um.get_value() >= 5.)
        assert (um.get_value() < 10.)

        um2 = evo.UniformMutation(value_range=(6., 11.), dtype='int')
        assert (isinstance(um2.get_value(), int))
        assert (um2.get_value() >= 6)
        assert (um2.get_value() < 11)

    def test_mutate_neuron(self):
        curr_neuron = fi.Neuron(baseline_rate=0.003, refractory_period=1.2)
        um_n = evo.NeuronMutation(baseline_mutation=evo.UniformMutation(value_range=(0., 0.05), dtype='float'),
                                  refractory_mutation=evo.UniformMutation(value_range=(1.3, 2.5), dtype='float'))
        mutated_neuron = evo.mutate_neuron(neuron=curr_neuron, neuron_mutation=um_n)
        assert (mutated_neuron.get_baseline_rate() >= 0.)
        assert (mutated_neuron.get_baseline_rate() < 0.05)
        assert (mutated_neuron.get_refractory_period() > 1.3)
        assert (mutated_neuron.get_refractory_period() <= 2.5)

        um_n2 = evo.NeuronMutation(baseline_mutation=evo.UniformMutation(value_range=(0., 0.05), dtype='float'),
                                   refractory_mutation=None)
        mutated_neuron2 = evo.mutate_neuron(neuron=curr_neuron, neuron_mutation=um_n2)
        assert (mutated_neuron2.get_refractory_period() == 1.2)

    def test_mutate_connection(self):
        curr_connection = fi.Connection(latency=3, amplitude=0.001, rise_time=5, decay_time=10)
        um_con = evo.ConnectionMutation(latency_mutation=None,
                                        amplitude_mutation=evo.UniformMutation(value_range=(0.0001, 0.005),
                                                                               dtype='float'),
                                        rise_time_mutation=None, decay_time_mutation=None)
        mutated_connection = evo.mutate_connection(connection=curr_connection, connection_mutation=um_con)
        assert (mutated_connection.get_latency() == 3)
        assert (mutated_connection.get_amplitude() >= 0.0001)
        assert (mutated_connection.get_amplitude() < 0.005)
        assert (mutated_connection.get_rise_time() == 5)
        assert (mutated_connection.get_decay_time() == 10)

    def test_choose_index_1d(self):
        lst = range(10)
        sub_lst = evo.choose_index_1d(lst, 0.35)
        assert (len(sub_lst) == 4)
        for item in sub_lst:
            assert (item in lst)

    def test_choose_index_2d(self):
        lst0 = range(5)
        lst1 = range(15, 20)
        sub_lst = evo.choose_index_2d(indices0=lst0, indices1=lst1, mutation_rate=0.5)
        assert (len(sub_lst) == 13)

        sub_lst0 = [coor[0] for coor in sub_lst]
        for sub_ind0 in sub_lst0:
            assert (sub_ind0 in lst0)

        sub_lst1 = [coor[1] for coor in sub_lst]
        for sub_ind1 in sub_lst1:
            assert (sub_ind1 in lst1)

    def test_mutate_brain(self):
        random.seed(5)
        brain = fi.generate_standard_fish().get_brain()
        eye_mutation = evo.NeuronMutation(baseline_mutation=evo.UniformMutation((0.1, 0.5), 'float'),
                                          refractory_mutation=evo.UniformMutation((5., 10.), 'float'))
        neuron_mutation = evo.NeuronMutation(baseline_mutation=evo.UniformMutation((0.01, 0.03), 'float'),
                                             refractory_mutation=None)
        muscle_mutation = evo.NeuronMutation(baseline_mutation=None,
                                             refractory_mutation=evo.UniformMutation((13., 14.5), 'float'))
        connection_mutation = evo.ConnectionMutation(latency_mutation=evo.UniformMutation((15, 20), 'int'),
                                                     amplitude_mutation=evo.UniformMutation((0.5, 0.6), 'float'),
                                                     rise_time_mutation=None,
                                                     decay_time_mutation=None)
        bm = evo.BrainMutation(neuron_mutation_rate=0.5,
                               eye_mutation=eye_mutation,
                               neuron_mutation=neuron_mutation,
                               muscle_mutation=muscle_mutation,
                               connection_mutation_rate=0.1,
                               connection_mutation=connection_mutation)
        mutated_brain = evo.mutate_brain(brain=brain, brain_mutation=bm, verbose=True)

        assert (mutated_brain.get_neurons().loc[0, 'neuron'].get_baseline_rate() >= 0.1)
        assert (mutated_brain.get_neurons().loc[0, 'neuron'].get_baseline_rate() < 0.5)
        assert (mutated_brain.get_neurons().loc[0, 'neuron'].get_refractory_period() >= 5.)
        assert (mutated_brain.get_neurons().loc[0, 'neuron'].get_refractory_period() < 10.)
        assert (mutated_brain.get_neurons().loc[12, 'neuron'].get_baseline_rate() >= 0.01)
        assert (mutated_brain.get_neurons().loc[12, 'neuron'].get_baseline_rate() < 0.03)
        assert (mutated_brain.get_neurons().loc[12, 'neuron'].get_refractory_period() == 1.2)
        assert (mutated_brain.get_neurons().loc[19, 'neuron'].get_baseline_rate() == 0.0001)
        assert (mutated_brain.get_neurons().loc[19, 'neuron'].get_refractory_period() >= 13.)
        assert (mutated_brain.get_neurons().loc[19, 'neuron'].get_refractory_period() < 14.5)

        test_conn = mutated_brain.get_connections()['L000_L001'].loc[9, 0]
        assert (test_conn.get_latency() >= 15)
        assert (test_conn.get_latency() < 20)
        assert (test_conn.get_amplitude() >= 0.5)
        assert (test_conn.get_amplitude() < 0.6)
        assert (test_conn.get_rise_time() == 2)
        assert (test_conn.get_decay_time() == 5)


if __name__ == '__main__':
    te = TestEvolution()
    te.test_UniformMutation()
    te.test_mutate_neuron()
    te.test_mutate_connection()
    te.test_choose_index_1d()
    te.test_choose_index_2d()
    te.test_mutate_brain()
