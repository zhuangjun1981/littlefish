import unittest
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

    def test_mutate_connectio(self):
        curr_connection = fi.Connection(latency=3, amplitude=0.001, rise_time=5, decay_time=10)
        um_con = evo.ConnectionMutation(latency_mutation=None,
                                        amplitude_mutation=evo.UniformMutation(value_range=(0.0001, 0.005),
                                                                               dtype='float'),
                                        rise_time_mutation=None, decay_time_mutation=None)
        mutated_connection = evo.mutation_connection(connection=curr_connection, connection_mutation=um_con)
        assert (mutated_connection.get_latency() == 3)
        assert (mutated_connection.get_amplitude() >= 0.0001)
        assert (mutated_connection.get_amplitude() < 0.005)
        assert (mutated_connection.get_rise_time() == 5)
        assert (mutated_connection.get_decay_time() == 10)


if __name__ == '__main__':
    te = TestEvolution()
    te.test_UniformMutation()
    te.test_mutate_neuron()
    te.test_mutate_connectio()