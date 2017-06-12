import random
import littlefish.core.fish as fi


def mutate_neuron(neuron, neuron_mutation):
    """
    mutate a neuron, can be Eye, Neuron or Muscle
    :param neuron: the initial little_fish.core.fish.Neuron object
    :param neuron_mutation: little_fish.core.evolution.NeuronMutation object
    :return: a mutated little_fish.core.fish.Neuron object 
    """

    mutated_neuron = neuron.copy()
    
    mutated_baseline = neuron_mutation.get_mutated_baseline()
    if mutated_baseline is not None:
        mutated_neuron.set_baseline_rate(mutated_baseline)
    
    mutated_refractory = neuron_mutation.get_mutated_refractory()
    if mutated_refractory is not None:
        mutated_neuron.set_refractory_period(mutated_refractory)

    return mutated_neuron
    

def mutation_connection(connection, connection_mutation):
    """
    mutate a connection
    :param connection: the initial little_fish.core.fish.Connection object
    :param connection_mutation: little_fish.core.evolution.ConnectionMutation object
    :return: a mutated little_fish.core.fish.Connection object
    """

    mutated_latency = connection_mutation.get_mutated_latency()
    if mutated_latency is None:
        mutated_latency = connection.get_latency()

    mutated_amplitude = connection_mutation.get_mutated_amplitude()
    if mutated_amplitude is None:
        mutated_amplitude = connection.get_amplitude()

    mutated_rise_time = connection_mutation.get_mutated_rise_time()
    if mutated_rise_time is None:
        mutated_rise_time = connection.get_rise_time()

    mutated_decay_time = connection_mutation.get_mutated_decay_time()
    if mutated_decay_time is None:
        mutated_decay_time = connection.get_decay_time()

    return fi.Connection(latency=mutated_latency, amplitude=mutated_amplitude, rise_time=mutated_rise_time,
                         decay_time=mutated_decay_time)


class UniformMutation(object):
    """
    definition of a single mutation of a single value, based on a uniform distribution of a value range. uses builtin
    random module
    """

    def __init__(self, value_range, dtype):
        """
        :param value_range: tuple of two numbers, the two value should be different.
        :param dtype: str, 'int' or 'float'. if 'int' random value will be drawn by random.randint()
                                             if 'float' random value will be drawn by random.uniform()
        """

        if len(value_range) != 2:
            raise ValueError('the input _value_range should be sequence with length of 2.')

        if dtype == 'int':
            v0 = int(value_range[0])
            v1 = int(value_range[1])
            self._dtype = 'int'
        elif dtype == 'float':
            v0 = float(value_range[0])
            v1 = float(value_range[1])
            self._dtype = 'float'
        else:
            raise ValueError('the _dtype shoule be either "int" or "float".')

        if v0 == v1:
            raise ValueError('the two values in the input _value_range should be different.')
        elif v0 < v1:
            self._value_range = (v0, v1)
        else:
            self._value_range = (v0, v1)

    def get_value(self):
        """
        return: a random value follow a uniform distribution with a range defined by self._value_range, including the
        start but excluding the end
        if self._dtype is 'int': uses random.randint() function
        if self._dtype is 'float': uses random.uniform() function
        """

        if self._dtype == 'int':
            return random.randint(self._value_range[0], self._value_range[1] - 1)
        elif self._dtype == 'float':
            return random.uniform(self._value_range[0], self._value_range[1])

    def get_dtype(self):
        return self._dtype

    def get_value_range(self):
        return self._value_range

    def __str__(self):
        return 'littlefish.core.evolution.UniformMutation object. dtype:{}; value_range:{}'\
            .format(self._dtype, self._value_range)


class NeuronMutation(object):

    """
    definition of a neuron mutation
    """

    def __init__(self, baseline_mutation=None, refractory_mutation=None):
        """
        :param baseline_mutation: a UniformMutation object, dtype should be 'float',
                                  reasonable value_range will be (0., 0.1), if one time unit is equivalent to 1 ms,
                                  then this range represents (0, 100) spike per second
        :param refractory_mutation: a UniformMutation object, dtype should be 'float',
                                    reasonable value_range will be (1., 3.), if one time unit is equivalent to 1 ms,
                                    then this range represent (1., 3.) ms.
        """

        if baseline_mutation is None or baseline_mutation.get_dtype() == 'float':
            self.baseline_mutation = baseline_mutation
        else:
            raise ValueError('the baseline_mutation should be None or '
                             'the dtype of baseline_mutation should be "float".')

        if refractory_mutation is None or refractory_mutation.get_dtype() == 'float':
            self.refractory_mutation = refractory_mutation
        else:
            raise ValueError('the refractory_mutation should be None or '
                             'the dtype of refractory_mutation should be "float".')

    def get_mutated_baseline(self):
        """
        return a mutated baseline rate by self.baseline_mutation
        """
        if self.baseline_mutation is None:
            return None
        else:
            return self.baseline_mutation.get_value()

    def get_mutated_refractory(self):
        """
        return a mutated refractory period by self.refractory_mutation
        """
        if self.refractory_mutation is None:
            return None
        else:
            return self.refractory_mutation.get_value()


class ConnectionMutation(object):

    """
    definition of a connection mutation
    """

    def __init__(self, latency_mutation=None, amplitude_mutation=None, rise_time_mutation=None,
                 decay_time_mutation=None):
        """

        :param latency_mutation: a UniformMutation object, dtype should be 'int',
                                 reasonable value_range will be (3, 10)
        :param amplitude_mutation: a UniformMutation object, dtype should be 'float', this value can be wild!
                                   reasonable value_range will be (-1., 1.), from totally inhibit postsynaptic neuron
                                   to totally excite postsynaptic neuron
        :param rise_time_mutation: a UniformMutation object, dtype should be 'int',
                                   reasonable value_range will be (1, 5)
        :param decay_time_mutation: a UniformMutation object, dtype should be 'int',
                                    reasonable value_range will be (5, 20)
        """

        if latency_mutation is None or latency_mutation.get_dtype() == 'int':
            self.latency_mutation = latency_mutation
        else:
            raise ValueError('the latency_mutation should be None or '
                             'the dtype of latency_mutation should be "int".')

        if amplitude_mutation is None or amplitude_mutation.get_dtype() == 'float':
            self.amplitude_mutation = amplitude_mutation
        else:
            raise ValueError('the amplitude_mutation should be None or '
                             'the dtype of amplitude_mutation should be "float".')
        
        if rise_time_mutation is None or rise_time_mutation.get_dtype() == 'int':
            self.rise_time_mutation = rise_time_mutation
        else:
            raise ValueError('the rise_time_mutation should be None or '
                             'the dtype of rise_time_mutation should be "int".')
        
        if decay_time_mutation is None or decay_time_mutation.get_dtype() == 'int':
            self.decay_time_mutation = decay_time_mutation
        else:
            raise ValueError('the decay_time_mutation should be None or '
                             'the dtype of decay_time_mutation should be "int".')

    def get_mutated_latency(self):
        if self.latency_mutation is None:
            return None
        else:
            return self.latency_mutation.get_value()

    def get_mutated_amplitude(self):
        if self.amplitude_mutation is None:
            return None
        else:
            return self.amplitude_mutation.get_value()

    def get_mutated_rise_time(self):
        if self.rise_time_mutation is None:
            return None
        else:
            return self.rise_time_mutation.get_value()

    def get_mutated_decay_time(self):
        if self.decay_time_mutation is None:
            return None
        else:
            return self.decay_time_mutation.get_value()


