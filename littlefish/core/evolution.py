import random
import datetime
import numpy as np
import itertools
import littlefish.core.fish as fi



def choose_index_1d(indices, mutation_rate):
    """
    randomly chooses a subset of indices from a list of indices based on the mutation_rate
    :param indices: list of unsigned integers, all indices to choose from
    :param mutation_rate: float, [0., 1.]
    :return: list of unsigned integers, a list of subset of the indices
    """
    mutate_num = int(np.ceil(len(indices) * float(mutation_rate)))
    return random.sample(indices, mutate_num)


def choose_index_2d(indices0, indices1, mutation_rate):
    """
    randomly choose a subset of index pairs from a 2d grid based on the mutation rates
    :param indices0: 1d seq, list of indices along axis 0 (rows)
    :param indices1: 1d seq, list of indices along axis 1 (columns)
    :param mutation_rate: float, [0., 1.]
    :return: list of index pairs, each pair contains two elements [index0, index1] representing the coordinates of a
             randomly chosen location
    """
    all_coordinates = list(itertools.product(indices0, indices1))
    mutate_num = int(np.ceil(len(all_coordinates) * float(mutation_rate)))
    return random.sample(all_coordinates, mutate_num)


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
    

def mutate_connection(connection, connection_mutation):
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


def mutate_brain(brain, brain_mutation, verbose=False):

    if verbose:
        print('\nmutating input brain ...')

    mutated_neurons = brain.get_neurons().copy()
    mutated_connections = dict(brain.get_connections())

    mutate_neuron_ind = choose_index_1d(mutated_neurons.index.values, brain_mutation.get_neuron_mutation_rate())

    if verbose:
        print('\nmutating neurons:')
        print('total number of neurons: {}. neuron mutation rate: {}. number of neurons to be mutated: {}.'
              .format(len(mutated_neurons), brain_mutation.get_neuron_mutation_rate(), len(mutate_neuron_ind)))

    for mni in mutate_neuron_ind:
        curr_neuron = mutated_neurons.loc[mni, 'neuron']
        if curr_neuron.get_neuron_type() == 'eye':
            curr_mutated_neuron = mutate_neuron(neuron=curr_neuron, neuron_mutation=brain_mutation.get_eye_mutation())
            mutated_neurons.loc[mni, 'neuron'] = curr_mutated_neuron

        elif curr_neuron.get_neuron_type() == 'neuron':
            curr_mutated_neuron = mutate_neuron(neuron=curr_neuron,
                                                neuron_mutation=brain_mutation.get_neuron_mutation())
            mutated_neurons.loc[mni, 'neuron'] = curr_mutated_neuron

        elif curr_neuron.get_neuron_type() == 'muscle':
            curr_mutated_neuron = mutate_neuron(neuron=curr_neuron,
                                                neuron_mutation=brain_mutation.get_muscle_mutation())
            mutated_neurons.loc[mni, 'neuron'] = curr_mutated_neuron

    if verbose:
        print('\nmutating connections:')

    for con_name, con_df in mutated_connections.items():

        indices0 = con_df.index.values
        indices1 = con_df.columns.values

        mutate_conn_coors = choose_index_2d(indices0=indices0, indices1=indices1,
                                            mutation_rate=brain_mutation.get_connection_mutation_rate())

        if verbose:
            print('layer: {}'.format(con_name))
            print('total number of connections: {}. connection mutation rate: {}. '
                  'number of connections to be mutated: {}.'.format(len(indices0) * len(indices1),
                                                                    brain_mutation.get_connection_mutation_rate(),
                                                                    len(mutate_conn_coors)))

        for mutate_coor in mutate_conn_coors:
            curr_con = con_df.loc[mutate_coor[0], mutate_coor[1]]
            curr_mutated_con = mutate_connection(connection=curr_con,
                                                 connection_mutation=brain_mutation.get_connection_mutation())
            con_df.loc[mutate_coor[0], mutate_coor[1]] = curr_mutated_con

    mutated_brain = fi.Brain(neurons=mutated_neurons, connections=mutated_connections)
    return mutated_brain


def mutate_fish(fish, brain_mutation):

    mutated_brain = mutate_brain(fish.get_brain(), brain_mutation)
    mother_name = fish.get_name()
    name = 'fish_' + datetime.datetime.now().strftime('%y%m%d_%H_%M_%S')

    mutated_fish = fi.Fish(name=name, mother_name=mother_name, brain=mutated_brain, max_health=fish.get_max_health(),
                           health_decay_rate=fish.get_health_decay_rate(),
                           land_penalty_rate=fish.get_land_penalty_rate(), food_rate=fish.get_food_rate())
    return mutated_fish


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


class BrainMutation(object):
    """
    definition of a brain mutation
    """

    def __init__(self, neuron_mutation_rate=0., eye_mutation=NeuronMutation(), neuron_mutation=NeuronMutation(),
                 muscle_mutation=NeuronMutation(), connection_mutation_rate=0.,
                 connection_mutation=ConnectionMutation()):
        """
        :param neuron_mutation_rate: float, [0, 1.], fraction of neurons (eyes, hidden neurons and muscles) to be
                                     mutated
        :param eye_mutation: littlefish.core.evolution.NeuronMutation object
        :param neuron_mutation: littlefish.core.evolution.NeuronMutation object
        :param muscle_mutation: littlefish.core.evolution.NeuronMutation object
        :param connection_mutation_rate: float, [0, 1.], fraction of connections to be mutated for each layer
        :param connection_mutation: littlefish.core.evolution.ConnectionMutation object
        """

        self._neuron_mutation_rate = neuron_mutation_rate
        self._eye_mutation = eye_mutation
        self._neuron_mutation = neuron_mutation
        self._muscle_mutation = muscle_mutation
        self._connection_mutation_rate = connection_mutation_rate
        self._connection_mutation = connection_mutation

    def get_neuron_mutation_rate(self):
        return self._neuron_mutation_rate

    def get_eye_mutation(self):
        return self._eye_mutation

    def get_neuron_mutation(self):
        return self._neuron_mutation

    def get_muscle_mutation(self):
        return self._muscle_mutation

    def get_connection_mutation_rate(self):
        return self._connection_mutation_rate

    def get_connection_mutation(self):
        return self._connection_mutation


