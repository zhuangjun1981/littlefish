import os
import h5py
import random
import datetime
import numpy as np
import pandas as pd
import itertools
import littlefish.core.fish as fi
import littlefish.core.utilities as util


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

    return fi.Connection(
        latency=mutated_latency,
        amplitude=mutated_amplitude,
        rise_time=mutated_rise_time,
        decay_time=mutated_decay_time,
    )


def mutate_brain(
    brain,
    brain_mutation,
    neuron_mutation_rate=0.01,
    connection_mutation_rate=0.01,
    verbose=False,
):
    if verbose:
        print("\nmutating input brain ...")

    mutated_neurons = brain.get_neurons().copy()
    mutated_connections = dict(brain.get_connections())

    mutate_neuron_ind = choose_index_1d(
        list(mutated_neurons.index.values), neuron_mutation_rate
    )

    if verbose:
        print("\nmutating neurons:")
        print(
            "total number of neurons: {}. neuron mutation rate: {}. number of neurons to be mutated: {}.".format(
                len(mutated_neurons), neuron_mutation_rate, len(mutate_neuron_ind)
            )
        )

    for mni in mutate_neuron_ind:
        curr_neuron = mutated_neurons.loc[mni, "neuron"]
        if curr_neuron.get_neuron_type() == "eye":
            if verbose:
                print("Evolution: mutating eye. Index: {}.".format(mni))
            curr_mutated_neuron = mutate_neuron(
                neuron=curr_neuron, neuron_mutation=brain_mutation.get_eye_mutation()
            )
            mutated_neurons.loc[mni, "neuron"] = curr_mutated_neuron

        elif curr_neuron.get_neuron_type() == "neuron":
            if verbose:
                print("Evolution: mutating hidden neuron. Index: {}.".format(mni))
            curr_mutated_neuron = mutate_neuron(
                neuron=curr_neuron, neuron_mutation=brain_mutation.get_neuron_mutation()
            )
            mutated_neurons.loc[mni, "neuron"] = curr_mutated_neuron

        elif curr_neuron.get_neuron_type() == "muscle":
            if verbose:
                print("Evolution: mutating muscle. Index: {}.".format(mni))
            curr_mutated_neuron = mutate_neuron(
                neuron=curr_neuron, neuron_mutation=brain_mutation.get_muscle_mutation()
            )
            mutated_neurons.loc[mni, "neuron"] = curr_mutated_neuron

    if verbose:
        print("\nmutating connections:")

    for con_name, con_df in mutated_connections.items():
        indices0 = con_df.index.values
        indices1 = con_df.columns.values

        mutate_conn_coors = choose_index_2d(
            indices0=indices0, indices1=indices1, mutation_rate=connection_mutation_rate
        )

        if verbose:
            print("layer: {}".format(con_name))
            print(
                "total number of connections: {}. connection mutation rate: {}. "
                "number of connections to be mutated: {}.".format(
                    len(indices0) * len(indices1),
                    connection_mutation_rate,
                    len(mutate_conn_coors),
                )
            )

        for mutate_coor in mutate_conn_coors:
            curr_con = con_df.loc[mutate_coor[0], mutate_coor[1]]
            curr_mutated_con = mutate_connection(
                connection=curr_con,
                connection_mutation=brain_mutation.get_connection_mutation(),
            )
            con_df.loc[mutate_coor[0], mutate_coor[1]] = curr_mutated_con

    mutated_brain = fi.Brain(neurons=mutated_neurons, connections=mutated_connections)
    return mutated_brain


def mutate_fish(
    fish,
    brain_mutation,
    neuron_mutation_rate=0.01,
    connection_mutation_rate=0.01,
    verbose=False,
):
    mutated_brain = mutate_brain(
        brain=fish.get_brain(),
        brain_mutation=brain_mutation,
        neuron_mutation_rate=neuron_mutation_rate,
        connection_mutation_rate=connection_mutation_rate,
        verbose=verbose,
    )
    mother_name = fish.get_name()
    name = "fish_" + datetime.datetime.now().strftime("%y%m%d_%H_%M_%S.%f")

    mutated_fish = fi.Fish(
        name=name,
        mother_name=mother_name,
        brain=mutated_brain,
        max_health=fish.get_max_health(),
        health_decay_rate=fish.get_health_decay_rate(),
        land_penalty_rate=fish.get_land_penalty_rate(),
        food_rate=fish.get_food_rate(),
        move_penalty_rate=fish.get_move_penalty_rate(),
    )

    if verbose:
        print("fish: {} generated.".format(name))
    return mutated_fish


def get_offspring_num(mother_life_spans, hard_thr, soft_thr, reproducing_rate=0.0002):
    """
    given a list of mother life spans in multiple simulations, return a number representing how many offsprings it
    will produce.

    :param mother_life_spans: list of non-negative int, mother's life spans in multiple simulation
    :param hard_thr: non-negative int, mother will have chance to reproduce only if all life spans in
                     mother_life_spans are no less than this threshold
    :param soft_thr: non-negative int, each life span in mother_life_spans longer than this threshold will be used to
                     calculated offspring number
    :param reproducing_rate: positive float, this rate time the life spans that exceed soft_thr x soft_thr_ratio
                             will be returned
    :return: non-negative int, number of offsprings the mother fish wil produce
    """

    offspring_num = 0
    if min(mother_life_spans) >= hard_thr:
        for mother_life_span in mother_life_spans:
            reproducing_life = mother_life_span - soft_thr
            if reproducing_life > 0:
                offspring_num += int(np.ceil((reproducing_life * reproducing_rate)))
    return offspring_num


def get_single_param_mutation(value_range, dtype):
    if value_range is None:
        mutation = None
    else:
        mutation = UniformMutation(value_range=value_range, dtype=dtype)
    return mutation


def get_brain_mutation_from_brain_mutation_config(brain_mutation_config):
    eye_bl_mutation = get_single_param_mutation(
        brain_mutation_config["eye_bl_r"], "float"
    )
    eye_rp_mutation = get_single_param_mutation(
        brain_mutation_config["eye_rp_r"], "float"
    )
    neuron_bl_mutation = get_single_param_mutation(
        brain_mutation_config["neuron_bl_r"], "float"
    )
    neuron_rp_mutation = get_single_param_mutation(
        brain_mutation_config["neuron_rp_r"], "float"
    )
    muscle_bl_mutation = get_single_param_mutation(
        brain_mutation_config["muscle_bl_r"], "float"
    )
    muscle_rp_mutation = get_single_param_mutation(
        brain_mutation_config["muscle_rp_r"], "float"
    )
    connection_l_mutation = get_single_param_mutation(
        brain_mutation_config["connection_l_r"], "int"
    )
    connection_a_mutation = get_single_param_mutation(
        brain_mutation_config["connection_a_r"], "float"
    )
    connection_rt_mutation = get_single_param_mutation(
        brain_mutation_config["connection_rt_r"], "int"
    )
    connection_dt_mutation = get_single_param_mutation(
        brain_mutation_config["connection_dt_r"], "int"
    )

    eye_mutation = NeuronMutation(
        baseline_mutation=eye_bl_mutation, refractory_mutation=eye_rp_mutation
    )
    neuron_mutation = NeuronMutation(
        baseline_mutation=neuron_bl_mutation, refractory_mutation=neuron_rp_mutation
    )
    muscle_mutation = NeuronMutation(
        baseline_mutation=muscle_bl_mutation, refractory_mutation=muscle_rp_mutation
    )
    connection_mutation = ConnectionMutation(
        latency_mutation=connection_l_mutation,
        amplitude_mutation=connection_a_mutation,
        rise_time_mutation=connection_rt_mutation,
        decay_time_mutation=connection_dt_mutation,
    )

    brain_mutation = BrainMutation(
        eye_mutation=eye_mutation,
        neuron_mutation=neuron_mutation,
        muscle_mutation=muscle_mutation,
        connection_mutation=connection_mutation,
    )

    return brain_mutation


def get_default_brain_mutation():
    defalut_config = util.get_default_config()
    return get_brain_mutation_from_brain_mutation_config(
        defalut_config["brain_mutation_config"]
    )


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
            raise ValueError(
                "the input _value_range should be sequence with length of 2."
            )

        if dtype == "int":
            v0 = int(value_range[0])
            v1 = int(value_range[1])
            self._dtype = "int"
        elif dtype == "float":
            v0 = float(value_range[0])
            v1 = float(value_range[1])
            self._dtype = "float"
        else:
            raise ValueError('the _dtype shoule be either "int" or "float".')

        if v0 == v1:
            raise ValueError(
                "the two values in the input _value_range should be different."
            )
        elif v0 < v1:
            self._value_range = (v0, v1)
        else:
            self._value_range = (v0, v1)

    def get_value(self):
        """
        if self._dtype is 'int': uses random.randint() function
        if self._dtype is 'float': uses random.uniform() function

        :return: a random value follow a uniform distribution with a range defined by self._value_range, including the
                 start but excluding the end
        """

        if self._dtype == "int":
            return random.randint(self._value_range[0], self._value_range[1] - 1)
        elif self._dtype == "float":
            return random.uniform(self._value_range[0], self._value_range[1])

    def get_dtype(self):
        return self._dtype

    def get_value_range(self):
        return self._value_range

    def __str__(self):
        return "littlefish.core.evolution.UniformMutation object. dtype:{}; value_range:{}".format(
            self._dtype, self._value_range
        )


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

        if baseline_mutation is None or baseline_mutation.get_dtype() == "float":
            self.baseline_mutation = baseline_mutation
        else:
            raise ValueError(
                "the baseline_mutation should be None or "
                'the dtype of baseline_mutation should be "float".'
            )

        if refractory_mutation is None or refractory_mutation.get_dtype() == "float":
            self.refractory_mutation = refractory_mutation
        else:
            raise ValueError(
                "the refractory_mutation should be None or "
                'the dtype of refractory_mutation should be "float".'
            )

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

    def __init__(
        self,
        latency_mutation=None,
        amplitude_mutation=None,
        rise_time_mutation=None,
        decay_time_mutation=None,
    ):
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

        if latency_mutation is None or latency_mutation.get_dtype() == "int":
            self.latency_mutation = latency_mutation
        else:
            raise ValueError(
                "the latency_mutation should be None or "
                'the dtype of latency_mutation should be "int".'
            )

        if amplitude_mutation is None or amplitude_mutation.get_dtype() == "float":
            self.amplitude_mutation = amplitude_mutation
        else:
            raise ValueError(
                "the amplitude_mutation should be None or "
                'the dtype of amplitude_mutation should be "float".'
            )

        if rise_time_mutation is None or rise_time_mutation.get_dtype() == "int":
            self.rise_time_mutation = rise_time_mutation
        else:
            raise ValueError(
                "the rise_time_mutation should be None or "
                'the dtype of rise_time_mutation should be "int".'
            )

        if decay_time_mutation is None or decay_time_mutation.get_dtype() == "int":
            self.decay_time_mutation = decay_time_mutation
        else:
            raise ValueError(
                "the decay_time_mutation should be None or "
                'the dtype of decay_time_mutation should be "int".'
            )

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

    def __init__(
        self,
        eye_mutation=NeuronMutation(),
        neuron_mutation=NeuronMutation(),
        muscle_mutation=NeuronMutation(),
        connection_mutation=ConnectionMutation(),
    ):
        """

        :param neuron_mutation_rate: float, [0, 1.], fraction of neurons (eyes, hidden neurons and muscles) to be
                                     mutated
        :param eye_mutation: littlefish.core.evolution.NeuronMutation object
        :param neuron_mutation: littlefish.core.evolution.NeuronMutation object
        :param muscle_mutation: littlefish.core.evolution.NeuronMutation object
        :param connection_mutation_rate: float, [0, 1.], fraction of connections to be mutated for each layer
        :param connection_mutation: littlefish.core.evolution.ConnectionMutation object
        """

        self._eye_mutation = eye_mutation
        self._neuron_mutation = neuron_mutation
        self._muscle_mutation = muscle_mutation
        self._connection_mutation = connection_mutation

    def get_eye_mutation(self):
        return self._eye_mutation

    def get_neuron_mutation(self):
        return self._neuron_mutation

    def get_muscle_mutation(self):
        return self._muscle_mutation

    def get_connection_mutation(self):
        return self._connection_mutation


class PopulationEvolution(object):
    def __init__(
        self,
        population_size: int,
        turnover_rate: float,
        neuron_mutation_rate: float,
        connection_mutation_rate: float,
        brain_mutation: BrainMutation,
        life_span_hard_threshold: int = None,
        generation_digits_num=7,
    ):
        """
        :param population_size: int, how many fish will be in the next generation, if None, it will be the same as
            last generation
        :param turnover_rate: float, (0., 1.), proportion of fish in current generation that will die out
        :param neuron_mutation_rate: float, (0., 1.), probability of neuron mutation (of each mutable component)
        :param connection_mutation_rate: float, (0., 1.), probability of connection mutation (of each mutable component)
        :param brain_mutation: BrainMutation object, defines the value range of each component of the brain
        :param life_span_hard_threshold: int, only the fish with life span larger than this number will have chances to
            generate offspring and pass to next generation
        :param generation_digit_num: positive int, number of digits to represent generation number,
                                     default: 7 (max generation num 10 million)
        """

        self.population_size = population_size
        self.turnover_rate = turnover_rate
        self.neuron_mutation_rate = neuron_mutation_rate
        self.connection_mutation_rate = connection_mutation_rate
        self.brain_mutation = brain_mutation
        self.life_span_hard_threshold = life_span_hard_threshold
        self.generation_digits_num = generation_digits_num

    def _calculate_offspring_num(
        self,
        base_folder,
        curr_generation_ind,
        simulation_ind=0,
    ):
        """
        calculate number of offsprings for each fish in the current generation, the mother fish will go to next
        generation as well. The number of offsprings a mother can produce is proportional to its extra life span
        exceeding the life threshold (calculated by the 'get_offspring_num()' function). If several mother fish
        has same life span, the ones with bigger generation numbers (which survived more generations) will have
        higher priority to pass to next generation. The number of offsprings plus the number of mother fish
        precisely equal to the population_size

        :param base_folder: str, path to the folder that saves the simulation results, each generation should be
            save as subfolders with names like "generation_0000000", "generation_0000001", ...
        :param generation_ind: non-negative integer, current generation number
        :param turnover_rate: float, (0., 1.), proportion of fish in current generation that will die out
        :param simulation_num: non-negative integer, the simulation index to extract life span
        :param population_size: positive integer, number of individuals of next generation, if None, it will be the
                                same as current generation.
        :return: life_thr, positive integer, only fish with life span longer than this number will have chance to
                           spawn offspring, the extra life (fish's life span - life_thr) determines the possibility
                           of its offspring among other fish in the current generation
                 fishes, pandas dataframe, rows: fish those will produce offspring, columns: ['fish_name', 'life_span',
                         'generation_num', 'extra_life', 'offspring_num']

        """

        curr_gen_dir = os.path.join(
            base_folder,
            util.get_generation_name(curr_generation_ind, self.generation_digits_num),
        )

        if not os.path.isdir(curr_gen_dir):
            raise LookupError(
                "PopulationEvolution: The path to current generation population does not exist!"
            )

        fish_ns = [
            f
            for f in os.listdir(curr_gen_dir)
            if f[-5:] == ".hdf5" and f[0:5] == "fish_"
        ]

        if len(fish_ns) == 0:
            raise LookupError(
                "PopulationEvolution: There is no fish in the current generation folder."
            )

        fish_ns.sort()

        if self.population_size is None:
            self.population_size = len(fish_ns)

        fish_ns = [os.path.splitext(f)[0] for f in fish_ns]
        life_spans = []
        generation_nums = []
        for fish_n in fish_ns:
            fish_f = h5py.File(os.path.join(curr_gen_dir, fish_n + ".hdf5"), "r")

            generation_nums.append(fish_f["generations"].shape[0])

            curr_sim_ns = [s for s in fish_f.keys() if s[:11] == "simulation_"]
            if len(curr_sim_ns) == 0:
                raise LookupError(
                    "PopulationEvolution: cannot find simulation results for fish: {}".format(
                        fish_n
                    )
                )

            curr_sim_n = [
                s for s in curr_sim_ns if int(s.split("_")[1]) == simulation_ind
            ]
            if len(curr_sim_n) != 1:
                raise LookupError(
                    "PopulationEvolution: there should one and only one simulation log matches the "
                    "specified simulation index: {} for fish: {}.".format(
                        simulation_ind, fish_n
                    )
                )
            curr_sim_n = curr_sim_n[0]

            life_spans.append(fish_f[curr_sim_n]["simulation_log/last_time_point"][()])
            fish_f.close()

        fishes = pd.DataFrame(
            list(zip(fish_ns, life_spans, generation_nums)),
            columns=["fish_name", "life_span", "generation_num"],
        )
        fishes.sort_values(
            by=["life_span", "generation_num"], ascending=False, inplace=True
        )

        retain_number = int(
            np.ceil(len(fish_ns) * (1.0 - self.turnover_rate))
        )  # number of fish to retain

        fishes = fishes[0:retain_number]

        if self.life_span_hard_threshold is not None:
            fishes = fishes.query("life_span > @self.life_span_hard_threshold").copy()

        if len(fishes) == 0:
            raise ValueError(
                "No fish qualifies as mother fish. Try reducing the 'life_span_hard_threshold'."
            )

        life_thr = fishes.iloc[-1, 1]

        fishes["extra_life"] = fishes["life_span"] - life_thr

        new_fish_number = self.population_size - fishes.shape[0]
        fishes["offspring_num"] = util.distrube_number(
            fishes["extra_life"], new_fish_number
        )

        print("================== mother fish ==================")
        print(fishes)
        print("=================================================")

        return life_thr, fishes

    def generate_next_generation(
        self,
        base_folder,
        curr_generation_ind,
        simulation_ind,
    ):
        print(
            "\n======================================================================"
        )
        print(
            "PopulationEvolution: generating fish for generation: {} ...".format(
                curr_generation_ind + 1
            )
        )

        curr_gen_folder = os.path.join(
            base_folder,
            util.get_generation_name(curr_generation_ind, self.generation_digits_num),
        )
        next_gen_folder = os.path.join(
            base_folder,
            util.get_generation_name(
                curr_generation_ind + 1, self.generation_digits_num
            ),
        )
        os.mkdir(next_gen_folder)

        life_thr, fishes = self._calculate_offspring_num(
            base_folder=base_folder,
            curr_generation_ind=curr_generation_ind,
            simulation_ind=simulation_ind,
        )

        for fish_ind, fish_row in fishes.iterrows():
            mother_fish_f = h5py.File(
                os.path.join(curr_gen_folder, fish_row["fish_name"] + ".hdf5"), "a"
            )
            mother_fish = fi.Fish.from_h5_group(mother_fish_f["fish"])
            mother_fish_gens = list(mother_fish_f["generations"][()])

            children_lst = []

            child_fish_f = h5py.File(
                os.path.join(next_gen_folder, mother_fish.name + ".hdf5"), "a"
            )
            child_fish_grp = child_fish_f.create_group("fish")
            mother_fish.to_h5_group(child_fish_grp)
            mother_fish_gens.append(curr_generation_ind + 1)
            child_fish_f["generations"] = mother_fish_gens
            child_fish_f.close()
            children_lst.append(mother_fish.name)

            for offspring_ind in range(fish_row["offspring_num"]):
                child_fish = mutate_fish(
                    fish=mother_fish,
                    brain_mutation=self.brain_mutation,
                    neuron_mutation_rate=self.neuron_mutation_rate,
                    connection_mutation_rate=self.connection_mutation_rate,
                )
                child_fish_f = h5py.File(
                    os.path.join(next_gen_folder, child_fish.name + ".hdf5"), "a"
                )
                child_fish_grp = child_fish_f.create_group("fish")
                child_fish.to_h5_group(child_fish_grp)
                child_fish_f["generations"] = [curr_generation_ind + 1]
                child_fish_f.close()
                children_lst.append(child_fish.name)

            ng_grp = mother_fish_f.create_group(
                "next_generation_" + datetime.datetime.now().strftime("%y%m%d_%H_%M_%S")
            )
            ng_grp["children_list"] = [c.encode("UTF-8") for c in children_lst]
            ng_grp["life_threshold"] = life_thr
            ng_grp["simulation_ind"] = simulation_ind
            ng_grp["neuron_mutation_rate"] = self.neuron_mutation_rate
            ng_grp["connection_mutation_rate"] = self.connection_mutation_rate

            mother_fish_f.close()

        print(
            "PopulationEvolution: fish generation for generation: {} finished.".format(
                curr_generation_ind + 1
            )
        )
        print("======================================================================")

        return next_gen_folder
