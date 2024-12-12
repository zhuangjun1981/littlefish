import os
import h5py
import random
import datetime
import copy
import numpy as np
import pandas as pd
import itertools
from typing import Union
import littlefish.core.fish as fi
import littlefish.brain.connection as conn
import littlefish.brain.neuron as neu
import littlefish.brain.brain as brain
import littlefish.core.utilities as util


def choose_index(
    indices: list[int],
    mutation_rate: float,
) -> list[int]:
    """
    randomly chooses a subset of indices from a list of indices based on the mutation_rate

    :param indices: list of unsigned integers, all indices to choose from
    :param mutation_rate: float, [0., 1.]
    :return: list of unsigned integers, a list of subset of the indices
    """

    mutate_num = int(np.ceil(len(indices) * float(mutation_rate)))
    return random.sample(indices, mutate_num)


# def choose_index_2d(indices0, indices1, mutation_rate):
#     """
#     randomly choose a subset of index pairs from a 2d grid based on the mutation rates

#     :param indices0: 1d seq, list of indices along axis 0 (rows)
#     :param indices1: 1d seq, list of indices along axis 1 (columns)
#     :param mutation_rate: float, [0., 1.]
#     :return: list of index pairs, each pair contains two elements [index0, index1] representing the coordinates of a
#              randomly chosen location
#     """

#     all_coordinates = list(itertools.product(indices0, indices1))
#     mutate_num = int(np.ceil(len(all_coordinates) * float(mutation_rate)))
#     return random.sample(all_coordinates, mutate_num)


def get_offspring_num(
    mother_life_spans: list[int],
    hard_thr: int,
    soft_thr: int,
    reproducing_rate: float = 0.0002,
) -> int:
    """
    given a list of mother life spans in multiple simulations, return a number representing how many offsprings it
    will produce.

    :param mother_life_spans: list of non-negative int, mother's life spans in multiple simulation
    :param hard_thr: non-negative int, mother will have chance to reproduce only if all life spans in
        mother_life_spans are no less than this threshold
    :param soft_thr: non-negative int, each life span in mother_life_spans longer than this threshold will be used to
        calculated offspring number
    :param reproducing_rate: positive float, this rate times the life spans that exceed soft_thr x soft_thr_ratio
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
    eye_gain_mutation = get_single_param_mutation(
        brain_mutation_config["eye_gain_r"], "float"
    )
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

    eye_mutation = EyeMutation(
        baseline_mutation=eye_bl_mutation,
        refractory_mutation=eye_rp_mutation,
        gain_mutation=eye_gain_mutation,
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

    def __init__(self, value_range: list[float], dtype: str):
        """

        :param value_range: tuple of two numbers, the two value should be different.
        :param dtype: str, 'int' or 'float'.
            if 'int' random value will be drawn by random.randint()
            if 'float' random value will be drawn by random.uniform()
        """

        if len(value_range) != 2:
            raise ValueError(
                "the input _value_range should be sequence with length of 2."
            )

        if dtype == "int":
            v0 = int(value_range[0])
            v1 = int(value_range[1])
            self.dtype = "int"
        elif dtype == "float":
            v0 = float(value_range[0])
            v1 = float(value_range[1])
            self.dtype = "float"
        else:
            raise ValueError('the _dtype shoule be either "int" or "float".')

        if v0 == v1:
            raise ValueError(
                "the two values in the input _value_range should be different."
            )
        elif v0 < v1:
            self.value_range = (v0, v1)
        else:
            self.value_range = (v0, v1)

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

    def __str__(self):
        return "littlefish.core.evolution.UniformMutation object. dtype:{}; value_range:{}".format(
            self._dtype, self._value_range
        )


class NeuronMutation(object):

    """
    definition of a neuron mutation
    """

    def __init__(
        self,
        baseline_mutation: UniformMutation = None,
        refractory_mutation: UniformMutation = None,
    ):
        """
        :param baseline_mutation: a UniformMutation object, dtype should be 'float',
            reasonable value_range will be (0., 0.1), if one time unit is equivalent to 1 ms,
            then this range represents (0, 100) spike per second
        :param refractory_mutation: a UniformMutation object, dtype should be 'float',
            reasonable value_range will be (1., 3.), if one time unit is equivalent to 1 ms,
            then this range represent (1., 3.) ms.
        """

        if baseline_mutation is None or baseline_mutation.dtype == "float":
            self.baseline_mutation = baseline_mutation
        else:
            raise ValueError(
                "the baseline_mutation should be None or "
                'the dtype of baseline_mutation should be "float".'
            )

        if refractory_mutation is None or refractory_mutation.dtype == "float":
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
        return (
            self.baseline_mutation.get_value()
            if self.baseline_mutation is not None
            else None
        )

    def get_mutated_refractory(self):
        """
        return a mutated refractory period by self.refractory_mutation
        """
        return (
            self.refractory_mutation.get_value()
            if self.refractory_mutation is not None
            else None
        )


class EyeMutation(NeuronMutation):
    """
    Define a mutation class for eyes.
    Similar to NeuronMutation, but with one more attribute "gain_mutation".
    Potentially, "rf_positions" and "rf_weights" can be mutated too, but not implemented now.
    """

    def __init__(
        self,
        baseline_mutation: UniformMutation = None,
        refractory_mutation: UniformMutation = None,
        gain_mutation: UniformMutation = None,
        # rf_positions_mutation: None,  # not implemented
        # rf_weights_mutation: None,  # not implemented
    ):
        super().__init__(
            baseline_mutation=baseline_mutation, refractory_mutation=refractory_mutation
        )
        if gain_mutation is None or gain_mutation.dtype == "float":
            self.gain_mutation = gain_mutation
        else:
            raise ValueError(
                "the gain_mutation should be None or "
                'the dtype of gain_mutation should be "float".'
            )

    def get_mutated_gain(self):
        return (
            self.gain_mutation.get_value() if self.gain_mutation is not None else None
        )


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

        if latency_mutation is None or latency_mutation.dtype == "int":
            self.latency_mutation = latency_mutation
        else:
            raise ValueError(
                "the latency_mutation should be None or "
                'the dtype of latency_mutation should be "int".'
            )

        if amplitude_mutation is None or amplitude_mutation.dtype == "float":
            self.amplitude_mutation = amplitude_mutation
        else:
            raise ValueError(
                "the amplitude_mutation should be None or "
                'the dtype of amplitude_mutation should be "float".'
            )

        if rise_time_mutation is None or rise_time_mutation.dtype == "int":
            self.rise_time_mutation = rise_time_mutation
        else:
            raise ValueError(
                "the rise_time_mutation should be None or "
                'the dtype of rise_time_mutation should be "int".'
            )

        if decay_time_mutation is None or decay_time_mutation.dtype == "int":
            self.decay_time_mutation = decay_time_mutation
        else:
            raise ValueError(
                "the decay_time_mutation should be None or "
                'the dtype of decay_time_mutation should be "int".'
            )

    def get_mutated_latency(self):
        return (
            self.latency_mutation.get_value()
            if self.latency_mutation is not None
            else None
        )

    def get_mutated_amplitude(self):
        return (
            self.amplitude_mutation.get_value()
            if self.amplitude_mutation is not None
            else None
        )

    def get_mutated_rise_time(self):
        return (
            self.rise_time_mutation.get_value()
            if self.rise_time_mutation is not None
            else None
        )

    def get_mutated_decay_time(self):
        return (
            self.decay_time_mutation.get_value()
            if self.decay_time_mutation is not None
            else None
        )


class BrainMutation(object):
    """
    definition of a brain mutation
    """

    def __init__(
        self,
        eye_mutation: EyeMutation = EyeMutation(),
        neuron_mutation: NeuronMutation = NeuronMutation(),
        muscle_mutation: NeuronMutation = NeuronMutation(),
        connection_mutation: ConnectionMutation = ConnectionMutation(),
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

        self.eye_mutation = eye_mutation
        self.neuron_mutation = neuron_mutation
        self.muscle_mutation = muscle_mutation
        self.connection_mutation = connection_mutation


class PopulationEvolution(object):
    def __init__(
        self,
        population_size: int,
        turnover_rate: float,
        neuron_mutation_rate: float,
        connection_mutation_rate: float,
        brain_mutation: BrainMutation,
        life_span_hard_threshold: int = 0,
        movement_hard_threshold: int = 0,
        generation_digits_num: int = 7,
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
        :param movement_hard_threshold: int, only the fish with movement more than this number will have chances to generate
            offspring and pass to next generation
        :param generation_digit_num: positive int, number of digits to represent generation number,
                                     default: 7 (max generation num 10 million)
        """

        self.population_size = population_size
        self.turnover_rate = turnover_rate
        self.neuron_mutation_rate = neuron_mutation_rate
        self.connection_mutation_rate = connection_mutation_rate
        self.brain_mutation = brain_mutation
        self.life_span_hard_threshold = life_span_hard_threshold
        self.movement_hard_threshold = movement_hard_threshold
        self.generation_digits_num = generation_digits_num

    @staticmethod
    def _find_single_simulation_log_name(grp_root: h5py.Group):
        """
        check if there is one and only one simulation log group in the grp_root
        and return the simulation log group name
        """
        sim_log_ns = [s for s in grp_root.keys() if s[:11] == "simulation_"]

        if len(sim_log_ns) == 0:
            raise LookupError("Cannot find simulation log")
        elif len(sim_log_ns) > 1:
            raise LookupError("More than one simulation logs found.")

        return sim_log_ns[0]

    def _calculate_offspring_num(
        self,
        base_folder: str,
        curr_generation_ind: int,
    ):
        """
        calculate number of offsprings for each fish in the current generation, the mother fish will go to next
        generation as well. The number of offsprings a mother can produce is proportional to its extra life span
        exceeding the life threshold (calculated by the 'get_offspring_num()' function). If several mother fish
        has the same life span, the ones with larger generation numbers (who survived more generations) will have
        higher priority to pass to next generation. The number of offsprings plus the number of mother fish
        precisely equal to the population_size

        :param base_folder: str, path to the folder that saves the simulation results, each generation should be
            save as subfolders with names like "generation_0000000", "generation_0000001", ...
        :param generation_ind: non-negative integer, current generation number
        :param turnover_rate: float, (0., 1.), proportion of fish in current generation that will die out
        :param simulation_num: non-negative integer, the simulation index to extract life span
        :param population_size: positive integer, number of individuals of next generation, if None, it will be the
                                same as current generation.
        :return:
            life_thr, positive integer, only fish with life span longer than this number will have chance to
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
        total_movements = []
        generation_nums = []
        for fish_n in fish_ns:
            fish_f = h5py.File(os.path.join(curr_gen_dir, fish_n + ".hdf5"), "r")

            generation_nums.append(fish_f["generations"].shape[0])

            curr_sim_n = self._find_single_simulation_log_name(fish_f)
            life_spans.append(
                fish_f[curr_sim_n]["simulation_cache/last_time_point"][()]
            )
            total_movements.append(fish_f[curr_sim_n][f"fish_{fish_n}/total_moves"][()])
            fish_f.close()

        fishes = pd.DataFrame(
            list(zip(fish_ns, life_spans, total_movements, generation_nums)),
            columns=["fish_name", "life_span", "total_movements", "generation_num"],
        )
        fishes.sort_values(
            by=["life_span", "generation_num"], ascending=False, inplace=True
        )

        retain_number = int(
            np.ceil(len(fish_ns) * (1.0 - self.turnover_rate))
        )  # number of fish to retain

        fishes = fishes[0:retain_number]

        fishes = fishes.query(
            "life_span >= @self.life_span_hard_threshold and total_movements >= @self.movement_hard_threshold"
        ).copy()

        if len(fishes) == 0:
            raise ValueError(
                "No fish qualifies as mother fish. Try reducing the 'life_span_hard_threshold' or the 'movement_hard_threshold'."
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
        base_folder: int,
        curr_generation_ind: int,
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


def mutate_neuron(
    neuron: Union[neu.Neuron, neu.Muscle],
    neuron_mutation: NeuronMutation,
) -> neu.Neuron:
    """
    mutate a neuron, can be Eye, Neuron or Muscle

    :param neuron: the initial little_fish.brain.neuron.Neuron object
    :param neuron_mutation: little_fish.core.evolution.NeuronMutation object
    :return: a mutated little_fish.brain.neuron.Neuron object
    """

    mutated_neuron = copy.deepcopy(neuron)

    mutated_baseline = neuron_mutation.get_mutated_baseline()
    if mutated_baseline is not None:
        mutated_neuron.baseline_rate = mutated_baseline

    mutated_refractory = neuron_mutation.get_mutated_refractory()
    if mutated_refractory is not None:
        mutated_neuron.refractory_period = mutated_refractory

    return mutated_neuron


def mutate_eye(eye: neu.Eye, eye_mutation: EyeMutation) -> neu.Eye:
    """
    mutate a neuron, can be Eye, Neuron or Muscle

    :param neuron: the initial little_fish.brain.neuron.Eye object
    :param neuron_mutation: little_fish.core.evolution.NeuronMutation object
    :return: a mutated little_fish.brain.neuron.Eye object
    """

    mutated_eye = copy.deepcopy(eye)

    mutated_gain = eye_mutation.get_mutated_gain()
    if mutated_gain is not None:
        mutate_eye.gain = mutated_gain

    mutated_baseline = eye_mutation.get_mutated_baseline()
    if mutated_baseline is not None:
        mutated_eye.baseline_rate = mutated_baseline

    mutated_refractory = eye_mutation.get_mutated_refractory()
    if mutated_refractory is not None:
        mutated_eye.refractory_period = mutated_refractory

    return mutated_eye


def mutate_connection(
    connection: conn.Connection, connection_mutation: ConnectionMutation
) -> conn.Connection:
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
    curr_brain: brain.Brain,
    brain_mutation: BrainMutation,
    neuron_mutation_rate: float = 0.01,
    connection_mutation_rate: float = 0.01,
    verbose=False,
) -> brain.Brain:
    if verbose:
        print("\nmutating input brain ...")

    mutated_neurons = curr_brain.neurons.copy()
    mutated_connections = curr_brain.connections.copy()

    mutate_neuron_idxs = choose_index(mutated_neurons.index, neuron_mutation_rate)

    if verbose:
        print("\nmutating neurons:")
        print(
            f"total number of neurons: {len(mutated_neurons)}. "
            f"neuron mutation rate: {neuron_mutation_rate}. "
            f"number of neurons to be mutated: {len(mutate_neuron_idxs)}."
        )

    for mni in mutate_neuron_idxs:
        curr_neuron = mutated_neurons.loc[mni, "neuron"]
        if curr_neuron.neuron_type == "eye":
            if verbose:
                print(f"Evolution: mutating eye. Index: {mni}.")
            mutated_neurons.loc[mni, "neuron"] = mutate_eye(
                curr_neuron, brain_mutation.eye_mutation
            )

        elif curr_neuron.neuron_type in ["neuron", "muscle"]:
            if verbose:
                print(f"Evolution: mutating hidden neuron. Index: {mni}.")
            mutated_neurons.loc[mni, "neuron"] = mutate_neuron(
                curr_neuron, brain_mutation.neuron_mutation
            )

    if verbose:
        print("\nmutating connections:")

    mutate_connection_idxs = choose_index(
        mutated_connections.index, connection_mutation_rate
    )

    for mci in mutate_connection_idxs:
        curr_connection = mutated_connections.loc[mci, "connection"]
        if verbose:
            print(f"Evolution: mutating connection. Index: {mci}.")
        mutated_connections.loc[mci, "connection"] = mutate_connection(
            curr_connection, brain_mutation.connection_mutation
        )

    mutated_brain = brain.Brain(
        neurons=mutated_neurons, connections=mutated_connections
    )
    return mutated_brain


def mutate_fish(
    fish: fi.Fish,
    brain_mutation: BrainMutation,
    neuron_mutation_rate: float = 0.01,
    connection_mutation_rate: float = 0.01,
    verbose=False,
) -> fi.Fish:
    mutated_brain = mutate_brain(
        brain=fish.brain,
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
        max_health=fish.max_health,
        health_decay_rate=fish.health_decay_rate,
        land_penalty_rate=fish.land_penalty_rate,
        food_rate=fish.food_rate,
        move_penalty_rate=fish.move_penalty_rate,
    )

    if verbose:
        print("fish: {} generated.".format(name))
    return mutated_fish
