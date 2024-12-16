import os
import h5py
import random
import datetime
import copy
import yaml
import numpy as np
import pandas as pd
from typing import Union
import littlefish.core.fish as fi
import littlefish.core.simulation as sim
import littlefish.brain.connection as conn
import littlefish.brain.neuron as neu
import littlefish.brain.brain as brain
import littlefish.brain.functional as fn
import littlefish.core.utilities as utils
from littlefish.core.utilities import is_integer
from littlefish.log_analysis.simulation_log import get_simulation_logs


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
    return random.sample(list(indices), mutate_num)


def distrube_number(
    values: np.ndarray,
    population_size: int,
    temperature: float = 1.0,
) -> list[int]:
    """
    distribute a number of individuals into each bucket according to a given list of numbers
    the probability of each value index is calculated as softmax with temperature

    :param values: 1d array like, softmax(values) specifies the probability distribution.
    :param population_size: positive integer, total number of individuals to be distributed
    :param temperature: float, temperature in softmax calculation, higher temperature will
        encourage more randomness in the output. should be no less than one.
    :return: list of non-negative integers, sum of which should precisely equal to population_size
    """

    assert temperature >= 1, "temperature should be no less than 1."

    if not is_integer(population_size) or population_size < 1.0:
        raise ValueError("Utility: population_size sould be positive integer.")

    if len(values) == 1:
        return [population_size]

    # # use rank instead of values for softmax
    # values = np.array(list(range(len(values)))[::-1])

    # softmax to get probabilities
    e_x = np.exp((values - np.max(values)) / temperature)
    probs = e_x / e_x.sum()

    offspring_number = [0] * len(probs)

    idxs = np.random.choice(len(probs), size=population_size, p=probs)

    for idx in idxs:
        offspring_number[idx] += 1

    return offspring_number


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


def get_single_param_mutation(
    value_range: list[Union[int, float]],
    dtype: str,
    perturb_rate: float,
):
    if value_range is None:
        mutation = None
    else:
        mutation = UniformMutation(
            value_range=value_range,
            dtype=dtype,
            perturb_rate=perturb_rate,
        )
    return mutation


def get_brain_mutation_from_brain_mutation_config(brain_mutation_config):
    eye_gain_mutation = get_single_param_mutation(
        value_range=brain_mutation_config["eye_gain_range"],
        dtype="float",
        perturb_rate=brain_mutation_config["perturb_rate"],
    )
    eye_bl_mutation = get_single_param_mutation(
        value_range=brain_mutation_config["eye_bl_range"],
        dtype="float",
        perturb_rate=brain_mutation_config["perturb_rate"],
    )
    eye_rp_mutation = get_single_param_mutation(
        value_range=brain_mutation_config["eye_rp_range"],
        dtype="float",
        perturb_rate=brain_mutation_config["perturb_rate"],
    )
    neuron_bl_mutation = get_single_param_mutation(
        value_range=brain_mutation_config["neuron_bl_range"],
        dtype="float",
        perturb_rate=brain_mutation_config["perturb_rate"],
    )
    neuron_rp_mutation = get_single_param_mutation(
        value_range=brain_mutation_config["neuron_rp_range"],
        dtype="float",
        perturb_rate=brain_mutation_config["perturb_rate"],
    )
    muscle_bl_mutation = get_single_param_mutation(
        value_range=brain_mutation_config["muscle_bl_range"],
        dtype="float",
        perturb_rate=brain_mutation_config["perturb_rate"],
    )
    muscle_rp_mutation = get_single_param_mutation(
        value_range=brain_mutation_config["muscle_rp_range"],
        dtype="float",
        perturb_rate=brain_mutation_config["perturb_rate"],
    )
    connection_l_mutation = get_single_param_mutation(
        value_range=brain_mutation_config["connection_l_range"],
        dtype="int",
        perturb_rate=brain_mutation_config["perturb_rate"],
    )
    connection_a_mutation = get_single_param_mutation(
        value_range=brain_mutation_config["connection_a_range"],
        dtype="float",
        perturb_rate=brain_mutation_config["perturb_rate"],
    )
    connection_rt_mutation = get_single_param_mutation(
        brain_mutation_config["connection_rt_range"],
        dtype="int",
        perturb_rate=brain_mutation_config["perturb_rate"],
    )
    connection_dt_mutation = get_single_param_mutation(
        value_range=brain_mutation_config["connection_dt_range"],
        dtype="int",
        perturb_rate=brain_mutation_config["perturb_rate"],
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
    defalut_config = utils.get_default_config()
    return get_brain_mutation_from_brain_mutation_config(
        defalut_config["brain_mutation_config"]
    )


class UniformMutation(object):
    """
    definition of a single mutation of a single value, based on a uniform distribution of a value range. uses builtin
    random module
    """

    def __init__(
        self,
        value_range: list[float],
        dtype: str,
        perturb_rate: float = 0.0,
    ) -> None:
        """

        :param value_range: tuple of two numbers, the two value should be different.
        :param dtype: str, 'int' or 'float'.
            if 'int' random value will be drawn by random.randint()
            if 'float' random value will be drawn by random.uniform()
        :param perturb_rate: float, the probability that the parameter will be perturbed
            if it is selected for mutation. If perturbed, the updated value will be uniformly
            choosen in the range plus and minus 10% of value_range around the current value.
            if not perturbed, the updated value will be uniformly choosen from the full
            value_range.
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

        self.perturb_rate = np.clip(perturb_rate, 0, 1, dtype=np.float32)

    @property
    def value_span(self):
        return self.value_range[1] - self.value_range[0]

    def get_value(self, curr_value: Union[float, int], should_perturb: bool = True):
        """
        if self._dtype is 'int': uses random.randint() function
        if self._dtype is 'float': uses random.uniform() function

        :param curr_value: float or int, the value to be updated
        :param should_perturb: bool, if True, apply self.perturb_rate to perturb the curr_value
            the parameter will have self.perturb_rate chace to be perturbed
            If perturbed, the updated value will be uniformly choosen in the range plus and minus
                10% of value_range around the current value.
            if not perturbed, the updated value will be uniformly choosen from the full
            value_range.

        :return: a random value follow a uniform distribution with a range defined by self._value_range, including the
            start but excluding the end
        """

        if should_perturb and random.random() <= self.perturb_rate:
            low = curr_value - 0.1 * self.value_span
            high = curr_value + 0.1 * self.value_span
        else:
            low = self.value_range[0]
            high = self.value_range[1]

        low = max(low, self.value_range[0])
        high = min(high, self.value_range[1])

        if self.dtype == "int":
            return random.randint(int(np.floor(low)), int(np.ceil(high)) - 1)
        elif self.dtype == "float":
            return random.uniform(low, high)

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

    def get_mutated_baseline(self, neuron: neu.Neuron, should_perturb: bool = True):
        """
        return a mutated baseline rate by self.baseline_mutation
        """
        return (
            self.baseline_mutation.get_value(
                curr_value=neuron.baseline_rate,
                should_perturb=should_perturb,
            )
            if self.baseline_mutation is not None
            else None
        )

    def get_mutated_refractory(self, neuron: neu.Neuron, should_perturb: bool = True):
        """
        return a mutated refractory period by self.refractory_mutation
        """
        return (
            self.refractory_mutation.get_value(
                curr_value=neuron.refractory_period,
                should_perturb=should_perturb,
            )
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

    def get_mutated_gain(self, neuron: neu.Eye, should_perturb: bool = True):
        return (
            self.gain_mutation.get_value(
                curr_value=neuron.gain,
                should_perturb=should_perturb,
            )
            if self.gain_mutation is not None
            else None
        )


class ConnectionMutation(object):

    """
    definition of a connection mutation
    """

    def __init__(
        self,
        latency_mutation: UniformMutation = None,
        amplitude_mutation: UniformMutation = None,
        rise_time_mutation: UniformMutation = None,
        decay_time_mutation: UniformMutation = None,
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

    def get_mutated_latency(
        self, connection: conn.Connection, should_perturb: bool = True
    ):
        return (
            self.latency_mutation.get_value(
                curr_value=connection.latency,
                should_perturb=should_perturb,
            )
            if self.latency_mutation is not None
            else None
        )

    def get_mutated_amplitude(
        self, connection: conn.Connection, should_perturb: bool = True
    ):
        return (
            self.amplitude_mutation.get_value(
                curr_value=connection.amplitude,
                should_perturb=should_perturb,
            )
            if self.amplitude_mutation is not None
            else None
        )

    def get_mutated_rise_time(
        self, connection: conn.Connection, should_perturb: bool = True
    ):
        return (
            self.rise_time_mutation.get_value(
                curr_value=connection.rise_time,
                should_perturb=should_perturb,
            )
            if self.rise_time_mutation is not None
            else None
        )

    def get_mutated_decay_time(
        self, connection: conn.Connection, should_perturb: bool = True
    ):
        return (
            self.decay_time_mutation.get_value(
                curr_value=connection.decay_time,
                should_perturb=should_perturb,
            )
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
        random_fish_num_per_generation: int = 50,
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

        assert (
            random_fish_num_per_generation <= population_size * turnover_rate
        ), "number of random fish per generation should not exceed the number of new fish in each generation."

        self.population_size = population_size
        self.turnover_rate = turnover_rate
        self.neuron_mutation_rate = neuron_mutation_rate
        self.connection_mutation_rate = connection_mutation_rate
        self.brain_mutation = brain_mutation
        self.life_span_hard_threshold = life_span_hard_threshold
        self.movement_hard_threshold = movement_hard_threshold
        self.random_fish_num_per_generation = random_fish_num_per_generation
        self.generation_digits_num = generation_digits_num

    # @staticmethod
    # def _find_single_simulation_log_name(grp_root: h5py.Group):
    #     """
    #     check if there is one and only one simulation log group in the grp_root
    #     and return the simulation log group name
    #     """
    #     sim_log_ns = [s for s in grp_root.keys() if s[:11] == "simulation_"]

    #     if len(sim_log_ns) == 0:
    #         raise LookupError("Cannot find simulation log")
    #     elif len(sim_log_ns) > 1:
    #         raise LookupError("More than one simulation logs found.")

    #     return sim_log_ns[0]

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
            utils.get_generation_name(curr_generation_ind, self.generation_digits_num),
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

            simulation_logs = get_simulation_logs(fish_f)
            life_spans.append(
                np.mean(
                    [s.last_time_point for s in simulation_logs],
                ),
            )
            total_movements.append(
                np.mean(
                    [s.get_fish_total_moves(fish_name=fish_n) for s in simulation_logs],
                )
            )

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

        new_fish_number = (
            self.population_size - fishes.shape[0] - self.random_fish_num_per_generation
        )
        fishes["offspring_num"] = distrube_number(
            values=fishes["extra_life"],
            population_size=new_fish_number,
            temperature=max(
                1.0, np.mean(fishes["extra_life"])
            ),  # temperature to increase randomness
        )

        print("================== mother fish ==================")
        print(fishes)
        print("=================================================")

        return life_thr, fishes

    def generate_next_generation(
        self,
        base_folder: str,
        curr_generation_ind: int,
        brain_config: dict,  # brain config to generate new random fish
        fish_config: dict,  # fish config to generate new random fish
        should_perturb: bool = True,
    ):
        print(
            "\n======================================================================"
        )
        print(
            f"PopulationEvolution: generating fish for generation: {curr_generation_ind + 1} ..."
        )

        curr_gen_folder = os.path.join(
            base_folder,
            utils.get_generation_name(curr_generation_ind, self.generation_digits_num),
        )
        next_gen_folder = os.path.join(
            base_folder,
            utils.get_generation_name(
                curr_generation_ind + 1, self.generation_digits_num
            ),
        )
        os.mkdir(next_gen_folder)

        life_thr, fishes = self._calculate_offspring_num(
            base_folder=base_folder,
            curr_generation_ind=curr_generation_ind,
        )

        for fish_ind, fish_row in fishes.iterrows():
            mother_fish_path = os.path.join(
                curr_gen_folder, fish_row["fish_name"] + ".hdf5"
            )
            mother_fish_f = h5py.File(mother_fish_path, "a")
            mother_fish_name = [
                k for k in mother_fish_f.keys() if k.startswith("fish_")
            ]
            if len(mother_fish_name) != 1:
                raise LookupError(
                    f"{mother_fish_path} should have one and only one fish."
                )
            mother_fish_name = mother_fish_name[0]
            mother_fish = fi.load_fish_from_h5_group(mother_fish_f[mother_fish_name])
            mother_fish_gens = list(mother_fish_f["generations"][()])

            children_lst = []

            child_fish_f = h5py.File(
                os.path.join(next_gen_folder, mother_fish.name + ".hdf5"), "a"
            )
            mother_fish.to_h5_group(child_fish_f)
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
                    should_perturb=should_perturb,
                )
                child_fish_f = h5py.File(
                    os.path.join(next_gen_folder, child_fish.name + ".hdf5"), "a"
                )
                child_fish.to_h5_group(child_fish_f)
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
            f"PopulationEvolution: generating random fish for generation: {curr_generation_ind +  1} ..."
        )
        for _ in range(self.random_fish_num_per_generation):
            curr_brain = fn.generate_brain_from_brain_config(brain_config)
            curr_fish = fi.Fish(brain=curr_brain, **fish_config)
            rand_fish = mutate_fish(
                curr_fish,
                brain_mutation=self.brain_mutation,
                neuron_mutation_rate=1.0,  # fully random fish
                connection_mutation_rate=1.0,  # fully random fish
                should_perturb=False,
                verbose=False,
            )
            rand_fish_f = h5py.File(
                os.path.join(next_gen_folder, rand_fish.name + ".hdf5"), "a"
            )
            rand_fish.to_h5_group(rand_fish_f)
            rand_fish_f["generations"] = [curr_generation_ind + 1]
            rand_fish_f.close()
        print(
            f"PopulationEvolution: generating random fish for generation: {curr_generation_ind +  1} finished."
        )

        print(
            f"PopulationEvolution: fish generation for generation: {curr_generation_ind + 1} finished."
        )
        print("======================================================================")

        return next_gen_folder


def mutate_neuron(
    neuron: Union[neu.Neuron, neu.Muscle],
    neuron_mutation: NeuronMutation,
    should_perturb: bool = True,
) -> neu.Neuron:
    """
    mutate a neuron, can be Eye, Neuron or Muscle

    :param neuron: the initial little_fish.brain.neuron.Neuron object
    :param neuron_mutation: little_fish.core.evolution.NeuronMutation object
    :return: a mutated little_fish.brain.neuron.Neuron object
    """

    mutated_neuron = copy.deepcopy(neuron)

    mutated_baseline = neuron_mutation.get_mutated_baseline(
        neuron=neuron,
        should_perturb=should_perturb,
    )
    if mutated_baseline is not None:
        mutated_neuron.baseline_rate = mutated_baseline

    mutated_refractory = neuron_mutation.get_mutated_refractory(
        neuron=neuron,
        should_perturb=should_perturb,
    )
    if mutated_refractory is not None:
        mutated_neuron.refractory_period = mutated_refractory

    return mutated_neuron


def mutate_eye(
    eye: neu.Eye,
    eye_mutation: EyeMutation,
    should_perturb: bool = True,
) -> neu.Eye:
    """
    mutate a neuron, can be Eye, Neuron or Muscle

    :param neuron: the initial little_fish.brain.neuron.Eye object
    :param neuron_mutation: little_fish.core.evolution.NeuronMutation object
    :return: a mutated little_fish.brain.neuron.Eye object
    """

    mutated_eye = copy.deepcopy(eye)

    mutated_gain = eye_mutation.get_mutated_gain(
        neuron=eye,
        should_perturb=should_perturb,
    )
    if mutated_gain is not None:
        mutate_eye.gain = mutated_gain

    mutated_baseline = eye_mutation.get_mutated_baseline(
        neuron=eye,
        should_perturb=should_perturb,
    )
    if mutated_baseline is not None:
        mutated_eye.baseline_rate = mutated_baseline

    mutated_refractory = eye_mutation.get_mutated_refractory(
        neuron=eye,
        should_perturb=should_perturb,
    )
    if mutated_refractory is not None:
        mutated_eye.refractory_period = mutated_refractory

    return mutated_eye


def mutate_connection(
    connection: conn.Connection,
    connection_mutation: ConnectionMutation,
    should_perturb: bool = True,
) -> conn.Connection:
    """
    mutate a connection

    :param connection: the initial little_fish.core.fish.Connection object
    :param connection_mutation: little_fish.core.evolution.ConnectionMutation object
    :return: a mutated little_fish.core.fish.Connection object
    """

    mutated_connection = copy.deepcopy(connection)

    mutated_latency = connection_mutation.get_mutated_latency(
        connection=connection,
        should_perturb=should_perturb,
    )
    if mutated_latency is not None:
        mutated_connection.latency = mutated_latency

    mutated_amplitude = connection_mutation.get_mutated_amplitude(
        connection=connection,
        should_perturb=should_perturb,
    )
    if mutated_amplitude is not None:
        mutated_connection.amplitude = mutated_amplitude

    mutated_rise_time = connection_mutation.get_mutated_rise_time(
        connection=connection,
        should_perturb=should_perturb,
    )
    if mutated_rise_time is not None:
        mutated_connection.rise_time = mutated_rise_time

    mutated_decay_time = connection_mutation.get_mutated_decay_time(
        connection=connection,
        should_perturb=should_perturb,
    )
    if mutated_decay_time is not None:
        mutated_connection.decay_time = mutated_decay_time

    return mutated_connection


def mutate_brain(
    curr_brain: brain.Brain,
    brain_mutation: BrainMutation,
    neuron_mutation_rate: float = 0.01,
    connection_mutation_rate: float = 0.01,
    should_perturb: bool = True,
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
                curr_neuron,
                brain_mutation.eye_mutation,
                should_perturb=should_perturb,
            )

        elif curr_neuron.neuron_type in ["neuron", "muscle"]:
            if verbose:
                print(f"Evolution: mutating hidden neuron. Index: {mni}.")
            mutated_neurons.loc[mni, "neuron"] = mutate_neuron(
                curr_neuron,
                brain_mutation.neuron_mutation,
                should_perturb=should_perturb,
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
            curr_connection,
            brain_mutation.connection_mutation,
            should_perturb=should_perturb,
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
    should_perturb: bool = True,
    verbose=False,
) -> fi.Fish:
    mutated_brain = mutate_brain(
        curr_brain=fish.brain,
        brain_mutation=brain_mutation,
        neuron_mutation_rate=neuron_mutation_rate,
        connection_mutation_rate=connection_mutation_rate,
        should_perturb=should_perturb,
        verbose=verbose,
    )
    mother_name = fish.name
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


def run_evoluation(run_config):
    base_folder = run_config["simulation_config"]["data_folder"]

    generation_digits_num = run_config["simulation_config"]["generation_digits_num"]

    start_generation_ind = run_config["simulation_config"]["start_generation_ind"]
    if not utils.is_integer(start_generation_ind) or start_generation_ind < 0:
        raise ValueError(
            "PopulationEvolution: start_generation_ind should be a non-negative integer."
        )

    end_generation_ind = run_config["simulation_config"]["end_generation_ind"]
    if not utils.is_integer(end_generation_ind) or end_generation_ind < 0:
        raise ValueError(
            "PopulationEvolution: end_generation_ind should be a non-negative integer."
        )

    if start_generation_ind >= end_generation_ind:
        raise ValueError(
            "PopulationEvolution: start_generation_ind should be smaller than end_generation_ind."
        )

    brain_mutation = get_brain_mutation_from_brain_mutation_config(
        run_config["brain_mutation_config"],
    )

    if run_config["brain_config"] is None and run_config["brain_config_path"] is None:
        raise ValueError(
            "One of 'brain_config' and 'brain_config_path' should not be None."
        )
    elif (
        run_config["brain_config"] is not None
        and run_config["brain_config_path"] is None
    ):
        pass
    elif (
        run_config["brain_config"] is None
        and run_config["brain_config_path"] is not None
    ):
        curr_folder = os.path.dirname(os.path.realpath(__file__))
        brain_config_path = os.path.join(
            os.path.dirname(curr_folder), "configs", run_config["brain_config_path"]
        )
        with open(brain_config_path, "r") as f:
            brain_config = yaml.load(f, Loader=yaml.FullLoader)
        run_config["brain_config"] = brain_config["brain_config"]
    else:
        raise ValueError(
            "One of 'brain_config' and 'brain_config_path' should not be None."
        )

    evolution = PopulationEvolution(
        brain_mutation=brain_mutation,
        generation_digits_num=generation_digits_num,
        **run_config["evolution_config"],
    )

    start_generation_folder = os.path.join(
        base_folder,
        utils.get_generation_name(
            start_generation_ind,
            generation_digits_num,
        ),
    )

    # check and generate starting folder structure
    if start_generation_ind == 0:
        if not os.path.isdir(start_generation_folder):
            os.makedirs(start_generation_folder)

        if os.listdir(start_generation_folder):
            raise LookupError(
                "PopulationEvolution: start generation folder ({}) is not empty.".format(
                    os.path.realpath(start_generation_folder)
                )
            )

        # save run_config to the current folder
        with open(os.path.join(start_generation_folder, "run_config.yml"), "w") as f:
            yaml.dump(run_config, f)

        print(
            "\n======================================================================"
        )
        print("PopulationEvolution: generating fish for generation: 0 ...")

        for fish_ind in range(run_config["evolution_config"]["population_size"]):
            curr_brain = fn.generate_brain_from_brain_config(run_config["brain_config"])
            curr_fish = fi.Fish(brain=curr_brain, **run_config["fish_config"])
            rand_fish = mutate_fish(
                curr_fish,
                brain_mutation=brain_mutation,
                neuron_mutation_rate=1.0,  # fully random fish
                connection_mutation_rate=1.0,  # fully random fish
                should_perturb=False,
                verbose=False,
            )
            rand_fish_f = h5py.File(
                os.path.join(start_generation_folder, rand_fish.name + ".hdf5"), "a"
            )
            rand_fish.to_h5_group(rand_fish_f)
            rand_fish_f["generations"] = [0]
            rand_fish_f.close()

        print("PopulationEvolution: fish generation for generation: 0 finished.")
        print("======================================================================")

        sim.run_simulation_multi_thread(
            base_folder=base_folder,
            generation_ind=0,
            process_num=run_config["simulation_config"]["process_num"],
            simulation_length=run_config["simulation_config"]["simulation_length"],
            should_use_mini_map=run_config["terrain_config"]["should_use_mini_map"],
            terrain_size=run_config["terrain_config"]["terrain_size"],
            sea_portion=run_config["terrain_config"]["sea_portion"],
            terrain_filter_sigma=run_config["terrain_config"]["terrain_filter_sigma"],
            food_num=run_config["terrain_config"]["food_num"],
            simulation_num=run_config["simulation_config"]["simulation_num"],
        )

    else:
        if not os.listdir(start_generation_folder):
            raise LookupError(
                "PopulationEvolution: start generation folder ({}) should not be empty.".format(
                    os.path.realpath(start_generation_folder)
                )
            )

    # run simulation
    curr_gen_ind = start_generation_ind
    while curr_gen_ind < end_generation_ind:
        next_generation_folder = evolution.generate_next_generation(
            base_folder=base_folder,
            curr_generation_ind=curr_gen_ind,
            should_perturb=True,
            fish_config=run_config["fish_config"],
            brain_config=run_config["brain_config"],
        )

        # save run_config to the next generation folder
        with open(os.path.join(next_generation_folder, "run_config.yml"), "w") as f:
            yaml.dump(run_config, f)

        sim.run_simulation_multi_thread(
            base_folder=base_folder,
            generation_ind=curr_gen_ind + 1,
            process_num=run_config["simulation_config"]["process_num"],
            simulation_length=run_config["simulation_config"]["simulation_length"],
            terrain_size=run_config["terrain_config"]["terrain_size"],
            sea_portion=run_config["terrain_config"]["sea_portion"],
            terrain_filter_sigma=run_config["terrain_config"]["terrain_filter_sigma"],
            food_num=run_config["terrain_config"]["food_num"],
            simulation_num=run_config["simulation_config"]["simulation_num"],
        )

        curr_gen_ind += 1


if __name__ == "__main__":
    values = [6795, 6728, 5833, 5787, 5722, 3000, 1500, 4320, 1010, 0, 0]

    numbers = distrube_number(values=values, population_size=1000, temperature=1.0)
    print(numbers)

    numbers2 = distrube_number(values=values, population_size=1000, temperature=20.0)
    print(numbers2)

    numbers3 = distrube_number(
        values=values, population_size=1000, temperature=np.mean(values)
    )
    print(numbers3)

    print("for debug ...")
