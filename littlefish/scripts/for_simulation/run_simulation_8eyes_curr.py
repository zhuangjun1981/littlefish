import random
import numpy as np
from littlefish.core import utilities as utils
from littlefish.core import evolution as evo


def run():
    random_seed = random.randrange(2**32 - 1)
    random.seed(random_seed)
    np.random.seed(random_seed)

    base_folder = r"F:\little_fish_simulation_logs_8eyes_curr"

    run_config = utils.get_default_config()
    run_config["brain_config_path"] = "brain_config_8eyes_curr.yml"
    run_config["brain_config"] = None

    run_config["simulation_config"]["data_folder"] = base_folder
    run_config["simulation_config"]["start_generation_ind"] = 0
    run_config["simulation_config"]["end_generation_ind"] = 5

    run_config["simulation_config"]["food_num"] = 100
    run_config["fish_config"]["move_penalty_rate"] = 0.0005
    run_config["evolution_config"]["neuron_mutation_rate"] = 0.3
    run_config["evolution_config"]["connection_mutation_rate"] = 0.3
    # run_config["evolution_config"]["life_span_hard_threshold"] = 2000
    # run_config["terrain_config"]["sea_portion"] = 0.5

    run_config["brain_mutation_config"]["perturb_rate"] = 0.9
    run_config["brain_mutation_config"]["neuron_bl_range"] = [-0.5, 0.5]
    run_config["brain_mutation_config"]["muscle_bl_range"] = [-0.5, 0.5]
    run_config["brain_mutation_config"]["connection_l_range"] = [1, 20]
    run_config["brain_mutation_config"]["connection_rt_range"] = [1, 10]
    run_config["brain_mutation_config"]["connection_rt_range"] = [5, 20]

    evo.run_evoluation(run_config=run_config)


if __name__ == "__main__":
    run()
