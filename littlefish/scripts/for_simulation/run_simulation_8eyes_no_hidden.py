import random
import numpy as np
from littlefish.core import utilities as utils
from littlefish.core import evolution as evo


def run():
    random_seed = random.randrange(2**32 - 1)
    random.seed(random_seed)
    np.random.seed(random_seed)

    base_folder = r"F:\little_fish_simulation_logs_8eyes_no_hidden"

    run_config = utils.get_default_config()
    run_config["brain_config_path"] = "brain_config_8eyes_no_hidden.yml"
    run_config["brain_config"] = None

    run_config["simulation_config"]["data_folder"] = base_folder
    run_config["simulation_config"]["start_generation_ind"] = 132
    run_config["simulation_config"]["end_generation_ind"] = 150
    run_config["simulation_config"]["simulation_length"] = 20000
    run_config["simulation_config"]["simulation_num"] = 10
    run_config["simulation_config"]["start_health"] = 10

    run_config["terrain_config"]["should_use_mini_map"] = True
    run_config["terrain_config"]["mini_map_size"] = 9
    run_config["terrain_config"]["food_num"] = 1

    run_config["brain_mutation_config"]["perturb_rate"] = 0.5

    run_config["fish_config"]["max_health"] = 50
    run_config["fish_config"]["move_penalty_rate"] = 0.0001
    run_config["fish_config"]["land_penalty_rate"] = 5

    run_config["evolution_config"]["neuron_mutation_rate"] = 0.5
    run_config["evolution_config"]["connection_mutation_rate"] = 0.5
    run_config["evolution_config"]["life_span_hard_threshold"] = 1100
    run_config["evolution_config"]["random_fish_num_per_generation"] = 400
    run_config["evolution_config"]["stats_for_evaluation"] = "mean"
    # run_config["terrain_config"]["sea_portion"] = 0.5

    evo.run_evoluation(run_config=run_config)


if __name__ == "__main__":
    import pandas as pd

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    run()
