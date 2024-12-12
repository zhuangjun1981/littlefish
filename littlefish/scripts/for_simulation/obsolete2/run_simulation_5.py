import random
import numpy as np
from littlefish.core import utilities as utils
from littlefish.core import simulation as sim


def run():
    random_seed = random.randrange(2**32 - 1)
    random.seed(random_seed)
    np.random.seed(random_seed)

    base_folder = r"F:\little_fish_simulation_logs_5"
    run_config = utils.get_default_config()

    run_config["simulation_config"]["data_folder"] = base_folder
    run_config["simulation_config"]["start_generation_ind"] = 61
    run_config["simulation_config"]["end_generation_ind"] = 100

    run_config["evolution_config"]["neuron_mutation_rate"] = 0.02
    run_config["evolution_config"]["connection_mutation_rate"] = 0.02
    run_config["evolution_config"]["turnover_rate"] = 0.8  # 0.8, 0.6
    run_config["evolution_config"]["movement_hard_threshold"] = 1

    run_config["fish_config"]["max_health"] = 20
    run_config["brain_config"]["hidden_neuron_nums"] = [20]
    run_config["brain_config"]["muscle_refractory_period"] = 10.0

    run_config["terrain_config"]["food_num"] = 50
    run_config["terrain_config"]["sea_portion"] = 0.90

    run_config["brain_mutation_config"]["eye_rp_r"] = [1.2, 5.0]
    run_config["brain_mutation_config"]["neuron_rp_r"] = [1.2, 5.0]
    run_config["brain_mutation_config"]["muscle_rp_r"] = [10.0, 50.0]
    run_config["brain_mutation_config"]["connection_l_r"] = [2, 50]
    run_config["brain_mutation_config"]["connection_rt_r"] = [1, 10]
    run_config["brain_mutation_config"]["connection_dt_r"] = [1, 20]

    sim.run_evoluation(run_config=run_config)


if __name__ == "__main__":
    run()
