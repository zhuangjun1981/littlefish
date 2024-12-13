import random
import numpy as np
from littlefish.core import utilities as utils
from littlefish.core import evolution as evo


def run():
    random_seed = random.randrange(2**32 - 1)
    random.seed(random_seed)
    np.random.seed(random_seed)

    base_folder = r"F:\little_fish_simulation_logs_4eyes_ff"
    run_config = utils.get_default_config()

    run_config["simulation_config"]["data_folder"] = base_folder
    run_config["simulation_config"]["start_generation_ind"] = 0
    run_config["simulation_config"]["end_generation_ind"] = 10

    run_config["evolution_config"]["population_size"] = 10

    evo.run_evoluation(run_config=run_config)


if __name__ == "__main__":
    run()
