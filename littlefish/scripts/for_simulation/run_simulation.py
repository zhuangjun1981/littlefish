import random
import numpy as np
from littlefish.core import utilities as utils
from littlefish.core import simulation as sim


def run():
    random_seed = random.randrange(2**32 - 1)
    random.seed(random_seed)
    np.random.seed(random_seed)

    base_folder = r"F:\little_fish_simulation_logs"
    run_config = utils.get_default_config()

    run_config["simulation_config"]["data_folder"] = base_folder
    run_config["simulation_config"]["start_generation_ind"] = 0
    run_config["simulation_config"]["end_generation_ind"] = 100

    sim.run_evoluation(run_config=run_config)


if __name__ == "__main__":
    run()
