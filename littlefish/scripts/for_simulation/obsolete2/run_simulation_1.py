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

    # the original run did not implement "move_penalty_rate"
    # so set it to 0.
    run_config["fish_config"]["move_penalty_rate"] = 0.0

    # this is not a good number, but was in the default config
    # and was used in the first simulation
    # now this value in default config has been changed
    # so this line will keep it the same as the original condition
    # for backward compatability
    run_config["fish_config"]["land_penalty_rate"] = 0.005

    sim.run_evoluation(run_config=run_config)


if __name__ == "__main__":
    run()
