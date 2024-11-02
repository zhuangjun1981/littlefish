import yaml
from littlefish.core import fish
from littlefish.core import utilities as utils
from littlefish.core import evolution as evo


run_config = utils.get_default_config()
run_config["simulation_config"]["data_folder"] = r"F:\little_fish_simulation_logs_2"
run_config["simulation_config"]["process_num"] = 1
run_config["evolution_config"]["population_size"] = 10


# run_config["brain_config"]["hidden_neuron_nums"] = [8, 8]


evo.run_evoluation(run_config)
