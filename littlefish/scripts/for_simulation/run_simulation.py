import random
import numpy as np
import littlefish.core.evolution as evo

def run():

    start_generation_ind = 0
    end_generation_ind = 10

    random_seed = random.randrange(2 ** 32 - 1)
    random.seed(random_seed)
    np.random.seed(random_seed)

    data_folder = r"F:\little_fish_simulation_logs"
    population_size = 1000
    process_num = 10
    turnover_rate = 0.6
    simulation_length = 50000
    terrain_size = (64, 64)
    terrain_filter_sigma = 3.
    sea_portion = 0.5
    food_num = 50
    neuron_mutation_rate = 0.01
    connection_mutation_rate = 0.01


    pe = evo.PopulationEvolution(base_folder=data_folder, generation_digits_num=7)
    brain_mutation = evo.get_default_brain_mutation()

    pe.run(start_generation_ind=start_generation_ind, end_generation_ind=end_generation_ind,
           brain_mutation=brain_mutation, population_size=population_size, process_num=process_num,
           turnover_rate=turnover_rate, simulation_length=simulation_length, terrain_size=terrain_size,
           sea_portion=sea_portion, terrain_filter_sigma=terrain_filter_sigma, food_num=food_num,
           neuron_mutation_rate=neuron_mutation_rate, connection_mutation_rate=connection_mutation_rate)


if __name__ == '__main__':
    run()