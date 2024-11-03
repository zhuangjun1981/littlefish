import os, h5py
from littlefish.core import fish
from littlefish.core import utilities as utils
from littlefish.core import evolution as evo

base_folder = r"F:\little_fish_simulation_logs_2"

run_config = utils.get_default_config()
run_config["simulation_config"]["data_folder"] = base_folder
run_config["simulation_config"]["process_num"] = 1
run_config["simulation_config"]["start_generation_ind"] = 0
run_config["simulation_config"]["end_generation_ind"] = 3
run_config["evolution_config"]["population_size"] = 10


generation_digits_num = run_config["simulation_config"]["generation_digits_num"]


# run_config["brain_config"]["hidden_neuron_nums"] = [8, 8]

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


brain_mutation = evo.get_brain_mutation_from_brain_mutation_config(
    run_config["brain_mutation_config"],
)

evolution = evo.PopulationEvolution(
    brain_mutation=brain_mutation,
    generation_digits_num=generation_digits_num,
    **run_config["evolution_config"]
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

    print("\n======================================================================")
    print("PopulationEvolution: generating fish for generation: 0 ...")

    for fish_ind in range(run_config["evolution_config"]["population_size"]):
        curr_brain = fish.genearte_brain_from_brain_config(run_config["brain_config"])
        curr_fish = fish.Fish(brain=curr_brain, **run_config["fish_config"])
        rand_fish = evo.mutate_fish(
            curr_fish,
            brain_mutation=brain_mutation,
            neuron_mutation_rate=run_config["evolution_config"]["neuron_mutation_rate"],
            connection_mutation_rate=run_config["evolution_config"][
                "connection_mutation_rate"
            ],
            verbose=False,
        )
        rand_fish_f = h5py.File(
            os.path.join(start_generation_folder, rand_fish.name + ".hdf5"), "a"
        )
        rand_fish_grp = rand_fish_f.create_group("fish")
        rand_fish.to_h5_group(rand_fish_grp)
        rand_fish_f["generations"] = [0]
        rand_fish_f.close()

    print("PopulationEvolution: fish generation for generation: 0 finished.")
    print("======================================================================")


# evolution.generate_next_generation(
#     base_folder=run_config["simulation_config"]["data_folder"],
#     curr_generation_ind=3,
#     simulation_ind=0
# )
