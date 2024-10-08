import os
import sys
import h5py
import time
import datetime
import random
import inspect
import numpy as np
import littlefish.core.utilities as util
import littlefish.core.evolution as evo
import littlefish.core.fish as fi

data_folder = r"C:\little_fish_simulation_logs"

gen_num = 98
hard_thr_ratio = 0.5
soft_thr_ratio = 1.0
reproducing_rate = 0.002  # 0.002
random_seed = random.randrange(2**32 - 1)

neuron_mr = 0.01  # mutation rate of all neurons (including all eyes, hidden neurons and muscles)
eye_bl_r = (
    -0.1,
    0.1,
)  # baseline rate range of eyes, 0 to 0.1 action per time unit (100 spk/sec)
eye_rp_r = None  # refractory period range of eyes, not mutating right now
neuron_bl_r = (
    -0.1,
    0.1,
)  # baseline rate range of hidden neurons, 0 to 0.1 action per time unit (100 spk/sec)
neuron_rp_r = None  # refractory period range of hidden neurons, not mutating right now
muscle_bl_r = (
    -0.1,
    0.1,
)  # baseline rate range of muscles, 0 to 0.1 action per time unit (100 spk/sec)
muscle_rp_r = None  # refractory period range of muscles, not mutating right now
connection_mr = 0.01  # mutation rate of connections for each layer
connection_l_r = None  # latency range of connections, not mutating right now
connection_a_r = (-1.0, 1.0)  # amplitude range of connections, (-100~100 spk/sec)
connection_rt_r = None  # rise time range of connections, not mutating right now
connection_dt_r = None  # decay time range of connections, not mutating right now

random.seed(random_seed)
np.random.seed(random_seed)


def get_single_param_mutation(value_range, dtype):
    if value_range is None:
        mutation = None
    else:
        mutation = evo.UniformMutation(value_range=value_range, dtype=dtype)
    return mutation


eye_bl_mutation = get_single_param_mutation(eye_bl_r, "float")
eye_rp_mutation = get_single_param_mutation(eye_rp_r, "float")
neuron_bl_mutation = get_single_param_mutation(neuron_bl_r, "float")
neuron_rp_mutation = get_single_param_mutation(neuron_rp_r, "float")
muscle_bl_mutation = get_single_param_mutation(muscle_bl_r, "float")
muscle_rp_mutation = get_single_param_mutation(muscle_rp_r, "float")
connection_l_mutation = get_single_param_mutation(connection_l_r, "int")
connection_a_mutation = get_single_param_mutation(connection_a_r, "float")
connection_rt_mutation = get_single_param_mutation(connection_rt_r, "int")
connection_dt_mutation = get_single_param_mutation(connection_dt_r, "int")

eye_mutation = evo.NeuronMutation(
    baseline_mutation=eye_bl_mutation, refractory_mutation=eye_rp_mutation
)
neuron_mutation = evo.NeuronMutation(
    baseline_mutation=neuron_bl_mutation, refractory_mutation=neuron_rp_mutation
)
muscle_mutation = evo.NeuronMutation(
    baseline_mutation=muscle_bl_mutation, refractory_mutation=muscle_rp_mutation
)
connection_mutation = evo.ConnectionMutation(
    latency_mutation=connection_l_mutation,
    amplitude_mutation=connection_a_mutation,
    rise_time_mutation=connection_rt_mutation,
    decay_time_mutation=connection_dt_mutation,
)

brain_mutation = evo.BrainMutation(
    neuron_mutation_rate=neuron_mr,
    eye_mutation=eye_mutation,
    neuron_mutation=neuron_mutation,
    muscle_mutation=muscle_mutation,
    connection_mutation_rate=connection_mr,
    connection_mutation=connection_mutation,
)

curr_gen_folder = os.path.join(data_folder, "generation_" + util.int2str(gen_num, 6))
next_gen_folder = os.path.join(
    data_folder, "generation_" + util.int2str(gen_num + 1, 6)
)
all_mother_fish_lst = [f for f in os.listdir(curr_gen_folder) if f[0:5] == "fish_"]
print("all mother fish:")
print("\n".join(all_mother_fish_lst))

if not os.path.isdir(next_gen_folder):
    os.mkdir(next_gen_folder)

for mother_fish_ind, mother_fish_fn in enumerate(all_mother_fish_lst):
    print("\n=========================================================================")
    print(
        "processing mother fish: {}. {} / {} ".format(
            mother_fish_fn, mother_fish_ind + 1, len(all_mother_fish_lst)
        )
    )

    mother_fish_f = h5py.File(os.path.join(curr_gen_folder, mother_fish_fn))
    mother_fish = fi.Fish.from_h5_group(mother_fish_f["fish"])
    mother_sim_ns = [sim for sim in mother_fish_f.keys() if sim[0:11] == "simulation_"]
    mother_life_spans = []

    for mother_sim_n in mother_sim_ns:
        curr_sim_log_grp = mother_fish_f[mother_sim_n]["simulation_log"]
        mother_life_spans.append(curr_sim_log_grp["last_time_point"].value)

    default_life_span = int(
        mother_fish.get_max_health() / mother_fish.get_health_decay_rate()
    )
    offspring_num = evo.get_offspring_num(
        mother_life_spans=mother_life_spans,
        hard_thr=int(default_life_span * hard_thr_ratio),
        soft_thr=int(default_life_span * soft_thr_ratio),
        reproducing_rate=reproducing_rate,
    )

    print(
        "life spans: {} time unit. Spawning {} child(ren).".format(
            mother_life_spans, offspring_num
        )
    )
    print("=========================================================================\n")
    children_lst = []
    for i in range(offspring_num):
        child_fish = evo.mutate_fish(fish=mother_fish, brain_mutation=brain_mutation)

        # ========= set muscle refractory period to 10, and child fish food rate to 20 ==========
        # curr_brain = child_fish.get_brain()
        # for neuron_ind in range(len(curr_brain.get_neurons())):
        #     if curr_brain.get_neurons().loc[neuron_ind, 'neuron'].get_neuron_type() == 'muscle':
        #         curr_brain.get_neurons().loc[neuron_ind, 'neuron'].set_refractory_period(20)
        # child_fish.set_brain(brain=curr_brain)
        # child_fish.set_food_rate(food_rate=20.)
        # =======================================================================================

        child_fish_f = h5py.File(
            os.path.join(next_gen_folder, child_fish.name + ".hdf5")
        )
        child_fish_grp = child_fish_f.create_group("fish")
        child_fish.to_h5_group(child_fish_grp)
        child_fish_f["generation"] = gen_num + 1
        child_fish_f.close()
        children_lst.append(child_fish.name)
        time.sleep(1.0)

    ng_grp = mother_fish_f.create_group(
        "next_generation_" + datetime.datetime.now().strftime("%y%m%d_%H_%M_%S")
    )
    ng_grp["children_list"] = [c.encode("UTF-8") for c in children_lst]
    ng_grp["random_seed"] = random_seed
    ng_grp["script_text"] = inspect.getsource(sys.modules[__name__])

    mother_fish_f.close()
