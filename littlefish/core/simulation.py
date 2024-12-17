import os
import time
import datetime
import random
import h5py
from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import littlefish.core.utilities as utils
import littlefish.core.terrain as tr
import littlefish.core.fish as fi

from littlefish.core.terrain import BinaryTerrain
from littlefish.core.fish import Fish
from littlefish.brain.functional import generate_brain_from_brain_config


class Simulation(object):
    """
    Simulation class takes fish(s) and terrain to run the simulation of a fish's activity during its life
    """

    def __init__(
        self,
        terrain: BinaryTerrain,
        fish_list: list[Fish],
        simulation_length: int = 100,
        food_num: int = 5,
    ) -> None:
        """
        designed to run only once after creation

        :param terrain: terrain object, current terrain.terrain_2d.BinaryTerrain object
        :param fish_list: list of fish object (fish.Fish class)
        :param simulation_length: positive integer, number of time points of the simulation
        :param food_num: positive integer, number of food pixels in food map
        :return: None
        """

        # simulation status: 0: not initialized yet;
        #                    1: started initialization
        #                    2: initialization finished
        #                    3: started simulation
        #                    4: after simulation
        self.simulation_status = 0
        self.terrain = terrain
        self.fish_list = fish_list
        self.simulation_length = int(simulation_length)
        self.food_num = int(food_num)

        assert self.food_num >= 0, "'food_num' should be a non-negative integer."

    def initiate_simulation(self) -> None:
        """
        initiate simulation
          1. check simulation status
          2. create empty food map
          3. generate self.simulation_cache = {"message": "", "food_position_history": 3d array}
          4. generate random starting postions for each fish
          5. for each fish, initiate simulation
          6. update simulation status
        """

        if self.simulation_status == 0:
            self.simulation_status = 1

            #  generate food map, food position history and initial food positions
            self.food_map = np.zeros(self.terrain.terrain_map.shape, dtype=int)

            self.simulation_cache = {
                "message": "",
                "food_pos_history": np.zeros(
                    (self.simulation_length, self.food_num, 2), dtype=int
                ),
            }

            #  generate non-overlapping positions for all fish
            fish_start_positions = self.terrain.generate_fish_starting_position(
                len(self.fish_list)
            )

            #  for each fish generate action histories, psp waveforms and life history
            for i, fish in enumerate(self.fish_list):
                fish.initiate_simulation(
                    position=fish_start_positions[i],
                    max_simulation_length=self.simulation_length,
                )

            #  simulation initialization finished, change simulation status
            self.simulation_status = 2

        else:
            raise RuntimeError(
                "Simulation: Cannot initiate simulation. Already initialized."
            )

    @property
    def fish_names(self):
        return [f.get_name() for f in self.fish_list]

    def generating_fish_map(self, time_point: int, is_plot: bool = False):
        """
        generating fish map containing all fishes according to self._fish_list at time_point

        :param time_point: non-negative int, time_point in the simulation
        :param is_plot: bool

        return: fish_map, 2d array, uint16,
        """
        n_rows, n_cols = self.terrain.terrain_map.shape
        if self.simulation_status in [2, 3, 4]:
            fish_map = np.zeros((n_rows, n_cols), dtype=int)
            for fish in self.fish_list:
                if fish.simulation_cache["health_history"][time_point] > 0.0:
                    curr_row, curr_col = fish.simulation_cache["position_history"][
                        time_point
                    ]
                    fish_map[
                        max(0, curr_row - 1) : min(n_rows - 1, curr_row + 2),
                        max(0, curr_col - 1) : min(n_cols - 1, curr_col + 2),
                    ] += 1

            if is_plot:
                f = plt.figure(figsize=(8, 7))
                ax = f.add_subplot(111)
                fig = ax.imshow(fish_map, cmap="magma", interpolation="nearest")
                f.colorbar(fig)
                ax.set_title("fish map at time: {}".format(time_point))
                plt.show()

            return fish_map
        else:
            raise RuntimeError(
                "Simulation: Cannot generate fish_map. Simulation not initialized properly."
            )

    def get_fish_health_status(self, t_point: int):
        """
        check the health status of all the fish in the simulation at the time point, t_point.
        if the time point is not simulated (at least one fish position is [0, 0]), an error will raise.

        :param t_point: non-negative integer, time point to evaluate
        :return: dataframe, with index = fish_name, column=['health']
        """

        # health_status = pd.DataFrame(index=self.fish_names, columns=["health"])

        fish_names = []
        healths = []

        for fish in self.fish_list:
            fish_names.append(fish.name)
            if np.array_equal(
                fish.simulation_cache["position_history"][t_point], [0, 0]
            ):
                raise RuntimeError(
                    f"fish: {fish.name} at time point {t_point} not simulated yet."
                )
            else:
                healths.append(fish.simulation_cache["health_history"][t_point])

        return pd.DataFrame({"fish_name": fish_names, "health": healths})

    def is_all_fish_dead(self, t_point):
        """
        check if the health of all fish are no higher than 0. at t_point in the simulation
        if the time point is not simulated (at least one fish position is [0, 0]), an error will raise.

        :param t_point: non-negative integer, time point to evaluate
        :return: bool
        """

        health_status = self.get_fish_health_status(t_point=t_point)
        return np.all(health_status["health"] <= 0.0)

    def _move_fish(
        self, curr_fish: Fish, curr_t: int, movement_attempt: list[int]
    ) -> list[int]:
        """
        move the fish based on the movement_attempt and return if the fish
        has moved.

        Update position in the simulation cache and return the new position
        """

        curr_pos = curr_fish.simulation_cache["position_history"][curr_t]

        # no movement attempt
        if np.array_equal(movement_attempt, [0, 0]):
            if curr_t + 1 < self.simulation_length:
                curr_fish.simulation_cache["position_history"][curr_t + 1] = curr_pos
            return curr_pos
        else:
            # update fish's body center postion at curr_t + 1
            new_pos_row = curr_pos[0] + movement_attempt[0]
            new_pos_row = max([1, new_pos_row])
            new_pos_row = min([new_pos_row, self.terrain.terrain_map.shape[0] - 2])

            new_pos_col = curr_pos[1] + movement_attempt[1]
            new_pos_col = max([1, new_pos_col])
            new_pos_col = min([new_pos_col, self.terrain.terrain_map.shape[1] - 2])

            new_pos = [new_pos_row, new_pos_col]

            if curr_t + 1 < self.simulation_length:
                curr_fish.simulation_cache["position_history"][curr_t + 1] = new_pos

            return new_pos

    def run(self, verbose=1):
        """

        :param verbose:
        :return: print out message as a string
        """

        if self.simulation_status == 2:
            # at t0
            self.simulation_status = 3
            t0 = time.time()
            start_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
            self.name = f"simulation_{start_time_str}"

            if verbose > 1:
                curr_msg = f"\nstart of simulation. start time: {start_time_str}"
                print(curr_msg)
                self.simulation_cache["message"] += curr_msg

            alive_fish_list = list(self.fish_list)
            curr_t = 0
            curr_progress = (
                -1
            )  # percentage finished, for printing the simulation progress

            while len(alive_fish_list) > 0 and curr_t < self.simulation_length:
                # print simulation progress
                if (verbose > 1) and (self.simulation_length > 100):
                    if curr_t // (self.simulation_length // 10) > curr_progress:
                        curr_msg = (
                            f"\n{time.time() - t0:09.2f} second: "
                            f"{(curr_t // (self.simulation_length // 10)) * 10:2d} %"
                        )
                        print(curr_msg)
                        self.simulation_cache["message"] += "\n" + curr_msg
                        curr_progress = curr_t // (self.simulation_length // 10)

                # update food position
                fish_positions = [
                    fish.simulation_cache["position_history"][curr_t, :]
                    for fish in alive_fish_list
                ]
                curr_food_positions = self.terrain.update_food_map(
                    self.food_num, self.food_map, fish_positions=fish_positions
                )
                self.simulation_cache["food_pos_history"][curr_t] = curr_food_positions

                dead_fish_list = []  # list of fish going to die at curr_t

                for curr_fish in alive_fish_list:  # loop through current live fish
                    updated_health, movement_attempt, food_eated = curr_fish.act(
                        t_point=curr_t,
                        terrain_map=self.terrain.terrain_map,
                        food_map=self.food_map,
                        fish_map=None,
                    )

                    # print eat food message
                    if (food_eated > 0) and (verbose > 1):
                        curr_msg = (
                            f"Time: {curr_t:08d}; Fish: {curr_fish.name}; "
                            f"eated {food_eated} food pellet(s). HP: "
                            f"{curr_fish.simulation_cache['health_history'][curr_t]:3.4f} -> "
                            f"{curr_fish.simulation_cache['health_history'][curr_t + 1]:3.4f}"
                        )
                        print(curr_msg)
                        self.simulation_cache["message"] += "\n" + curr_msg

                    if updated_health > 0:  # if not dead
                        curr_position = curr_fish.simulation_cache["position_history"][
                            curr_t
                        ]
                        new_position = self._move_fish(
                            curr_fish=curr_fish,
                            curr_t=curr_t,
                            movement_attempt=movement_attempt,
                        )

                        if not np.array_equal(curr_position, new_position):
                            curr_fish.simulation_cache["total_moves"] += 1

                        if verbose > 1:
                            curr_msg = (
                                f"Time: {curr_t+1:08d}; Fish: {curr_fish.name}; "
                                f"health: {updated_health:3.4f}; "
                                f"position: {new_position}"
                            )
                            print(curr_msg)
                            self.simulation_cache["message"] += "\n" + curr_msg

                    else:  # if dead
                        if verbose > 1:
                            curr_msg = (
                                f"Time: {curr_t:08d}; Fish: {curr_fish.name} died. "
                                f"Total number of movements: {curr_fish.simulation_cache['total_moves']}"
                            )
                            print(curr_msg)
                            self.simulation_cache["message"] += "\n" + curr_msg
                        curr_fish.simulation_cache[
                            "health_history"
                        ] = curr_fish.simulation_cache["health_history"][: curr_t + 1]
                        curr_fish.simulation_cache[
                            "position_history"
                        ] = curr_fish.simulation_cache["position_history"][: curr_t + 1]
                        dead_fish_list.append(curr_fish)

                for dead_fish in dead_fish_list:
                    alive_fish_list.remove(dead_fish)

                curr_t += 1

            else:
                self.simulation_cache["last_time_point"] = curr_t
                self.simulation_cache["ending_time"] = datetime.datetime.now().strftime(
                    "%y%m%d_%H_%M_%S"
                )

                if curr_t == self.simulation_length - 1:
                    if verbose > 1:
                        curr_msg = "Simulation: End of Simulation. Prespecified simulation length reached.\n"

                        for sur_fish in alive_fish_list:
                            curr_msg = (
                                f"\nSurvived fish: {sur_fish.name}. Final health: "
                                f"{sur_fish.simulation_cache['health_history'][curr_t]:3.4f}. "
                                f"Total number of movements: {sur_fish.simulation_cache['total_moves']}."
                            )
                        print(curr_msg)
                        self.simulation_cache["message"] += "\n" + curr_msg

                elif len(alive_fish_list) == 0:
                    self.simulation_cache["food_pos_history"] = self.simulation_cache[
                        "food_pos_history"
                    ][0 : curr_t + 1]
                    if verbose > 1:
                        curr_msg = (
                            f"\nSimulation: End of Simulation. All fish are dead. "
                            f"Last simulated time point: {curr_t}"
                        )
                        print(curr_msg)
                        self.simulation_cache["message"] += "\n" + curr_msg

            self.simulation_status = 4

        else:
            raise RuntimeError(
                "Simulation: Cannot run simulation. Simulation not initialized properly."
            )

        return

    def to_h5_group(
        self,
        h5_group: h5py.Group,
        should_save_psp_waveforms: bool = False,
    ) -> None:
        """
        save simulation results into a hdf5 group

        :param h5_group, hdf5.Group
        :param should_save_psp_waveforms: bool
        :return: None
        """

        if self.simulation_status != 4:
            raise RuntimeError(
                "Simulation: Cannot save save_log. Simulation has not run yet."
            )

        save_group = h5_group.create_group(self.name)
        save_group.create_dataset("name", data=self.name)
        save_group.create_dataset(
            "terrain_map", data=self.terrain.terrain_map, dtype=np.uint8
        )
        save_group.create_dataset(
            "max_simulation_length", data=self.simulation_length, dtype=np.int32
        )
        save_group.create_dataset("food_num", data=self.food_num, dtype=np.int32)
        save_group.create_dataset(
            "sea_portion", data=self.terrain.get_sea_portion(), dtype=np.float32
        )

        sim_cache_group = save_group.create_group("simulation_cache")
        for k, v in self.simulation_cache.items():
            dset = utils.save_h5_dataset(sim_cache_group, k, v)
            if k == "food_pos_history":
                dset.attrs[
                    "data_format"
                ] = "time_points x food_num x food position [row, col]"

        for fish in self.fish_list:
            fish_grp = save_group.create_group(f"fish_{fish.name}")
            fish.save_simulation_cache_to_h5_group(fish_grp)
            brain_grp = fish_grp.create_group("brain_simulation_cache")
            fish.brain.save_simulation_cache_to_h5_group(
                brain_grp, should_save_psp_waveforms=should_save_psp_waveforms
            )


def simulate_one_fish(
    fish_path: str,
    simulation_length: int,
    simulation_num: int,
    terrain: BinaryTerrain,
    food_num: int,
    hard_thr: int = 0,
    fish_ind: int = 0,
    fish_num: int = 0,
    verbose: int = 0,
):
    """
    the function to simulate a fish's lives in multiple terrain maps. the simulation results will be saved in the same
    input path 'fish_path'.

    :param fish_path: str, the path of an hdf5 file, with has a group 'fish' in root, which contains data of one and
        only one fish. Ideally generated by littlefish.core.fish.to_h5_group() method.
    :param simulation_length: positive int, top length of each simulation in time unit.
    :param simulation_num: positive int, number of terrain maps to simulate
    :param terrain: littlefish.core.terrain.BinaryTerrain object
    :param food_num: non-negative int, how many food pellet(s) are presented at any given time in terrain map.
    :param hard_thr: positive int, fish should have a life span longer than this threshold to be kept for simulation.
    :param fish_ind: non-negative int, the index of the current fish in the whole population (generation). just for
        printout purpose, if not known, keep 0.
    :param fish_num: non-negative int, the total number of fish in the whole population (generation). just for
        printout purpose, if not known, keep 0.
    :return: None
    """

    curr_fish_f = h5py.File(fish_path, "a")
    fish_names = [k[5:] for k in curr_fish_f.keys() if k.startswith("fish_")]

    if len(fish_names) == 0:
        raise ValueError(f"{fish_path}, cannot find fish group")
    elif len(fish_names) > 1:
        raise ValueError("found more than one fish groups")
    else:
        fish_name = fish_names[0]

    curr_fish = fi.load_fish_from_h5_group(curr_fish_f[f"fish_{fish_name}"])

    for sim_ind in range(simulation_num):
        curr_seed = random.randrange(2**31 - 1)
        random.seed(curr_seed)
        np.random.seed(curr_seed)

        if verbose:
            print(
                "\n\n========================= {}/{}; fish: {} ; simulation: {}/{} start ===========================".format(
                    fish_ind + 1, fish_num, curr_fish.name, sim_ind + 1, simulation_num
                )
            )

        curr_simulation = Simulation(
            terrain=terrain,
            fish_list=[curr_fish],
            simulation_length=simulation_length,
            food_num=food_num,
        )

        curr_simulation.initiate_simulation()
        curr_simulation.run(verbose=verbose)
        curr_simulation.simulation_cache["random_seed"] = curr_seed
        curr_simulation.simulation_cache["numpy_random_seed"] = curr_seed
        # curr_simulation.simulation_cache["script_txt"] = inspect.getsource(sys.modules[__name__])

        curr_simulation.to_h5_group(curr_fish_f, should_save_psp_waveforms=False)

        if verbose:
            print(
                "\n========================== {}/{}; fish: {}; simulation: {}/{} end ============================".format(
                    fish_ind + 1, fish_num, curr_fish.name, sim_ind + 1, simulation_num
                )
            )

        if curr_simulation.simulation_cache["last_time_point"] < hard_thr:
            if verbose:
                print(
                    "\nSimulation: fish life span was less than the hard threshold. end the simulation of current"
                    "fish: {}".format(curr_fish.name)
                )
            break

    curr_fish_f.close()


def simulate_fish_multiprocessing(simulation_params):
    """
    warpper of "simulate_one_fish" for multi-processing
    """

    (
        f_path,
        fish_ind,
        fish_num,
        simulation_length,
        terrain,
        food_num,
        simulation_num,
    ) = simulation_params
    simulate_one_fish(
        fish_path=f_path,
        simulation_length=simulation_length,
        simulation_num=simulation_num,
        terrain=terrain,
        food_num=food_num,
        hard_thr=0,
        fish_ind=fish_ind,
        fish_num=fish_num,
    )


def run_simulation_multi_thread(
    base_folder,
    generation_ind,
    process_num=6,
    simulation_num=1,
    simulation_length=50000,
    should_use_mini_map=True,
    terrain_size=(64, 64),
    sea_portion=0.5,
    terrain_filter_sigma=3.0,
    food_num=50,
    generation_digits_num=7,
):
    print("\n======================================================================")
    print(
        "PopulationEvolution: start simulating generation: {} ...".format(
            generation_ind
        )
    )

    gen_folder = os.path.join(
        base_folder, utils.get_generation_name(generation_ind, generation_digits_num)
    )
    fish_ns = [
        f for f in os.listdir(gen_folder) if f[:5] == "fish_" and f[-5:] == ".hdf5"
    ]
    fish_ns.sort()
    fish_ps = [os.path.join(gen_folder, f) for f in fish_ns]

    # predetermined terrain for evaluating fish, much more efficient
    if should_use_mini_map:
        ter_map = np.zeros((11, 11), dtype=np.uint8)
        ter_map[:, :2] = 1
        ter_map[:, -2:] = 1
        ter_map[:2, :] = 1
        ter_map[-2:, :] = 1
        ter = tr.BinaryTerrain(ter_map)
    else:
        # # random terrain for evaluation fish
        tg = tr.TerrainGenerator(size=terrain_size, sea_portion=sea_portion)
        ter = tr.BinaryTerrain(tg.generate_binary_map(sigma=terrain_filter_sigma))

    sim_params = []
    for fish_ind, fish_p in enumerate(fish_ps):
        sim_params.append(
            (
                fish_p,
                fish_ind,
                len(fish_ps),
                simulation_length,
                ter,
                food_num,
                simulation_num,
            )
        )

    with Pool(process_num) as p:
        p.map(simulate_fish_multiprocessing, sim_params)

    # for sim_param in sim_params:
    #     simulate_fish_multiprocessing(sim_param)

    print(
        "PopulationEvolution: simulation of generation: {} finished.".format(
            generation_ind
        )
    )
    print("======================================================================")
