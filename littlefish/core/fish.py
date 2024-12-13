import h5py
import numpy as np
from littlefish.core import utilities as utils
from littlefish.brain.brain import Brain
from littlefish.brain.functional import (
    load_brain_from_h5_group,
    generate_brain_from_brain_config,
)


class Fish:
    """
    the main fish class

    a 'fish' has a body occupies a 3x3 space.

    a 'fish' has health point (goes down over time and increases after eating a food). fish moves around in a 2d
    landscape (2-d binary map, 0 represents water, 1 represents land), when hits land, health point will go down
    quickly.

    the purpose of simulation is to train the fish use its eyes, brain and muscles to avoid land and look for food.

    the simulation works on "real-time" basis on a time unit axis (consider one time unit is equivalent to 0.1
    millisecond.

    :attr brain: a brain.Brain object
    :attr max_health: float, maximum health point a fish can have
    :attr health_decay_rate: float, the constant rate of health reduction, health point / time unit
    :attr land_penalty_rate: float, the penalty of health point, if the fish's body covers land pixels (1s) in
        the terrain map, health point / (pixel * time unit)
    :attr food_rate: float, the gaining of health point if fish's body covers food pixels (1s) in the food map,
        health point / pixel. the food after taken will disappear, so no health gaining is a transient event
    :attr move_penalty_rate: float, the penalty of healh point, if the fish move once this amount will be subtracted
        from its health.
    """

    def __init__(
        self,
        name: str = None,
        mother_name: str = None,
        brain: Brain = None,
        max_health: float = 20.0,
        health_decay_rate: float = 0.01,
        land_penalty_rate: float = 0.5,
        food_rate: float = 20.0,
        move_penalty_rate: float = 0.001,
        firing_penalty_rate: float = 0.0,
        generations: list[int] = [],
    ) -> None:
        """

        :param brain: a littlefish.brain.brain.Brain object
        :param max_health: float, maximum health point a fish can have
        :param health_decay_rate: float, the constant rate of health reduction, health point / time unit
        :param land_penalty_rate: float, the penalty of health point, if the fish's body covers land pixels (1s) in
            the terrain map, health point / (pixel * time unit)
        :param food_rate: float, the gaining of health point if fish's body covers food pixels (1s) in the food map,
            health point / pixel. the food after taken will disappear, so no health gaining is a transient event
        :param move_penalty_rate: float, the penalty of healh point, if the fish move once this amount will be subtracted
            from its health.
        :param firing_penalty_rate: float, health decay per action potential, implement this can encourage
            sparse firing of the neurons.
        :param generations: list[int], generations that the fish lasted during evolution
        """

        # print('\nFish: Creating littlefish.core.fish.Fish object.')

        if name is None:
            self.name = "unknown_fish"
        else:
            self.name = name

        if mother_name is None:
            self.mother_name = "unknown_mother_fish"
        else:
            self.mother_name = mother_name

        self.max_health = float(max_health)
        self.health_decay_rate = float(health_decay_rate)
        self.land_penalty_rate = float(land_penalty_rate)
        self.food_rate = float(food_rate)
        self.move_penalty_rate = float(move_penalty_rate)
        self.firing_penalty_rate = float(firing_penalty_rate)
        self.generations = generations

        if brain is None:
            self.brain = Brain()
        else:
            brain.check_integrity()
            self.brain = brain

        self.clear_simulation_cache()

        # print('Fish: littlefish.core.fish.Fish object created successfully.')

    def clear_simulation_cache(self):
        self.simulation_cache = {"health_history": None}

    def initiate_simulation(
        self, position: list[int], max_simulation_length: int
    ) -> None:
        """
        pre-allocate memorys for runing the simulation.
        first initiate simulation for the brain
        then initiate simulation fot the fish
        the self.simulation_cache is a dictionary:
          {
            "health_history": ndarray, shape: (simulation length, )
            "position_history": ndarray, shape: (simulation length, 2), first column: row_idx, second column: col_idx
          }
        """

        self.brain.initiate_simulation(max_simulation_length=max_simulation_length)

        health_history = np.zeros((max_simulation_length,), dtype=np.float32)
        health_history[0] = self.max_health

        position_history = np.zeros((max_simulation_length, 2), dtype=int)
        position_history[0, :] = position

        self.simulation_cache = {
            "health_history": health_history,
            "position_history": position_history,
            "total_moves": 0,
        }

    def set_name(self, name: str) -> None:
        self.name = name

    def set_brain(self, brain: Brain) -> None:
        brain.check_integrity()
        self.brain = brain

    def set_food_rate(self, food_rate: float) -> None:
        self.food_rate = float(food_rate)

    def set_health_decay_rate(self, health_decay_rate: float) -> None:
        self.health_decay_rate = health_decay_rate

    def set_move_penalty_rate(self, move_penalty_rate: float) -> None:
        self.move_penalty_rate = float(move_penalty_rate)

    def act(
        self,
        t_point: int,
        terrain_map: np.ndarray,
        food_map: np.ndarray = None,
        fish_map: np.ndarray = None,
    ):
        """
        simulate the fish's action at a given time point

        :param t_point: non-negative int, time point
        :param curr_health: positive float, health point at the beginning of t_point
        :param terrain_map: 2d array, with only 0s (water) and 1s (land). represents the land scape of the world
        :param food_map: 2d array, with only 0s (no food) and 1s (food). represents the distribution of food
        :param fish_map: not fully implemented right now.
        :return updated_health: float, health point at the end of t_point
        :return movement_attempt: list of two integers, the attempt the fish is trying to move [row_shift, col_shift].
                                  None if updated_health is below 0 (fish is dead).
        """

        curr_health = self.simulation_cache["health_history"][t_point]
        curr_position = self.simulation_cache["position_history"][t_point, :]

        # evaluate food
        if food_map is not None:
            food_eaten = self._eval_food(food_map=food_map, curr_position=curr_position)

            # update food map
            food_map[
                curr_position[0] - 1 : curr_position[0] + 2,
                curr_position[1] - 1 : curr_position[1] + 2,
            ] = 0
        else:
            food_eaten = 0

        # apply food reward
        updated_health = min(self.max_health, curr_health + self.food_rate * food_eaten)

        # evaluate the extend of how much of the fish is on the land
        body_land_overlap = self._eval_terrain(
            terrain_map=terrain_map, curr_position=curr_position
        )

        # update current health with land penalty
        updated_health -= body_land_overlap * self.land_penalty_rate

        # ----------------- not implemented --------------------------
        # if fish_map is not None:
        #     self._eval_fish(fish_map=fish_map)
        # ----------------- not implemented --------------------------

        # apply health decay
        updated_health -= self.health_decay_rate

        if updated_health > 0:  # still alive
            movement_attempt, action_potential_num = self.brain.act(
                t_point=t_point,
                body_position=curr_position,
                terrain_map=terrain_map,
                food_map=food_map,
                fish_map=fish_map,
            )
        else:
            movement_attempt = np.array([0, 0], dtype=int)
            action_potential_num = 0

        # apply movement penalty
        updated_health -= self.move_penalty_rate * (
            abs(movement_attempt[0]) + abs(movement_attempt[1])
        )

        # apply action potential penalty
        updated_health -= self.firing_penalty_rate * action_potential_num

        if t_point + 1 < len(self.simulation_cache["health_history"]):
            self.simulation_cache["health_history"][t_point + 1] = updated_health

        return updated_health, movement_attempt, food_eaten

    @staticmethod
    def _eval_terrain(terrain_map: np.ndarray, curr_position: list[int]) -> int:
        """
        Evaluate the coverage of fish body on terrain map, return the sum of all terrain pixels that are covered by
        the fish body
        """

        if terrain_map is None:
            raise ValueError("Fish: _eval_terrain failure. terrain_map is None.")
        else:
            curr_body = terrain_map[
                curr_position[0] - 1 : curr_position[0] + 2,
                curr_position[1] - 1 : curr_position[1] + 2,
            ]
            body_land_overlap = np.sum(curr_body.flat)
        return body_land_overlap

    @staticmethod
    def _eval_food(food_map: np.ndarray, curr_position: list[int]) -> int:
        """
        find out how many foods are covered by the fish body

        :param food_map: 2d array, binary map of current food
        :param curr_position: tuple of two positive ints, (row, col) of current location of fish
        :return: non-negative int, number of food taken
        """

        curr_body = food_map[
            curr_position[0] - 1 : curr_position[0] + 2,
            curr_position[1] - 1 : curr_position[1] + 2,
        ]
        body_food_overlap = np.sum(curr_body.flat)

        return body_food_overlap

    @staticmethod
    def _eval_fish(fish_map: np.ndarray, curr_position: list[int]):
        """currently not implemented"""
        pass

    def to_h5_group(
        self, h5_group: h5py.Group, should_save_cache: bool = False
    ) -> None:
        fish_group = h5_group.create_group(f"fish_{self.name}")

        attributes = vars(self)

        for k, v in attributes.items():
            if k not in ["brain", "simulation_cache"]:
                utils.save_h5_dataset(fish_group, k, v)

        grp_brain = fish_group.create_group("brain")
        self.brain.to_h5_group(h5_group=grp_brain)

        if should_save_cache and self.simulation_cache is not None:
            grp_sim_cache = fish_group.create_group("simulation_cache")
            self.save_simulation_cache_to_h5_group(h5_group=grp_sim_cache)

    def save_simulation_cache_to_h5_group(self, h5_group: h5py.Group) -> None:
        for k, v in self.simulation_cache.items():
            dset = utils.save_h5_dataset(h5_group, k, v)
            if k == "position_history":
                dset.attrs["data_format"] = "time_point x center position [row, col]"


def load_fish_from_h5_group(
    h5_group: h5py.Group,
    should_load_simulation_cache: bool = False,
) -> Fish:
    """load Fish object from a hdf5 group."""

    fish_params = {}
    brain_grp = h5_group["brain"]
    fish_params["brain"] = load_brain_from_h5_group(brain_grp)

    for key, dset in h5_group.items():
        if key not in ["brain", "simulation_cache"]:
            if key in ["name", "mother_name"]:
                fish_params[key] = utils.decode(dset[()])
            else:
                fish_params[key] = dset[()]

    # temporal bug fix
    if "action_potential_penalty_rate" in fish_params:
        fish_params["firing_penalty_rate"] = fish_params[
            "action_potential_penalty_rate"
        ]
        del fish_params["action_potential_penalty_rate"]

    fish = Fish(**fish_params)

    if should_load_simulation_cache and "simulation_cache" in h5_group:
        fish.simulation_cache = {}
        for k, v in h5_group["simulation_cache"].items():
            fish.simulation_cache[k] = v[()]

    return fish


def generate_fish_from_config(fish_config: dict, brain_config: dict) -> Fish:
    brain = generate_brain_from_brain_config(brain_config=brain_config)
    return Fish(brain=brain, **fish_config)
