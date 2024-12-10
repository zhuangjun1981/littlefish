import h5py
import numpy as np
from littlefish.core import utilities as util
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
        action_potential_penalty_rate: float = 0.0,
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
        :param action_potential_penalty_rate: float, health decay per action potential, implement this can encourage
            sparse firing of the neurons.
        """

        # print('\nFish: Creating littlefish.core.fish.Fish object.')

        if name is None:
            self.name = "test_fish"
        else:
            self.name = name

        if mother_name is None:
            self.mother_name = ""
        else:
            self.mother_name = mother_name

        self.max_health = float(max_health)
        self.health_decay_rate = float(health_decay_rate)
        self.land_penalty_rate = float(land_penalty_rate)
        self.food_rate = float(food_rate)
        self.move_penalty_rate = float(move_penalty_rate)
        self.action_potential_penalty_rate = float(action_potential_penalty_rate)

        if brain is None:
            self.brain = Brain()
        else:
            brain.check_integrity()
            self.brain = brain

        # print('Fish: littlefish.core.fish.Fish object created successfully.')

    def set_name(self, name):
        self.name = name

    def set_brain(self, brain):
        brain.check_integrity()
        self.brain = brain

    def set_food_rate(self, food_rate):
        self.food_rate = float(food_rate)

    def set_health_decay_rate(self, health_decay_rate):
        self.health_decay_rate = health_decay_rate

    def set_move_penalty_rate(self, move_penalty_rate):
        self.move_penalty_rate = float(move_penalty_rate)

    def act(
        self,
        t_point,
        curr_position,
        curr_health,
        terrain_map,
        food_map=None,
        fish_map=None,
    ):
        """
        simulate the fish's action at a given time point

        :param t_point: non-negative int, time point
        :param curr_position: list or tuple of two non-negative integers, coordinates of fish center position,
                              [row, col]
        :param curr_health: positive float, health point at the beginning of t_point
        :param terrain_map: 2d array, with only 0s (water) and 1s (land). represents the land scape of the world
        :param food_map: 2d array, with only 0s (no food) and 1s (food). represents the distribution of food
        :param fish_map: not fully implemented right now.
        :return updated_health: float, health point at the end of t_point
        :return movement_attempt: list of two integers, the attempt the fish is trying to move [row_shift, col_shift].
                                  None if updated_health is below 0 (fish is dead).
        """

        updated_health = float(curr_health)

        # evaluate food
        if food_map is not None:
            body_food_overlap = self._eval_food(
                food_map=food_map, curr_position=curr_position
            )
            updated_health = self._eat_food(
                body_food_overlap=body_food_overlap, curr_health=updated_health
            )
            food_eated = body_food_overlap

            # update food map
            food_map[
                curr_position[0] - 1 : curr_position[0] + 2,
                curr_position[1] - 1 : curr_position[1] + 2,
            ] = 0
        else:
            food_eated = 0

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

        # update health
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
            movement_attempt = np.array([0, 0], dtype=np.uint8)
            action_potential_num = 0

        updated_health -= self.move_penalty_rate * (
            abs(movement_attempt[0]) + abs(movement_attempt[1])
        )

        updated_health -= self.action_potential_penalty_rate * action_potential_num

        return updated_health, movement_attempt, food_eated

    @staticmethod
    def _eval_terrain(terrain_map, curr_position):
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
    def _eval_food(food_map, curr_position):
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
    def _eval_fish(self, fish_map):
        """currently not implemented"""
        pass

    def _eat_food(self, body_food_overlap, curr_health):
        """
        count the number of food to be taken, add relevant HP to curr_health, but not exceed the maximum health
        """
        return min(self.max_health, curr_health + self.food_rate * body_food_overlap)

    def to_h5_group(self, h5_group):
        attributes = vars(self)

        for k, v in attributes.items():
            if k != "brain":
                h5_group.create_dataset(k, data=v)

        grp_brain = h5_group.create_group("brain")
        self.brain.to_h5_group(h5_group=grp_brain)


def load_fish_from_h5_group(h5_group: h5py.Group) -> Fish:
    """load Fish object from a hdf5 group."""

    fish_params = {}
    brain_grp = h5_group["brain"]
    fish_params["brain"] = load_brain_from_h5_group(brain_grp)

    for key, dset in h5_group.items():
        if key != "brain":
            if key in ["name", "mother_name"]:
                fish_params[key] = util.decode(dset[()])
            else:
                fish_params[key] = dset[()]
    return Fish(**fish_params)


def generate_fish_from_config(fish_config, brain_config) -> Fish:
    brain = generate_brain_from_brain_config(brain_config=brain_config)

    return Fish(brain=brain, **fish_config)
