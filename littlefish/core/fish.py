# from __future__ import (absolute_import, division,
#                         print_function, unicode_literals)
# from builtins import *
import numpy as np
from littlefish.core import utilities as util
from littlefish.brain.functional import genearte_brain_from_brain_config


def generate_standard_fish():
    default_config = util.get_default_config()

    brain = genearte_brain_from_brain_config(default_config["brain_config"])

    return Fish(brain=brain, **default_config["fish_config"])


class Fish(object):
    """
    the main fish class

    a 'fish' has a body occupies a 3x3 space.

    a 'fish' has health point (goes down over time and increases after eating a food). fish moves around in a 2d
    landscape (2-d binary map, 0 represents water, 1 represents land), when hits land, health point will go down
    quickly.

    the purpose of simulation is to train the fish use its eyes, brain and muscles to avoid land and look for food.

    the simulation works on "real-time" basis on a time unit axis (consider one time unit is equivalent to 0.1
    millisecond.

    :attr _brain: a brain.Brain object
    :attr _max_health: float, maximum health point a fish can have
    :attr _health_decay_rate: float, the constant rate of health reduction, health point / time unit
    :attr _land_penalty_rate: float, the penalty of health point, if the fish's body covers land pixels (1s) in
        the terrain map, health point / (pixel * time unit)
    :attr _food_rate: float, the gaining of health point if fish's body covers food pixels (1s) in the food map,
        health point / pixel. the food after taken will disappear, so no health gaining is a transient event
    :attr _move_penalty_rate: float, the penalty of healh point, if the fish move once this amount will be subtracted
        from its health.
    """

    def __init__(
        self,
        name=None,
        mother_name=None,
        brain=None,
        max_health=100.0,
        health_decay_rate=0.0001,
        land_penalty_rate=0.005,
        food_rate=20.0,
        move_penalty_rate=0.001,
    ):
        """

        :param brain: a brain.Brain object
        :param max_health: float, maximum health point a fish can have
        :param health_decay_rate: float, the constant rate of health reduction, health point / time unit
        :param land_penalty_rate: float, the penalty of health point, if the fish's body covers land pixels (1s) in
                                  the terrain map, health point / (pixel * time unit)
        :param food_rate: float, the gaining of health point if fish's body covers food pixels (1s) in the food map,
                          health point / pixel. the food after taken will disappear, so no health gaining is a
                          transient event
        """

        # print('\nFish: Creating littlefish.core.fish.Fish object.')

        if name is None:
            self._name = "test_fish"
        else:
            self._name = name

        if mother_name is None:
            self._mother_name = ""
        else:
            self._mother_name = mother_name

        self._max_health = float(max_health)
        self._health_decay_rate = float(health_decay_rate)
        self._land_penalty_rate = land_penalty_rate
        self._food_rate = food_rate
        self._move_penalty_rate = move_penalty_rate

        if brain is None:
            self._brain = Brain()
        else:
            brain.check_integrity(verbose=False)
            self._brain = brain

        # print('Fish: littlefish.core.fish.Fish object created successfully.')

    def copy(self):
        """

        :return: a copy of self for i.e. mutation
        """

        return Fish(
            name=self.get_name(),
            mother_name=self.get_mother_name(),
            brain=self.get_brain(),
            max_health=self.get_max_health(),
            health_decay_rate=self.get_health_decay_rate(),
            food_rate=self.get_food_rate(),
        )

    def get_name(self):
        return self._name

    @property
    def name(self):
        return self.get_name()

    def get_mother_name(self):
        return self._mother_name

    def get_brain(self):
        return self._brain

    def get_max_health(self):
        return self._max_health

    def get_health_decay_rate(self):
        return self._health_decay_rate

    def get_land_penalty_rate(self):
        return self._land_penalty_rate

    def get_food_rate(self):
        return self._food_rate

    def get_move_penalty_rate(self):
        return self._move_penalty_rate

    def set_name(self, name):
        self._name = name

    def set_brain(self, brain):
        brain.check_integrity(verbose=False)
        self._brain = brain

    def set_food_rate(self, food_rate):
        self._food_rate = float(food_rate)

    def set_health_decay_rate(self, health_decay_rate):
        self._health_decay_rate = health_decay_rate

    def set_move_penalty_rate(self, move_penalty_rate):
        self._move_penalty_rate = float(move_penalty_rate)

    def act(
        self,
        t_point,
        curr_position,
        curr_health,
        action_histories,
        psp_waveforms,
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
        :param action_histories: data frame of lists, each list is the action history of a particular neuron, in the
                                 same order as self._neurons data frame, columns = ['action_history'], used by
                                 self._brain.act()
        :param psp_waveforms: 2d-array of floats, psp waveforms of all neurons in the brain, each row represents one
                              neuron in the same order as self._neurons data frame, each column represents a time point,
                              used by self._brain.act()
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
        updated_health -= body_land_overlap * self._land_penalty_rate

        # ----------------- not implemented --------------------------
        # if fish_map is not None:
        #     self._eval_fish(fish_map=fish_map)
        # ----------------- not implemented --------------------------

        # update health
        updated_health -= self._health_decay_rate

        if updated_health > 0:  # still alive
            movement_attempt = self._brain.act(
                t_point=t_point,
                action_histories=action_histories,
                psp_waveforms=psp_waveforms,
                body_position=curr_position,
                terrain_map=terrain_map,
                food_map=food_map,
                fish_map=fish_map,
            )
        else:
            movement_attempt = None

        if movement_attempt is not None and any(movement_attempt):
            updated_health -= self._move_penalty_rate

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

        if body_food_overlap == 0:
            return curr_health
        else:
            updated_health = curr_health + self._food_rate * body_food_overlap
            if updated_health > self._max_health:
                updated_health = self._max_health
            return updated_health

    def to_h5_group(self, h5_grp):
        h5_grp.create_dataset("name", data=self._name)
        h5_grp.create_dataset("mother_name", data=self._mother_name)
        h5_grp.create_dataset("max_health", data=self._max_health)
        h5_grp.create_dataset("health_decay_rate_per_tu", data=self._health_decay_rate)
        h5_grp.create_dataset(
            "land_penalty_rate_per_pixel_tu", data=self._land_penalty_rate
        )
        h5_grp.create_dataset("food_rate_per_pixel", data=self._food_rate)
        h5_grp.create_dataset("move_penalty_rate", data=self._move_penalty_rate)
        brain_group = h5_grp.create_group("brain")
        self._brain.to_h5_group(brain_group)

    @staticmethod
    def from_h5_group(h5_grp):
        brain_grp = h5_grp["brain"]
        curr_brain = Brain.from_h5_group(brain_grp)
        curr_name = util.decode(h5_grp["name"][()])
        curr_mother_name = util.decode(h5_grp["mother_name"][()])
        curr_max_health = h5_grp["max_health"][()]
        curr_health_decay_rate = h5_grp["health_decay_rate_per_tu"][()]
        curr_land_penalty_rate = h5_grp["land_penalty_rate_per_pixel_tu"][()]
        curr_food_rate = h5_grp["food_rate_per_pixel"][()]
        curr_move_penalty_rate = h5_grp["move_penalty_rate"][()]

        curr_fish = Fish(
            name=curr_name,
            mother_name=curr_mother_name,
            brain=curr_brain,
            max_health=curr_max_health,
            health_decay_rate=curr_health_decay_rate,
            land_penalty_rate=curr_land_penalty_rate,
            food_rate=curr_food_rate,
            move_penalty_rate=curr_move_penalty_rate,
        )

        return curr_fish
