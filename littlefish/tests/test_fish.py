import os
import h5py
import random
import unittest
from littlefish.core.fish import (
    Fish,
    load_fish_from_h5_group,
    generate_fish_from_config,
)
import littlefish.core.utilities as util
import numpy as np


class TestFish(unittest.TestCase):
    def setup(self):
        pass

    def test_load_fish_from_config(self):
        import os
        import yaml

        curr_folder = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(curr_folder, "fish_config.yml")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        fish = generate_fish_from_config(
            fish_config=config["fish_config"],
            brain_config=config["brain_config"],
        )

        assert fish.name == "fish"
        assert fish.mother_name == "mother_fish"
        assert fish.max_health == 100.0
        assert fish.food_rate == 10.0
        assert fish.land_penalty_rate == 1.0
        assert fish.health_decay_rate == 0.05
        assert fish.move_penalty_rate == 0.0001
        assert fish.action_potential_penalty_rate == 0.000001
        assert np.array_equal(
            fish.brain.neurons.layer,
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
        )
        assert fish.brain.connections.shape == (96, 3)

    def test_io(self):
        curr_folder = os.path.dirname(os.path.abspath(__file__))
        temp_path = os.path.join(curr_folder, "temp_file.h5")

        if os.path.isfile(temp_path):
            os.remove(temp_path)

        fish = Fish(name="aa", mother_name="bb")
        f_temp = h5py.File(temp_path, "a")
        h5_grp = f_temp.create_group("fish")
        fish.to_h5_group(h5_group=h5_grp)
        fish2 = load_fish_from_h5_group(h5_group=h5_grp)

        assert fish2.name == "aa"
        assert fish2.mother_name == "bb"
        assert fish2.max_health == fish.max_health
        assert fish2.health_decay_rate == fish.health_decay_rate
        assert fish2.land_penalty_rate == fish.land_penalty_rate
        assert fish2.food_rate == fish.food_rate
        assert fish2.move_penalty_rate == fish.move_penalty_rate
        assert fish2.action_potential_penalty_rate == fish.action_potential_penalty_rate
        assert fish2.brain.neurons.shape == fish.brain.neurons.shape
        assert fish2.brain.connections.shape == fish.brain.connections.shape
        assert np.array_equal(fish2.brain.neurons.layer, fish.brain.neurons.layer)
        assert np.array_equal(
            fish2.brain.connections.pre_idx, fish.brain.connections.pre_idx
        )
        assert np.array_equal(
            fish2.brain.connections.post_idx, fish.brain.connections.post_idx
        )

        f_temp.close()
        os.remove(temp_path)

    def test_act(self):
        from littlefish.brain.functional import generate_brain_from_brain_config

        random.seed(42)
        np.random.seed(42)
        terrain_map = np.zeros((4, 5))
        terrain_map[3, 2] = 1
        terrain_map[0, 3] = 1
        food_map = np.zeros((4, 5))
        body_position = [2, 3]
        max_simulation_length = 20
        brain_config = {
            "eye_layer": {
                "eye_set": "ONE_EYE",
                "input_types": ["terrain", "food"],
            },
            "hidden_layers": {"neuron_nums": [2]},
            "muscle_layer": {"muscle_set": "ONE_MUSCLE"},
            "connection_0_1": {"connection_type": "full"},
            "connection_1_2": {"connection_type": "full"},
        }

        brain = generate_brain_from_brain_config(brain_config=brain_config)
        brain.neurons.loc[0, "neuron"].baseline_rate = 0.1
        brain.neurons.loc[1, "neuron"].baseline_rate = 0.1
        brain.neurons.loc[2, "neuron"].baseline_rate = 0.0
        brain.neurons.loc[3, "neuron"].baseline_rate = 0.0
        brain.neurons.loc[4, "neuron"].baseline_rate = 0.5
        brain.neurons.loc[0, "neuron"].gain = 0.9
        brain.connections.loc[0, "connection"].set_amplitude(1.0)
        brain.connections.loc[3, "connection"].set_amplitude(1.0)
        brain.connections.loc[4, "connection"].set_amplitude(0.5)
        brain.connections.loc[5, "connection"].set_amplitude(0.5)

        fish = Fish(
            brain=brain,
            max_health=20.0,
            health_decay_rate=0.01,
            land_penalty_rate=0.5,
            food_rate=20.0,
            move_penalty_rate=0.001,
            action_potential_penalty_rate=0.0001,
        )

        fish.initiate_simulation(
            position=body_position, max_simulation_length=max_simulation_length
        )

        movement_attempt, food_eaten = fish.act(
            t_point=0,
            terrain_map=terrain_map,
            food_map=food_map,
        )

        assert fish.brain.simulation_cache["action_histories"] == [[0], [0], [], [], []]
        assert np.allclose(
            fish.simulation_cache["health_history"][0:2], [20, 19.4898], atol=1e-8
        )
        assert np.array_equal(movement_attempt, [0, 0])
        assert food_eaten == 0

        food_map[1, 4] = 1
        food_map[0, 3] = 1
        terrain_map[3, 2] = 0
        fish.simulation_cache["position_history"][1] = body_position

        movement_attempt, food_eaten = fish.act(
            t_point=1,
            terrain_map=terrain_map,
            food_map=food_map,
        )
        assert fish.brain.simulation_cache["action_histories"] == [
            [0],
            [0],
            [],
            [],
            [1],
        ]
        assert np.allclose(
            fish.simulation_cache["health_history"][0:3],
            [20, 19.4898, 19.9889],
            atol=1e-8,
        )
        assert np.array_equal(movement_attempt, [1, 0])
        assert food_eaten == 1


if __name__ == "__main__":
    test_fish = TestFish()
    test_fish.test_io()
    test_fish.test_load_fish_from_config()
    test_fish.test_act()
